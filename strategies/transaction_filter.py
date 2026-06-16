"""
transaction_filter.py — Filtro de Costos de Transacción

Previene el overtrading bloqueando operaciones que:
  1. Tienen un tamaño menor al umbral mínimo (cambios < 0.5% del portafolio)
  2. Cuyo costo estimado (spread + comisión + slippage) supera el beneficio esperado
  3. Exceden el límite de trades por día

Esto protege el capital de la erosión por costos de transacción en cuentas
pequeñas (< $10K), donde el impacto relativo es mucho mayor.
"""

import logging
import numpy as np
from datetime import datetime, date

logger = logging.getLogger("alpaca_bot")

# ─── Configuración ──────────────────────────────────────────────────────────
COSTO_POR_TRADE = 0.001       # 0.1% estimado (slippage ≈ 0.3-0.5% en micro-caps)
UMBRAL_MINIMO_PESO_CAMBIO = 0.0005  # 0.05% mínimo de cambio en asignación
MAX_TRADES_POR_DIA = 8        # límite diario de órdenes
CAPITAL_REFERENCIA = 10000    # capital de referencia para calcular % real

# Cache de trades del día (se resetea al mediodía UTC)
_trades_hoy = 0
_fecha_hoy = None


def resetear_contador():
    """Reinicia el contador diario de trades."""
    global _trades_hoy, _fecha_hoy
    _trades_hoy = 0
    _fecha_hoy = date.today()
    logger.debug("🔄 Contador de trades reiniciado.")


def verificar_limite_diario():
    """
    Retorna True si aún no se ha excedido el límite diario de trades.
    """
    global _trades_hoy, _fecha_hoy

    hoy = date.today()
    if _fecha_hoy != hoy:
        resetear_contador()

    if _trades_hoy >= MAX_TRADES_POR_DIA:
        logger.warning(f"⏸️  Límite diario alcanzado ({MAX_TRADES_POR_DIA} trades). "
                       "No se ejecutarán más órdenes hoy.")
        return False
    return True


def registrar_trade():
    """Incrementa el contador de trades del día."""
    global _trades_hoy, _fecha_hoy
    hoy = date.today()
    if _fecha_hoy != hoy:
        resetear_contador()
    _trades_hoy += 1
    logger.debug(f"📊 Trade registrado: {_trades_hoy}/{MAX_TRADES_POR_DIA} hoy.")


def costo_estimado_operacion(monto, precio, spread_estimado=0.001):
    """
    Estima el costo total de una operación.

    Incluye:
      - Spread estimado (default 0.1% para ETFs/Liquid stocks)
      - Slippage (0.3-0.5% según liquidez del activo)
      - Comisión Alpaca ($0 para acciones/ETFs)

    Parámetros
    ----------
    monto : float
        Monto en USD de la operación.
    precio : float
        Precio actual del activo.
    spread_estimado : float
        Spread relativo estimado (default 0.1%).

    Retorna
    -------
    dict : {
        'costo_total': float,
        'costo_relativo': float,
        'slippage_estimado': float,
        'detalle': str
    }
    """
    slippage = monto * COSTO_POR_TRADE
    costo_spread = monto * spread_estimado
    costo_total = slippage + costo_spread
    costo_relativo = costo_total / monto if monto > 0 else 0

    return {
        'costo_total': round(costo_total, 2),
        'costo_relativo': round(costo_relativo * 100, 3),  # en %
        'slippage_estimado': round(slippage, 2),
        'detalle': f"Spread={spread_estimado*100:.1f}% + Slippage={COSTO_POR_TRADE*100:.1f}%"
    }


def filtrar_operacion(simbolo, peso_actual, peso_objetivo, efectivo_total,
                      precio_actual, retorno_esperado):
    """
    Filtro principal: decide si una operación debe ejecutarse o bloquearse.

    Reglas de bloqueo:
      1. Cambio de peso < umbral mínimo → muy pequeño, no vale la pena
      2. Costo estimado > beneficio esperado → pierde dinero
      3. Límite diario de trades alcanzado → ya operó suficiente hoy

    Parámetros
    ----------
    simbolo : str
    peso_actual : float
        Peso actual del activo en el portafolio (0-1).
    peso_objetivo : float
        Peso objetivo según optimización (0-1).
    efectivo_total : float
        Capital total disponible.
    precio_actual : float
        Precio actual del activo.
    retorno_esperado : float
        Retorno esperado del activo (anualizado, decimal).

    Retorna
    -------
    dict : {
        'permitido': bool,
        'razon': str,
        'monto_operacion': float,
        'costo_estimado': float
    }
    """
    cambio_peso = abs(peso_objetivo - peso_actual)

    # ─── Regla 1: Cambio mínimo ─────────────────────────────────────────
    if cambio_peso < UMBRAL_MINIMO_PESO_CAMBIO:
        return {
            'permitido': False,
            'razon': f"Cambio mínimo ({cambio_peso*100:.2f}%) < umbral ({UMBRAL_MINIMO_PESO_CAMBIO*100:.1f}%)",
            'monto_operacion': 0.0,
            'costo_estimado': 0.0
        }

    # ─── Regla 2: Costo vs beneficio ────────────────────────────────────
    monto_operacion = efectivo_total * cambio_peso
    costo = costo_estimado_operacion(monto_operacion, precio_actual)
    beneficio_esperado = monto_operacion * abs(retorno_esperado) * (1 / 252)  # retorno diario aprox

    if costo['costo_total'] > beneficio_esperado:
        return {
            'permitido': False,
            'razon': (f"Costo ${costo['costo_total']:.2f} > "
                      f"Beneficio esperado ${beneficio_esperado:.2f}"),
            'monto_operacion': monto_operacion,
            'costo_estimado': costo['costo_total']
        }

    # ─── Regla 3: Límite diario ─────────────────────────────────────────
    if not verificar_limite_diario():
        return {
            'permitido': False,
            'razon': "Límite diario de trades alcanzado",
            'monto_operacion': monto_operacion,
            'costo_estimado': costo['costo_total']
        }

    # ─── Todo bien, operación permitida ──────────────────────────────────
    registrar_trade()
    logger.info(f"✅ {simbolo}: Cambio={cambio_peso*100:.2f}% | "
                f"Monto=${monto_operacion:.2f} | Costo=${costo['costo_total']:.2f} | "
                f"Beneficio_esp=${beneficio_esperado:.2f}")

    return {
        'permitido': True,
        'razon': "Operación rentable y dentro de límites",
        'monto_operacion': round(monto_operacion, 2),
        'costo_estimado': costo['costo_total']
    }

def obtener_estado_trades_hoy():
    """Retorna el número de trades ejecutados hoy."""
    global _trades_hoy, _fecha_hoy
    hoy = date.today()
    if _fecha_hoy != hoy:
        resetear_contador()
    return _trades_hoy
