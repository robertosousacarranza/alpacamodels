"""
regime_detector.py — Detector de Regímenes de Mercado

Identifica periodos de volatilidad extrema para reducir riesgo automáticamente.
Combina tres métricas independientes:
  1. Realized Volatility (ventana rodante 21 días ≈ 1 mes)
  2. Desviación respecto a la media histórica de la volatilidad
  3. Drawdown máximo reciente (21 días)

Output: factor_riesgo ∈ [0.5, 1.0]
  • 1.0  → régimen normal, riesgo completo
  • 0.5  → volatilidad extrema, reducir riesgo a la mitad
  • (valores interpolados para transiciones suaves)
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("alpaca_bot")

# ─── Thresholds calibrados para SPY/AAPL/GLD/XOM ───────────────────────────
# Percentil 90 de volatilidad anualizada histórica para activos individuales
# SPY ~20-25%, AAPL ~30-40%, GLD ~15-20%, XOM ~30-35%
# Usamos un threshold unificado conservador: 35% anualizado
UMBRAL_VOLATILIDAD_ALTA = 0.35      # 35% anualizado → alerta
UMBRAL_VOLATILIDAD_EXTREMA = 0.50   # 50% anualizado → reducir riesgo
VENTANA_VOLATILIDAD = 21            # ~1 mes de trading
VENTANA_DRAWDOWN = 21
FRACCIÓN_SEGURA = 0.5               # riesgo mínimo en modo extremo


def calcular_realized_volatility(df_close, ventana=VENTANA_VOLATILIDAD):
    """
    Calcula la volatilidad realizada anualizada.

    Parámetros
    ----------
    df_close : pd.Series
        Serie de precios de cierre.
    ventana : int
        Días hábiles para la ventana rodante.

    Retorna
    -------
    float : volatilidad anualizada (último valor disponible).
    """
    if len(df_close) < ventana + 2:
        return None

    retornos = df_close.pct_change().dropna()
    vol_diaria = retornos.tail(ventana).std()
    vol_anualizada = vol_diaria * np.sqrt(252)

    return vol_anualizada


def calcular_drawdown_reciente(df_close, ventana=VENTANA_DRAWDOWN):
    """
    Calcula el drawdown máximo en los últimos N días.

    Retorna
    -------
    float : drawdown como proporción (ej. 0.08 = 8% de caída).
    """
    if len(df_close) < ventana:
        return 0.0

    reciente = df_close.tail(ventana)
    pico_max = reciente.cummax()
    drawdowns = (reciente - pico_max) / pico_max
    max_drawdown = abs(drawdowns.min())

    return max_drawdown


def evaluar_factor_riesgo(dfs):
    """
    Evalúa el régimen de mercado y retorna un factor de riesgo.
    Itera sobre todos los activos y toma el más conservador.

    Parámetros
    ----------
    dfs : dict {str: pd.DataFrame}
        Diccionario con {símbolo: df_histórico}.

    Retorna
    -------
    dict : {
        'factor_riesgo': float (0.5-1.0),
        'regimen': str,
        'vol_max': float,
        'drawdown_max': float,
        'detalle': str
    }
    """
    factores = []
    vols = []
    dds = []

    for simb, df in dfs.items():
        if 'close' not in df.columns:
            continue

        vol = calcular_realized_volatility(df['close'])
        dd = calcular_drawdown_reciente(df['close'])

        if vol is None:
            continue

        vols.append(vol)
        dds.append(dd)

        # Factor individual por activo
        if vol >= UMBRAL_VOLATILIDAD_EXTREMA or dd > 0.15:
            factores.append(FRACCIÓN_SEGURA)
        elif vol >= UMBRAL_VOLATILIDAD_ALTA or dd > 0.10:
            # Interpolación lineal entre 0.5 y 1.0 según qué tan lejos de normal
            exceso = (vol - UMBRAL_VOLATILIDAD_ALTA) / (UMBRAL_VOLATILIDAD_EXTREMA - UMBRAL_VOLATILIDAD_ALTA)
            exceso = min(1.0, max(0.0, exceso))
            factor = 1.0 - exceso * (1.0 - FRACCIÓN_SEGURA)
            factores.append(factor)
        else:
            factores.append(1.0)

    if not factores:
        logger.warning("⚠️ No se pudo evaluar régimen — datos insuficientes.")
        return {
            'factor_riesgo': 1.0,
            'regimen': 'desconocido',
            'vol_max': 0.0,
            'drawdown_max': 0.0,
            'detalle': 'Sin datos para evaluar régimen'
        }

    factor_global = min(factores)  # El más conservador
    vol_max = max(vols) if vols else 0.0
    dd_max = max(dds) if dds else 0.0

    if factor_global <= FRACCIÓN_SEGURA:
        regimen = 'VOLATILIDAD_EXTREMA'
        detalle = f"Vol máx={vol_max*100:.1f}%, Drawdown={dd_max*100:.1f}% → Riesgo={factor_global*100:.0f}%"
    elif factor_global < 0.85:
        regimen = 'VOLATILIDAD_ELEVADA'
        detalle = f"Vol máx={vol_max*100:.1f}% → Riesgo={factor_global*100:.0f}%"
    else:
        regimen = 'NORMAL'
        detalle = f"Vol máx={vol_max*100:.1f}% → Riesgo completo"

    logger.info(f"🌡️ Régimen: {regimen} | {detalle}")

    return {
        'factor_riesgo': factor_global,
        'regimen': regimen,
        'vol_max': vol_max,
        'drawdown_max': dd_max,
        'detalle': detalle
    }