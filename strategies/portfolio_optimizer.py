"""
portfolio_optimizer.py — Asignación Dinámica Optimizada

Construye una matriz de covarianza rodante y utiliza scipy.optimize.minimize
para encontrar los pesos óptimos de portafolio que maximizan el Sharpe Ratio,
con restricciones (suma = 1, sin short, máximo 50% por activo).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger("alpaca_bot")


# ─── Paso 1: Matriz de Covarianza Rodante ───────────────────────────────────

def calcular_matriz_covarianza_rodante(dfs, ventana=252):
    """
    Construye la matriz de covarianza usando una ventana rodante.

    Parámetros
    ----------
    dfs : dict {str: pd.DataFrame}
        Diccionario con {símbolo: df_histórico}. Cada df debe tener columna 'close'.
    ventana : int
        Días hábiles para la ventana (252 = 1 año de trading aprox).

    Retorna
    -------
    matriz_cov : pd.DataFrame (n_activos × n_activos)
    retornos : pd.DataFrame
    """
    logger.info(f"📐 Construyendo matriz de covarianza rodante (ventana={ventana} días)...")

    # Unir precios de cierre en un solo DataFrame
    precios = pd.DataFrame({simbolo: df['close'] for simbolo, df in dfs.items()})

    # Calcular retornos diarios porcentuales
    retornos = precios.pct_change().dropna()

    # Solo los últimos `ventana` días
    retornos_recientes = retornos.tail(ventana)

    # Matriz de covarianza (annualizada: *252)
    matriz_cov = retornos_recientes.cov() * 252

    logger.info(f"   Matriz de covarianza: {matriz_cov.shape[0]}×{matriz_cov.shape[1]}")
    return matriz_cov, retornos_recientes


# ─── Paso 2: Probabilidad → Retorno Esperado ────────────────────────────────

def probabilidad_a_retorno_esperado(probabilidades, factor_escala=0.02):
    """
    Convierte las probabilidades de cada modelo (LSTM, ML, etc.)
    en un vector de retornos esperados anualizados.

    La fórmula mapea:
        prob=0.50 → retorno=0.0   (neutral)
        prob=0.75 → retorno=+2%   (~factor_escala)
        prob=0.25 → retorno=-2%   (~-factor_escala)

    Parámetros
    ----------
    probabilidades : dict {str: float}
        {símbolo: probabilidad_de_subida} — valor entre 0 y 1.
    factor_escala : float
        Factor de escala para convertir probabilidad a retorno.

    Retorna
    -------
    np.ndarray — Vector de retornos esperados (mismo orden que keys).
    """
    simbolos = list(probabilidades.keys())
    retornos = []

    for simb in simbolos:
        prob = probabilidades[simb]
        # Mapeo lineal: [0,1] → [-factor_escala, +factor_escala]
        retorno = (prob - 0.5) * 2 * factor_escala
        retornos.append(retorno)
        logger.debug(f"   {simb}: prob={prob:.4f} → retorno_esp={retorno:.4f}")

    logger.info(f"📊 Retornos esperados generados para {len(simbolos)} activos.")
    return np.array(retornos)


# ─── Paso 3: Optimización con scipy ─────────────────────────────────────────

def optimizar_portafolio(matriz_cov, retornos_esperados):
    """
    Usa scipy.optimize.minimize (método SLSQP) para encontrar los pesos
    óptimos w que maximizan el Ratio de Sharpe.

    Maximizar Sharpe = w·r / sqrt(w·Σ·w)

    Restricciones:
        • Σ w_i = 1          (capital total invertido)
        • w_i ≥ 0            (no short / sin ventas en corto)
        • w_i ≤ 0.5          (máximo 50% del portafolio en un solo activo)

    Parámetros
    ----------
    matriz_cov : pd.DataFrame o np.ndarray
        Matriz de covarianza anualizada (n × n).
    retornos_esperados : np.ndarray
        Vector de retornos esperados anualizados (n,).

    Retorna
    -------
    np.ndarray — Pesos óptimos w (n,). Sum(w) ≈ 1.0
    """
    n = len(retornos_esperados)
    logger.info(f"⚙️ Optimizando portafolio ({n} activos)...")

    # ── Función objetivo: Sharpe negativo (minimizar = maximizar Sharpe) ──
    def sharpe_negativo(pesos):
        retorno_port = np.dot(pesos, retornos_esperados)
        riesgo_port = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))
        if riesgo_port == 0:
            return 0.0
        return -retorno_port / riesgo_port

    # ── Punto de partida: equitativo ──
    w0 = np.ones(n) / n

    # ── Restricción: suma de pesos = 1 ──
    restricciones = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    # ── Límites: 0 ≤ w_i ≤ 0.5 ──
    limites = [(0.0, 0.5) for _ in range(n)]

    # ── Ejecutar optimización ──
    resultado = minimize(
        sharpe_negativo,
        w0,
        method='SLSQP',
        bounds=limites,
        constraints=restricciones,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if resultado.success:
        pesos = resultado.x
        sharpe_optimo = -resultado.fun
        logger.info(f"✅ Optimización exitosa — Sharpe óptimo: {sharpe_optimo:.4f}")
    else:
        logger.warning(f"⚠️ Optimización falló: {resultado.message}. Usando pesos equitativos.")
        pesos = np.ones(n) / n

    # Mostrar pesos finales
    for i in range(n):
        logger.info(f"   Peso {i+1}: {pesos[i]*100:.2f}%")

    return pesos


# ─── Paso 4: Ejecutar órdenes basadas en los pesos dinámicos ───────────────

def ejecutar_asignacion_dinamica(api, simbolos, pesos, efectivo_total, ultimos_precios):
    """
    Lee los pesos devueltos por el optimizador y ejecuta las órdenes en Alpaca.
    Si un activo tiene peso 0.0, no lo compra.
    Si ya tenemos una posición, la ajusta al nuevo peso objetivo.

    Parámetros
    ----------
    api : tradeapi.REST
    simbolos : list[str]
    pesos : np.ndarray
    efectivo_total : float
    ultimos_precios : dict {str: float}
    """
    logger.info("💼 Ejecutando asignación dinámica de capital...")
    logger.info(f"   Capital disponible: ${efectivo_total:.2f}")

    for i, simb in enumerate(simbolos):
        peso = pesos[i]
        precio = ultimos_precios.get(simb, 0)

        if peso <= 0.001:
            logger.info(f"   ⏭️  {simb}: peso={peso*100:.1f}% → Saltando (muy bajo o volátil)")
            # Si tenemos posición, vender
            try:
                pos = api.get_position(simb)
                qty_actual = float(pos.qty)
                if qty_actual > 0:
                    logger.info(f"      Vendiendo {qty_actual} acciones de {simb} (peso objetivo 0%)")
                    api.submit_order(symbol=simb, qty=qty_actual, side='sell', type='market', time_in_force='day')
            except:
                pass
            continue

        if precio <= 0:
            logger.warning(f"   ⚠️  {simb}: precio no disponible, saltando.")
            continue

        # Calcular monto objetivo para este activo
        monto_objetivo = efectivo_total * peso
        logger.info(f"   {simb}: peso={peso*100:.1f}% → ${monto_objetivo:.2f} a invertir")

        # Revisar posición actual
        qty_actual = 0
        try:
            pos = api.get_position(simb)
            qty_actual = float(pos.qty)
        except:
            pass

        valor_actual = qty_actual * precio
        diferencia = monto_objetivo - valor_actual

        if abs(diferencia) < 1.0:
            logger.info(f"      Sin cambios necesarios (diferencia < $1).")
            continue

        if diferencia > 0:
            # Comprar más
            logger.info(f"      Comprando ${diferencia:.2f} USD de {simb}...")
            api.submit_order(
                symbol=simb, notional=round(diferencia, 2),
                side='buy', type='market', time_in_force='day'
            )
        else:
            # Vender
            qty_vender = abs(diferencia) / precio
            qty_vender = min(qty_vender, qty_actual)  # no vender más de lo que tenemos
            if qty_vender > 0.001:
                logger.info(f"      Vendiendo {qty_vender:.4f} acciones de {simb}...")
                api.submit_order(
                    symbol=simb, qty=round(qty_vender, 4),
                    side='sell', type='market', time_in_force='day'
                )

    logger.info("✅ Asignación dinámica completada.\n")
