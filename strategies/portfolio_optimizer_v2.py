"""
portfolio_optimizer_v2.py — Optimización Dinámica: Markowitz + Kelly Criterion

Mejora fundamental sobre v1:
  • Maximiza la tasa de crecimiento esperada (Kelly) en vez de solo Sharpe
  • Matriz de covarianza RODANTE de 60 días (mucho más reactiva que 252)
  • Fracción Kelly configurable → el resto del capital queda en cash
  • Escala posiciones por la volatilidad actual de cada activo (risk-parity dinámico)
  • La señal LSTM se integra directamente como retorno esperado en la optimización

Arquitectura:
  bot → LSTM(prob) ──┐
  bot → Cov 60d   ──┼──→ portfolio_optimizer_v2 → pesos dinámicos → Alpaca
  bot → Regímen   ──┘
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

logger = logging.getLogger("alpaca_bot")

# ─── Configuración global ───────────────────────────────────────────────────
VENTANA_COVARIANZA = 60            # 60 días hábiles (~3 meses)
FRACCIÓN_KELLY = 0.50              # 50% de Kelly → mitad en cash
MAX_PESO_POR_ACTIVO = 0.50         # máximo 50% del portafolio en un activo
VOLATILIDAD_OBJETIVO = 0.15        # target 15% anualizado para el portafolio


# ═════════════════════════════════════════════════════════════════════════════
#  PASO 1: MATRIZ DE COVARIANZA RODANTE (60 DÍAS)
# ═════════════════════════════════════════════════════════════════════════════

def calcular_matriz_covarianza_rodante(dfs, ventana=VENTANA_COVARIANZA):
    """
    Construye la matriz de covarianza usando los últimos `ventana` días.

    ⚡ Ventana corta (60d) para capturar la estructura de correlación RECIENTE.
    Anualizada (×252) para consistencia con los retornos esperados anualizados.

    Parámetros
    ----------
    dfs : dict {str: pd.DataFrame}
    ventana : int

    Retorna
    -------
    matriz_cov : pd.DataFrame (n × n)
    retornos_df : pd.DataFrame
    """
    logger.info(f"📐 Matriz de covarianza rodante ({ventana} días)...")

    precios = pd.DataFrame({simb: df['close'] for simb, df in dfs.items()
                            if df is not None and not df.empty})
    retornos = precios.pct_change().dropna()
    retornos_recientes = retornos.tail(ventana)

    if len(retornos_recientes) < 10:
        logger.warning(f"⚠️ Muy pocos datos para covarianza: {len(retornos_recientes)} días")
        # Fallback: todo lo que tengamos
        retornos_recientes = retornos.tail(max(10, len(retornos)))

    matriz_cov = retornos_recientes.cov() * 252

    logger.info(f"   Covarianza: {matriz_cov.shape[0]}×{matriz_cov.shape[1]}")
    return matriz_cov, retornos_recientes


# ═════════════════════════════════════════════════════════════════════════════
#  PASO 2: PROBABILIDAD LSTM → RETORNO ESPERADO
# ═════════════════════════════════════════════════════════════════════════════

def probabilidad_a_retorno_esperado(probabilidades, volatilidades=None, factor_escala=0.04):
    """
    Convierte la probabilidad LSTM en retorno esperado anualizado.

    v2: Ahora escala el retorno por la volatilidad del activo.
    Activos más volátiles → retorno esperado más alto (o baja más pronunciada).

    Fórmula:
      señal = (prob - 0.5) * 2            → [-1, +1]
      retorno = señal * factor_escala * (vol_i / vol_promedio)

    Esto le dice al optimizador: "Si hay señal fuerte en un activo volátil,
    el retorno esperado es alto, pero la covarianza también lo penalizará".

    Parámetros
    ----------
    probabilidades : dict {str: float}
    volatilidades : dict {str: float} | None
        Volatilidad anualizada de cada activo. Si None, usa factor_escala puro.
    factor_escala : float
        Factor base (default 4% → ~2% por desviación estándar de señal)

    Retorna
    -------
    np.ndarray : Vector de retornos esperados anualizados.
    """
    simbolos = list(probabilidades.keys())
    retornos = []

    if volatilidades and len(volatilidades) > 0:
        vol_promedio = np.mean(list(volatilidades.values()))
        if vol_promedio <= 0:
            vol_promedio = 1.0
    else:
        vol_promedio = 1.0

    for simb in simbolos:
        prob = probabilidades[simb]
        # Señal bipolar: [-1, +1]
        senal = (prob - 0.5) * 2.0

        # Ajuste por volatilidad del activo
        vol_ajuste = 1.0
        if volatilidades and simb in volatilidades and vol_promedio > 0:
            vol_ajuste = volatilidades[simb] / vol_promedio

        retorno = senal * factor_escala * vol_ajuste
        retornos.append(retorno)

        logger.debug(f"   {simb}: prob={prob:.3f} → señal={senal:.3f} → "
                     f"retorno_esp={retorno*100:.2f}% (vol_ajuste={vol_ajuste:.2f})")

    logger.info(f"📊 Retornos esperados (Kelly-input): {len(simbolos)} activos")
    return np.array(retornos)


# ═════════════════════════════════════════════════════════════════════════════
#  PASO 3: OPTIMIZACIÓN CON KELLY CRITERION
# ═════════════════════════════════════════════════════════════════════════════

def optimizar_portafolio_kelly(matriz_cov, retornos_esperados,
                                fraccion_kelly=FRACCIÓN_KELLY,
                                max_peso=MAX_PESO_POR_ACTIVO):
    """
    MAXIMIZA LA TASA DE CRECIMIENTO ESPERADA (Kelly Criterion).

    Objetivo:
      Maximizar  G(w) = μ·w - 0.5·w'Σw

    donde:
      μ = vector de retornos esperados (de las LSTM)
      Σ = matriz de covarianza (60 días rodante)
      w = vector de pesos a optimizar

    Esto es el CRITERIO DE KELLY para múltiples activos correlacionados.
    La solución analítica sin restricciones es w* = Σ⁻¹μ.

    Aplicamos:
      1. Kelly analítico (si la matriz es invertible)
      2. Proyección a restricciones: no short, max peso, suma = fracción
      3. Fracción Kelly: el capital restante (1-fracción) queda en CASH
         como reserva de riesgo (Kelly fraccionado)

    Parámetros
    ----------
    matriz_cov : np.ndarray (n × n)
    retornos_esperados : np.ndarray (n,)
    fraccion_kelly : float (0.0-1.0)
        Fracción del capital a invertir. El resto queda en cash.
        Kelly completo = 1.0 (agresivo), Kelly 0.5 = mitad en cash (recomendado)
    max_peso : float
        Máximo peso por activo individual.

    Retorna
    -------
    np.ndarray : Pesos óptimos w (n,). Sum(w) = fraccion_kelly.
    """
    n = len(retornos_esperados)
    logger.info(f"⚙️ Optimización Kelly ({n} activos, fracción={fraccion_kelly*100:.0f}%)...")

    # ─── Estrategia 1: Kelly analítico (si la matriz es invertible) ────
    w_kelly = None
    try:
        # w* = Σ⁻¹ · μ  (Kelly analítico)
        cov_inv = np.linalg.inv(matriz_cov.values.astype(np.float64)
                                if hasattr(matriz_cov, 'values')
                                else np.array(matriz_cov, dtype=np.float64))
        mu = np.array(retornos_esperados, dtype=np.float64)
        w_kelly = cov_inv @ mu

        logger.info(f"   Kelly analítico: w_raw = {w_kelly}")

    except np.linalg.LinAlgError as e:
        logger.warning(f"⚠️ Matriz de covarianza singular ({e}). "
                       "Usando optimización numérica directa.")

    # ─── Aplicar restricciones ─────────────────────────────────────────
    if w_kelly is not None:
        # No short: recortar negativos a 0
        w_kelly = np.maximum(w_kelly, 0.0)

        # Max peso por activo
        w_kelly = np.minimum(w_kelly, max_peso)

        # Normalizar a la fracción Kelly
        suma = w_kelly.sum()
        if suma > 0:
            w_kelly = w_kelly / suma * fraccion_kelly
        else:
            logger.warning("⚠️ Kelly analítico: todos los pesos son 0. "
                           "Usando optimización numérica.")
            w_kelly = None

    # ─── Estrategia 2: Optimización numérica (fallback o si Kelly dió 0) ─
    if w_kelly is None:
        # Objetivo: Maximizar G(w) = μ·w - 0.5·w'Σw
        def crecimiento_negativo(pesos):
            ret = np.dot(pesos, retornos_esperados)
            riesgo = np.dot(pesos.T, np.dot(matriz_cov, pesos))
            return -(ret - 0.5 * riesgo)  # −G(w) para minimize

        # Punto de partida: distribución equitativa ajustada por fracción
        w0 = np.ones(n) / n * fraccion_kelly

        restricciones = {'type': 'eq', 'fun': lambda w: np.sum(w) - fraccion_kelly}
        limites = [(0.0, max_peso) for _ in range(n)]

        resultado = minimize(
            crecimiento_negativo,
            w0,
            method='SLSQP',
            bounds=limites,
            constraints=restricciones,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )

        if resultado.success:
            w_kelly = resultado.x
            g_optimo = -resultado.fun
            logger.info(f"✅ Optimización Kelly (numérica) — G(w)={g_optimo:.6f}")
        else:
            logger.warning(f"⚠️ Optimización Kelly falló: {resultado.message}. "
                           "Usando Markowitz como fallback.")
            return optimizar_portafolio_sharpe(matriz_cov, retornos_esperados, max_peso)

    # Mostrar pesos
    for i in range(n):
        logger.info(f"   Peso Kelly {i+1}: {w_kelly[i]*100:.2f}%")

    return w_kelly


# ═════════════════════════════════════════════════════════════════════════════
#  PASO 3b: MARKOWITZ (FALLBACK)
# ═════════════════════════════════════════════════════════════════════════════

def optimizar_portafolio_sharpe(matriz_cov, retornos_esperados,
                                 max_peso=MAX_PESO_POR_ACTIVO):
    """
    Optimización clásica de Markowitz: maximiza Sharpe Ratio.
    Usado como fallback si Kelly falla.
    """
    n = len(retornos_esperados)
    logger.info("⚙️ Fallback: Optimización Markowitz (Sharpe)...")

    def sharpe_negativo(pesos):
        ret = np.dot(pesos, retornos_esperados)
        riesgo = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))
        if riesgo == 0:
            return 0.0
        return -ret / riesgo

    w0 = np.ones(n) / n
    restricciones = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    limites = [(0.0, max_peso) for _ in range(n)]

    resultado = minimize(
        sharpe_negativo, w0,
        method='SLSQP',
        bounds=limites, constraints=restricciones,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if resultado.success:
        pesos = resultado.x
        sharpe = -resultado.fun
        logger.info(f"✅ Markowitz — Sharpe={sharpe:.4f}")
    else:
        logger.warning(f"⚠️ Markowitz falló: {resultado.message}. Pesos equitativos.")
        pesos = np.ones(n) / n

    for i in range(n):
        logger.info(f"   Peso Sharpe {i+1}: {pesos[i]*100:.2f}%")

    return pesos


# ═════════════════════════════════════════════════════════════════════════════
#  PASO 4: ESCALAR POR VOLATILIDAD (POSITION SIZING DINÁMICO)
# ═════════════════════════════════════════════════════════════════════════════

def escalar_por_volatilidad(pesos, matriz_cov, retornos_esperados,
                             vol_objetivo=VOLATILIDAD_OBJETIVO):
    """
    Escala los pesos para que la volatilidad esperada del portafolio
    se ajuste a un objetivo.

    Si la volatilidad actual del portafolio es 20% y el objetivo es 15%,
    todos los pesos se reducen por 15/20 = 0.75.

    Esto implementa VOLATILIDAD OBJETIVO DINÁMICA:
      • Cuando el mercado está tranquilo → apalancamiento completo
      • Cuando el mercado está volátil → reducción automática de posiciones

    Parámetros
    ----------
    pesos : np.ndarray
        Pesos del portafolio.
    matriz_cov : np.ndarray
        Matriz de covarianza.
    retornos_esperados : np.ndarray
        Retornos esperados (solo para log).
    vol_objetivo : float
        Volatilidad objetivo anualizada.

    Retorna
    -------
    np.ndarray : Pesos escalados.
    """
    volatilidad_port = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))

    if volatilidad_port <= 0:
        logger.info("   Volatilidad del portafolio ≈ 0% → sin escalado")
        return pesos

    ratio_escalado = vol_objetivo / volatilidad_port
    ratio_escalado = min(2.0, max(0.1, ratio_escalado))  # clamp 0.1x a 2x

    pesos_escalados = pesos * ratio_escalado

    # Re-aplicar restricciones después del escalado
    pesos_escalados = np.maximum(pesos_escalados, 0.0)
    pesos_escalados = np.minimum(pesos_escalados, MAX_PESO_POR_ACTIVO)

    # Normalizar a la suma original (o a 1 si excede)
    suma_original = pesos.sum()
    suma_nueva = pesos_escalados.sum()
    if suma_nueva > 0:
        pesos_escalados = pesos_escalados / suma_nueva * suma_original

    volatilidad_final = np.sqrt(np.dot(pesos_escalados.T,
                                       np.dot(matriz_cov, pesos_escalados)))

    logger.info(f"📏 Vol scaling: {volatilidad_port*100:.1f}% → "
                f"{volatilidad_final*100:.1f}% (ratio={ratio_escalado:.2f})")

    return pesos_escalados


# ═════════════════════════════════════════════════════════════════════════════
#  PASO 5: EJECUTAR ÓRDENES DINÁMICAS
# ═════════════════════════════════════════════════════════════════════════════

def ejecutar_asignacion_dinamica(api, simbolos, pesos, efectivo_total,
                                  ultimos_precios, factor_riesgo=1.0,
                                  retornos_esperados=None):
    """
    Ejecuta las órdenes en Alpaca basadas en los pesos optimizados.

    v2: Aplica factor de riesgo (del regime detector) y escala posiciones
    según la volatilidad de cada activo individualmente.

    Parámetros
    ----------
    api : tradeapi.REST
    simbolos : list[str]
    pesos : np.ndarray
        Pesos del optimizador.
    efectivo_total : float
        Capital disponible.
    ultimos_precios : dict {str: float}
    factor_riesgo : float
        1.0 = riesgo completo, 0.5 = riesgo a la mitad.
    retornos_esperados : np.ndarray | None
        Para el filtro de transacción.
    """
    from strategies.transaction_filter import filtrar_operacion

    logger.info(f"💼 Ejecutando asignación dinámica...")
    logger.info(f"   Capital: ${efectivo_total:.2f} | Factor riesgo: {factor_riesgo:.2f}")

    # Aplicar factor de riesgo a todos los pesos
    pesos_riesgo = pesos * factor_riesgo

    for i, simb in enumerate(simbolos):
        peso = pesos_riesgo[i]
        precio = ultimos_precios.get(simb, 0)
        ret_esp = retornos_esperados[i] if retornos_esperados is not None else 0.0

        if precio <= 0:
            logger.warning(f"   ⚠️  {simb}: precio no disponible, saltando.")
            continue

        # Obtener posición actual
        qty_actual = 0
        try:
            pos = api.get_position(simb)
            qty_actual = float(pos.qty)
        except:
            pass

        valor_actual = qty_actual * precio
        monto_objetivo = efectivo_total * peso
        peso_actual = valor_actual / efectivo_total if efectivo_total > 0 else 0

        # ─── Filtro de transacción ──────────────────────────────────
        resultado_filtro = filtrar_operacion(
            simbolo=simb,
            peso_actual=peso_actual,
            peso_objetivo=peso,
            efectivo_total=efectivo_total,
            precio_actual=precio,
            retorno_esperado=ret_esp
        )

        if not resultado_filtro['permitido']:
            logger.info(f"   ⏭️  {simb}: {resultado_filtro['razon']}")
            continue

        # ─── Ejecutar orden ─────────────────────────────────────────
        monto_objetivo = resultado_filtro['monto_operacion']
        diferencia = monto_objetivo - valor_actual

        if abs(diferencia) < 1.0:
            logger.info(f"   ✅ {simb}: Sin cambios (diferencia < $1)")
            continue

        if diferencia > 0:
            logger.info(f"   🟢 {simb}: Comprando ${diferencia:.2f} USD...")
            api.submit_order(
                symbol=simb, notional=round(diferencia, 2),
                side='buy', type='market', time_in_force='day'
            )
        else:
            qty_vender = min(abs(diferencia) / precio, qty_actual)
            if qty_vender > 0.001:
                logger.info(f"   🔴 {simb}: Vendiendo {qty_vender:.4f} acc...")
                api.submit_order(
                    symbol=simb, qty=round(qty_vender, 4),
                    side='sell', type='market', time_in_force='day'
                )

    logger.info("✅ Asignación dinámica completada.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  PASO 6: PIPELINE COMPLETO (un solo llamado)
# ═════════════════════════════════════════════════════════════════════════════

def pipeline_optimizacion_completo(dfs, probabilidades, api, efectivo_disponible,
                                    ventana_cov=VENTANA_COVARIANZA,
                                    fraccion_kelly=FRACCIÓN_KELLY,
                                    vol_objetivo=VOLATILIDAD_OBJETIVO,
                                    factor_riesgo=1.0):
    """
    Ejecuta el pipeline completo:
      1. Matriz de covarianza rodante
      2. Probabilidades → retornos esperados (con ajuste de volatilidad)
      3. Optimización Kelly
      4. Escalado por volatilidad
      5. Ejecución de órdenes

    Parámetros
    ----------
    dfs : dict {str: pd.DataFrame}
    probabilidades : dict {str: float}
    api : tradeapi.REST
    efectivo_disponible : float
    ventana_cov : int
    fraccion_kelly : float
    vol_objetivo : float
    factor_riesgo : float

    Retorna
    -------
    dict : Resumen de la ejecución.
    """
    logger.info("🔄 Pipeline de optimización completo (Kelly + Vol Dinámica)...")

    # 1. Covarianza
    matriz_cov, retornos_df = calcular_matriz_covarianza_rodante(dfs, ventana_cov)

    # 2. Volatilidades de cada activo (para escalar retornos)
    volatilidades = {}
    for simb, df in dfs.items():
        if df is not None and not df.empty and 'close' in df.columns:
            ret = df['close'].tail(ventana_cov).pct_change().dropna()
            volatilidades[simb] = ret.std() * np.sqrt(252) if len(ret) > 0 else 0.2

    # 3. Probabilidades → retornos esperados (ajustados por vol)
    simbolos_ordenados = list(probabilidades.keys())
    retornos_esperados = probabilidad_a_retorno_esperado(
        probabilidades, volatilidades
    )

    # 4. Optimización Kelly
    pesos_kelly = optimizar_portafolio_kelly(
        matriz_cov, retornos_esperados,
        fraccion_kelly=fraccion_kelly
    )

    # 5. Escalar por volatilidad objetivo
    pesos_finales = escalar_por_volatilidad(
        pesos_kelly, matriz_cov, retornos_esperados,
        vol_objetivo=vol_objetivo
    )

    # 6. Últimos precios
    ultimos_precios = {}
    for simb, df in dfs.items():
        if df is not None and not df.empty:
            ultimos_precios[simb] = df['close'].iloc[-1]

    # 7. Resumen de asignación
    logger.info("=" * 50)
    logger.info(f"📋 ASIGNACIÓN KELLY ({fraccion_kelly*100:.0f}% fracción, "
                f"riesgo={factor_riesgo*100:.0f}%):")
    for i, simb in enumerate(simbolos_ordenados):
        monto = efectivo_disponible * pesos_finales[i] * factor_riesgo
        logger.info(f"   {simb}: {pesos_finales[i]*100:5.1f}% → ${monto:>8.2f}")
    logger.info("=" * 50)

    # 8. Ejecutar
    ejecutar_asignacion_dinamica(
        api, simbolos_ordenados, pesos_finales,
        efectivo_disponible, ultimos_precios,
        factor_riesgo=factor_riesgo,
        retornos_esperados=retornos_esperados
    )

    resumen = {
        'simbolos': simbolos_ordenados,
        'pesos_finales': pesos_finales.tolist(),
        'retornos_esperados': retornos_esperados.tolist(),
        'factor_riesgo': factor_riesgo,
        'fraccion_kelly': fraccion_kelly,
        'vol_objetivo': vol_objetivo,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    return resumen