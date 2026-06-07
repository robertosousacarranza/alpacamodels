"""
bot_ejecucion.py — Bot de Ejecución Autónoma con Asignación Dinámica Optimizada

Loop continuo con:
  • Logging profesional a archivo + terminal
  • Bitácora de operaciones en CSV
  • Reintentos automáticos en fallos de API
  • Verificación de horario de mercado
  • Asignación dinámica con Matriz de Covarianza Rodante + optimización SciPy
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from tensorflow.keras.models import load_model

# ─── Importar el optimizador de portafolio ───────────────────────────────────
from strategies.portfolio_optimizer import (
    calcular_matriz_covarianza_rodante,
    probabilidad_a_retorno_esperado,
    optimizar_portafolio,
    ejecutar_asignacion_dinamica,
)

# ─── Configuración de Logging ───────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alpaca_bot")

# Suprimir logs molestos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── Configuración Inicial ──────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# ─── Multi-activo ───────────────────────────────────────────────────────────
SIMBOLOS = ['SPY', 'AAPL', 'GLD', 'XOM']
SIMBOLOS.sort()  # orden consistente para evitar sorpresas

VENTANA_COVARIANZA = 252     # 1 año de trading
FACTOR_ESCALA_RETORNO = 0.02  # 2% base para mapeo prob→retorno

RUTA_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "strategies", "models")

# ─── Bitácora de operaciones ────────────────────────────────────────────────
BITACORA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "operaciones.csv")


def registrar_operacion(tipo, simbolo, cantidad, precio, monto, razon):
    os.makedirs(os.path.dirname(BITACORA_PATH), exist_ok=True)
    registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": tipo, "simbolo": simbolo,
        "cantidad": cantidad, "precio": precio,
        "monto": monto, "razon": razon,
    }
    df_nuevo = pd.DataFrame([registro])
    try:
        df_existente = pd.read_csv(BITACORA_PATH)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    except FileNotFoundError:
        df_final = df_nuevo
    df_final.to_csv(BITACORA_PATH, index=False)
    logger.info(f"📝 Operación registrada: {tipo} {simbolo}")


# ─── Conexión Alpaca ────────────────────────────────────────────────────────
logger.info("=" * 55)
logger.info("🤖 BOT DE EJECUCIÓN AUTÓNOMA — ASIGNACIÓN DINÁMICA")
logger.info(f"📈 Activos: {', '.join(SIMBOLOS)}")
logger.info("=" * 55)

try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
    cuenta = api.get_account()
    logger.info(f"✅ Conectado a Alpaca — Cuenta: {cuenta.status}")
except Exception as e:
    logger.error(f"❌ No se pudo conectar con Alpaca: {e}")
    exit(1)

# ─── Cargar modelo LSTM (SPY) ──────────────────────────────────────────────
modelo = None
scaler = None
try:
    modelo_path = os.path.join(RUTA_MODELOS, "lstm_spy_modelo.keras")
    scaler_path = os.path.join(RUTA_MODELOS, "scaler_spy.pkl")
    modelo = load_model(modelo_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"✅ Modelo LSTM cargado desde: {modelo_path}")
except Exception as e:
    logger.warning(f"⚠️ No se pudo cargar el modelo LSTM: {e}")
    logger.warning("   Se usará predicción por SMA como fallback.")


# ─── Datos ──────────────────────────────────────────────────────────────────
def obtener_datos_historicos(simbolo, dias=400, intentos=3):
    """Descarga datos históricos de un símbolo con reintentos."""
    for intento in range(intentos):
        try:
            fecha_fin = datetime.now()
            fecha_ini = fecha_fin - timedelta(days=dias)
            df = api.get_bars(
                simbolo, tradeapi.TimeFrame.Day,
                start=fecha_ini.strftime('%Y-%m-%d'),
                end=fecha_fin.strftime('%Y-%m-%d'),
                feed='iex', adjustment='all'
            ).df
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.warning(f"⚠️ {simbolo} intento {intento+1}/{intentos}: {e}")
            time.sleep(3)
    logger.error(f"❌ No se pudieron obtener datos de {simbolo}")
    return None


def obtener_todos_los_datos():
    """Descarga datos para todos los símbolos y retorna un dict {simb: df}."""
    dfs = {}
    for simb in SIMBOLOS:
        df = obtener_datos_historicos(simb)
        if df is not None and len(df) > VENTANA_COVARIANZA:
            dfs[simb] = df
            logger.info(f"✅ {simb}: {len(df)} filas descargadas")
        else:
            logger.warning(f"⚠️ {simb}: datos insuficientes ({len(df) if df is not None else 0})")
    if len(dfs) < 2:
        logger.error("❌ No hay suficientes activos con datos para optimizar.")
        return None
    return dfs


# ─── Predicciones ───────────────────────────────────────────────────────────
def predecir_probabilidad_lstm(df):
    """Usa el modelo LSTM entrenado (solo para SPY) para predecir prob. de subida."""
    if modelo is None or scaler is None:
        return None
    try:
        df_pred = df.copy()
        df_pred['Retorno'] = df_pred['close'].pct_change()
        df_pred['Volatilidad'] = df_pred['Retorno'].rolling(window=10).std()
        df_pred.dropna(inplace=True)
        if len(df_pred) < 15:
            return None
        ultimos = df_pred.tail(15).copy()
        features = ['open', 'high', 'low', 'close', 'volume', 'Retorno', 'Volatilidad']
        escalados = scaler.transform(ultimos[features])
        tensor = np.array([escalados])
        prob = modelo.predict(tensor, verbose=0)[0][0]
        return float(prob)
    except Exception as e:
        logger.warning(f"⚠️ Error en predicción LSTM: {e}")
        return None


def predecir_probabilidad_sma(df, corta=20, larga=50):
    """
    Fallback simple basado en cruce de medias móviles.
    Si SMA(corta) > SMA(larga) → prob > 0.5 (alcista).
    """
    sma_c = df['close'].rolling(corta).mean()
    sma_l = df['close'].rolling(larga).mean()
    if pd.isna(sma_c.iloc[-1]) or pd.isna(sma_l.iloc[-1]):
        return 0.50
    señal = sma_c.iloc[-1] > sma_l.iloc[-1]
    # Mapeamos: 1 → 0.65, 0 → 0.35 (nunca extremos)
    return 0.65 if señal else 0.35


def obtener_probabilidades(dfs):
    """
    Obtiene probabilidades para todos los símbolos.
    Usa LSTM para SPY (si está disponible) y SMA para el resto.
    """
    probs = {}
    for simb, df in dfs.items():
        prob = None
        if simb == 'SPY' and modelo is not None:
            prob = predecir_probabilidad_lstm(df)
        if prob is None:
            prob = predecir_probabilidad_sma(df)
        probs[simb] = prob
        logger.info(f"🔮 {simb}: probabilidad de subida = {prob*100:.1f}%")
    return probs


# ─── Ejecución del Ciclo Optimizado ─────────────────────────────────────────
def ejecutar_ciclo_optimizado():
    """Un ciclo completo: datos → predicciones → covarianza → optimizar → ejecutar."""
    logger.info("🔄 Iniciando ciclo de asignación dinámica...")

    # 1. Descargar datos de todos los activos
    dfs = obtener_todos_los_datos()
    if dfs is None:
        return

    # 2. Obtener probabilidades de cada activo
    probabilidades = obtener_probabilidades(dfs)

    # 3. Construir Matriz de Covarianza Rodante
    matriz_cov, retornos_df = calcular_matriz_covarianza_rodante(dfs, VENTANA_COVARIANZA)

    # 4. Traducir probabilidades → retornos esperados
    retornos_esperados = probabilidad_a_retorno_esperado(probabilidades, FACTOR_ESCALA_RETORNO)

    # 5. Optimizar portafolio con scipy
    simbolos_ordenados = list(probabilidades.keys())
    pesos = optimizar_portafolio(matriz_cov.values, retornos_esperados)

    # 6. Obtener capital disponible
    try:
        cuenta = api.get_account()
        efectivo_disponible = float(cuenta.cash)
    except Exception as e:
        logger.error(f"❌ Error al consultar cuenta: {e}")
        return

    # 7. Obtener últimos precios
    ultimos_precios = {}
    for simb, df in dfs.items():
        ultimos_precios[simb] = df['close'].iloc[-1]

    # 8. Mostrar asignación final
    logger.info("=" * 50)
    logger.info("📋 ASIGNACIÓN OPTIMIZADA:")
    for i, simb in enumerate(simbolos_ordenados):
        monto = efectivo_disponible * pesos[i]
        logger.info(f"   {simb}: {pesos[i]*100:5.1f}% → ${monto:>8.2f}")
    logger.info("=" * 50)

    # 9. Ejecutar órdenes dinámicas
    ejecutar_asignacion_dinamica(api, simbolos_ordenados, pesos, efectivo_disponible, ultimos_precios)

    # 10. Registrar en bitácora resumen
    for i, simb in enumerate(simbolos_ordenados):
        monto = efectivo_disponible * pesos[i]
        registrar_operacion(
            "REBALANCEO", simb, pesos[i],
            ultimos_precios.get(simb, 0), monto,
            f"Peso óptimo: {pesos[i]*100:.1f}%"
        )

    logger.info("✅ Ciclo de asignación dinámica completado.\n")


# ─── Verificación de horario ────────────────────────────────────────────────
def en_horario_mercado():
    """Simplificado: lun-vie, 6:00-21:00 UTC (~2:00-17:00 ET en verano)."""
    ahora = datetime.now()
    if ahora.weekday() >= 5:
        return False
    return 6 <= ahora.hour <= 21


# ─── MAIN LOOP ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("🚀 BOT INICIADO — Modo Asignación Dinámica Optimizada")
    logger.info(f"   Activos: {', '.join(SIMBOLOS)}")
    logger.info(f"   Ventana Covarianza: {VENTANA_COVARIANZA} días")

    while True:
        try:
            if en_horario_mercado():
                ejecutar_ciclo_optimizado()
            else:
                logger.info("🌙 Fuera del horario de mercado. Esperando...")

            logger.info("😴 Durmiendo 30 minutos hasta el próximo ciclo...")
            time.sleep(1800)

        except KeyboardInterrupt:
            logger.info("🛑 Bot detenido por el usuario.")
            break
        except Exception as e:
            logger.error(f"💥 Error inesperado: {e}")
            logger.info("Reintentando en 5 minutos...")
            time.sleep(300)
