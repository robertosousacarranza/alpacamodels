"""
bot_multi_diario.py — Bot Multi-Activo Diario (Ciclo Lento)

  • Rebalanceo cada 6 horas durante horario de mercado
  • Datos: velas diarias (1D)
  • Optimización: Kelly Criterion + Covarianza 60d + Volatilidad Objetivo
  • LSTM: modelo entrenado con datos diarios
  • Full pipeline: DataSanity → Regímen → Predicción → Cov → Kelly → TransFilter → Ejecución

Uso:
  python bot_multi_diario.py            # Un solo ciclo
  python bot_multi_diario.py --loop     # Loop infinito (rebalanceo cada 6h)
"""

import os
import sys
import time
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from tensorflow.keras.models import load_model

# ─── Componentes propios ────────────────────────────────────────────────────
from strategies.portfolio_optimizer_v2 import (
    pipeline_optimizacion_completo,
    calcular_matriz_covarianza_rodante,
    probabilidad_a_retorno_esperado,
)
from strategies.regime_detector import evaluar_factor_riesgo
from strategies.data_sanity import ejecutar_sanity_check, verificar_pausa
from strategies.transaction_filter import resetear_contador, obtener_estado_trades_hoy

# ─── Logging ────────────────────────────────────────────────────────────────
DIR_PROYECTO = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(DIR_PROYECTO, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "bot_diario.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bot_diario")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── Config ─────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

SIMBOLOS = sorted(['SPY', 'AAPL', 'GLD', 'XOM'])
VENTANA_COVARIANZA = 60
FACTOR_ESCALA_RETORNO = 0.04
FRACCION_KELLY = 0.50
VOLATILIDAD_OBJETIVO = 0.15
REBALANCEO_HORAS = 6             # cada 6h en horario de mercado

RUTA_MODELOS = os.path.join(DIR_PROYECTO, "strategies", "models")
BITACORA_PATH = os.path.join(DIR_PROYECTO, "data", "operaciones_diario.csv")
RESUMEN_PATH = os.path.join(DIR_PROYECTO, "data", "ultimo_ciclo_diario.json")


# ═════════════════════════════════════════════════════════════════════════════
#  MODELO LSTM
# ═════════════════════════════════════════════════════════════════════════════

def cargar_modelo_lstm():
    """Carga el modelo LSTM entrenado con datos diarios."""
    modelo = None
    scaler = None
    try:
        modelo_path = os.path.join(RUTA_MODELOS, "lstm_spy_modelo.keras")
        scaler_path = os.path.join(RUTA_MODELOS, "scaler_spy.pkl")
        if os.path.exists(modelo_path) and os.path.exists(scaler_path):
            modelo = load_model(modelo_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"✅ Modelo LSTM (diario): {modelo_path}")
        else:
            logger.warning(f"⚠️ No se encontró modelo LSTM en {RUTA_MODELOS}")
    except Exception as e:
        logger.warning(f"⚠️ Error cargando LSTM: {e}")
    return modelo, scaler

modelo, scaler = cargar_modelo_lstm()


# ═════════════════════════════════════════════════════════════════════════════
#  DATOS (Diarios)
# ═════════════════════════════════════════════════════════════════════════════

def obtener_datos_historicos(simbolo, dias=500, intentos=3):
    """Descarga velas diarias de un símbolo con reintentos."""
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
    """Descarga datos diarios para todos los símbolos."""
    dfs = {}
    for simb in SIMBOLOS:
        df = obtener_datos_historicos(simb)
        if df is not None and len(df) > VENTANA_COVARIANZA:
            dfs[simb] = df
            logger.info(f"✅ {simb}: {len(df)} filas (diarias)")
        else:
            n = len(df) if df is not None else 0
            logger.warning(f"⚠️ {simb}: datos insuficientes ({n})")
    if len(dfs) < 2:
        logger.error("❌ No hay suficientes activos con datos.")
        return None
    return dfs


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICCIONES
# ═════════════════════════════════════════════════════════════════════════════

def predecir_probabilidad_lstm(df):
    """Usa LSTM para predecir probabilidad de subida en próxima vela diaria."""
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
        logger.warning(f"⚠️ Error en predicción LSTM diaria: {e}")
        return None


def predecir_probabilidad_sma(df, corta=20, larga=50):
    """Fallback SMA para datos diarios."""
    sma_c = df['close'].rolling(corta).mean()
    sma_l = df['close'].rolling(larga).mean()
    if pd.isna(sma_c.iloc[-1]) or pd.isna(sma_l.iloc[-1]):
        return 0.50
    senal = sma_c.iloc[-1] > sma_l.iloc[-1]
    return 0.65 if senal else 0.35


def obtener_probabilidades(dfs):
    """Probabilidades: LSTM para SPY, SMA para el resto."""
    probs = {}
    for simb, df in dfs.items():
        prob = None
        if simb == 'SPY' and modelo is not None:
            prob = predecir_probabilidad_lstm(df)
        if prob is None:
            prob = predecir_probabilidad_sma(df)
        probs[simb] = prob
        logger.info(f"🔮 {simb}: {prob*100:.1f}%")
    return probs


# ═════════════════════════════════════════════════════════════════════════════
#  CICLO PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def ejecutar_ciclo():
    """
    Un ciclo completo del bot diario:
    Sanity → Regímen → Predicción → Kelly/Cov → TransFilter → Ejecución
    """
    logger.info("=" * 55)
    logger.info("🔄 CICLO DIARIO — Iniciando pipeline multi-activo...")
    logger.info(f"📈 Activos: {', '.join(SIMBOLOS)}")
    logger.info(f"📐 Kelly: {FRACCION_KELLY*100:.0f}% | Vol target: {VOLATILIDAD_OBJETIVO*100:.0f}%")
    logger.info("=" * 55)

    # 1. Verificar si el bot está pausado por sanity
    if verificar_pausa():
        logger.error("🛑 BOT PAUSADO — Elimina .PAUSA para reanudar.")
        return

    # 2. Conectar a Alpaca
    try:
        api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
        cuenta = api.get_account()
        logger.info(f"✅ Alpaca: status={cuenta.status}, cash=${float(cuenta.cash):.2f}")
        efectivo = float(cuenta.cash)
    except Exception as e:
        logger.error(f"❌ Error conectando con Alpaca: {e}")
        return

    # 3. Descargar datos
    dfs = obtener_todos_los_datos()
    if dfs is None:
        return

    # 4. Data Sanity Check (valida precios antes de cualquier cálculo)
    sanity = ejecutar_sanity_check(dfs)
    if not sanity['paso']:
        logger.error("❌ Data Sanity Check FALLÓ — abortando ciclo.")
        return

    # 5. Detector de regímenes de mercado
    regimen = evaluar_factor_riesgo(dfs)
    factor_riesgo = regimen['factor_riesgo']

    # 6. Predicciones LSTM/SMA
    probabilidades = obtener_probabilidades(dfs)

    # 7. Pipeline completo de optimización (Kelly + Cov + Vol)
    resumen = pipeline_optimizacion_completo(
        dfs=dfs,
        probabilidades=probabilidades,
        api=api,
        efectivo_disponible=efectivo,
        ventana_cov=VENTANA_COVARIANZA,
        fraccion_kelly=FRACCION_KELLY,
        vol_objetivo=VOLATILIDAD_OBJETIVO,
        factor_riesgo=factor_riesgo
    )

    # 8. Guardar resumen del ciclo
    os.makedirs(os.path.dirname(RESUMEN_PATH), exist_ok=True)
    try:
        with open(RESUMEN_PATH, 'w') as f:
            json.dump(resumen, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo guardar resumen: {e}")

    trades_hoy = obtener_estado_trades_hoy()
    logger.info(f"📊 Trades hoy: {trades_hoy}")
    logger.info("✅ CICLO DIARIO COMPLETADO.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  HORARIO
# ═════════════════════════════════════════════════════════════════════════════

def en_horario_mercado():
    """Lun–vie, 6:00–21:00 UTC (~2:00–17:00 ET en verano)."""
    ahora = datetime.now()
    if ahora.weekday() >= 5:
        return False
    return 6 <= ahora.hour <= 21


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    modo_loop = '--loop' in sys.argv

    logger.info(f"🚀 BOT DIARIO INICIADO — "
                f"{'Loop infinito (cada ' + str(REBALANCEO_HORAS) + 'h)' if modo_loop else 'Un solo ciclo'}")

    if modo_loop:
        while True:
            try:
                if en_horario_mercado():
                    ejecutar_ciclo()
                else:
                    logger.info("🌙 Fuera de horario. Esperando...")

                logger.info(f"😴 Próximo ciclo en {REBALANCEO_HORAS}h...")
                time.sleep(REBALANCEO_HORAS * 3600)

            except KeyboardInterrupt:
                logger.info("🛑 Bot detenido por el usuario.")
                break
            except Exception as e:
                logger.error(f"💥 Error inesperado: {e}")
                time.sleep(300)
    else:
        ejecutar_ciclo()