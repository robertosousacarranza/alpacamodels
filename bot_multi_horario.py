"""
bot_multi_horario.py — Bot Multi-Activo Intradía (Ciclo Rápido)

  • Rebalanceo cada 1 hora durante horario de mercado
  • Datos: velas de 15 minutos (entrenamiento) y ejecución en ventanas de 1h
  • Optimización: Kelly Criterion + Covarianza 60d + Volatilidad Objetivo
  • LSTM: modelo entrenado con datos intradía (15-min)
  • Full pipeline: DataSanity → Regímen → Predicción → Cov → Kelly → TransFilter → Ejecución

Diferencia clave con bot_multi_diario.py:
  • Escala temporal: 15m/1h vs 1D
  • Rebalanceo más frecuente
  • Modelos LSTM separados (entrenados con datos intradía)
  • Filtro de transacción más restrictivo (más trades → más costos)

Uso:
  python bot_multi_horario.py            # Un solo ciclo
  python bot_multi_horario.py --loop     # Loop infinito (rebalanceo cada 1h)
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
        logging.FileHandler(os.path.join(LOG_DIR, "bot_horario.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bot_horario")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── Config ─────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

SIMBOLOS = sorted(['SPY', 'AAPL', 'GLD', 'XOM'])
VENTANA_COVARIANZA = 60              # ~60 velas de 15min = 15h de trading
FACTOR_ESCALA_RETORNO = 0.03         # factor más conservador para intradía
FRACCION_KELLY = 0.40                # más conservador (40% invertido, 60% cash)
VOLATILIDAD_OBJETIVO = 0.12          # target más bajo (12%)
REBALANCEO_MINUTOS = 60              # cada 1 hora

# Timelframe para datos intradía
TIMEFRAME_INTRADIA = tradeapi.TimeFrame.Minute
TIMEFRAME_MINUTOS = 15               # velas de 15 minutos para features
DIAS_HISTORIA = 30                   # 30 días de velas de 15min ≈ 1920 velas

RUTA_MODELOS = os.path.join(DIR_PROYECTO, "strategies", "models")
BITACORA_PATH = os.path.join(DIR_PROYECTO, "data", "operaciones_horario.csv")
RESUMEN_PATH = os.path.join(DIR_PROYECTO, "data", "ultimo_ciclo_horario.json")

# Cache: no ejecutar más de una vez cada N minutos
_ultima_ejecucion = None


# ═════════════════════════════════════════════════════════════════════════════
#  MODELO LSTM (intradía)
# ═════════════════════════════════════════════════════════════════════════════

def cargar_modelo_lstm():
    """Carga modelo LSTM intradía (busca modelo horario primero, luego diario)."""
    modelo = None
    scaler = None

    # Prioridad 1: modelo específico para 15min
    rutas = [
        ("lstm_spy_intraday.keras", "scaler_spy_intraday.pkl"),
        ("lstm_spy_modelo.keras", "scaler_spy.pkl"),  # fallback al diario
    ]

    for nom_modelo, nom_scaler in rutas:
        modelo_path = os.path.join(RUTA_MODELOS, nom_modelo)
        scaler_path = os.path.join(RUTA_MODELOS, nom_scaler)
        if os.path.exists(modelo_path) and os.path.exists(scaler_path):
            try:
                modelo = load_model(modelo_path)
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Modelo LSTM (intradía): {modelo_path}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Error cargando {nom_modelo}: {e}")
                continue

    if modelo is None:
        logger.warning("⚠️ No hay modelo LSTM intradía. Se usará SMA fallback.")

    return modelo, scaler

modelo, scaler = cargar_modelo_lstm()


# ═════════════════════════════════════════════════════════════════════════════
#  DATOS (Intradía: velas de 15 minutos)
# ═════════════════════════════════════════════════════════════════════════════

def obtener_datos_intradia(simbolo, minutos=TIMEFRAME_MINUTOS,
                            dias=DIAS_HISTORIA, intentos=3):
    """
    Descarga velas intradía de un símbolo.
    Alpaca da velas de 1 minuto; las agregamos a `minutos` con resample.

    Si la API devuelve datos muy largos, se toman solo los últimos N días.
    """
    for intento in range(intentos):
        try:
            fecha_fin = datetime.now()
            fecha_ini = fecha_fin - timedelta(days=dias)

            # Alpaca: máximo 10000 velas por request para 1Min
            # 30 días × 6.5h × 60min = 11700 velas — está en el límite
            df = api.get_bars(
                simbolo, tradeapi.TimeFrame.Minute,
                start=fecha_ini.strftime('%Y-%m-%d'),
                end=fecha_fin.strftime('%Y-%m-%d'),
                feed='iex', adjustment='all'
            ).df

            if df.empty:
                logger.warning(f"⚠️ {simbolo}: datos intradía vacíos")
                continue

            df.index = df.index.tz_localize(None)

            # Agregar a la resolución deseada (ej. 15 min)
            # Usamos resample: OHLC
            rule = f'{minutos}T'
            ohlc_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df_resampled = df.resample(rule).agg(ohlc_dict).dropna()

            logger.debug(f"   {simbolo}: {len(df)} velas 1Min → {len(df_resampled)} velas {minutos}min")
            return df_resampled

        except Exception as e:
            logger.warning(f"⚠️ {simbolo} intento {intento+1}/{intentos}: {e}")
            time.sleep(3)

    logger.error(f"❌ No se pudieron obtener datos intradía de {simbolo}")
    return None


def obtener_todos_los_datos():
    """Descarga datos intradía para todos los símbolos."""
    dfs = {}
    for simb in SIMBOLOS:
        df = obtener_datos_intradia(simb)
        if df is not None and len(df) > VENTANA_COVARIANZA:
            dfs[simb] = df
            logger.info(f"✅ {simb}: {len(df)} velas {TIMEFRAME_MINUTOS}min")
        else:
            n = len(df) if df is not None else 0
            logger.warning(f"⚠️ {simb}: datos insuficientes ({n})")
    if len(dfs) < 2:
        logger.error("❌ No hay suficientes activos con datos intradía.")
        return None
    return dfs


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICCIONES (intradía)
# ═════════════════════════════════════════════════════════════════════════════

def predecir_probabilidad_lstm(df):
    """Usa LSTM para predecir próxima vela. Features: open, high, low, close, volume, ret, vol."""
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
        logger.warning(f"⚠️ Error en predicción LSTM intradía: {e}")
        return None


def predecir_probabilidad_sma(df, corta=5, larga=20):
    """SMA rápida para datos intradía (5 y 20 velas de 15min)."""
    sma_c = df['close'].rolling(corta).mean()
    sma_l = df['close'].rolling(larga).mean()
    if pd.isna(sma_c.iloc[-1]) or pd.isna(sma_l.iloc[-1]):
        return 0.50
    senal = sma_c.iloc[-1] > sma_l.iloc[-1]
    return 0.60 if senal else 0.40


def obtener_probabilidades(dfs):
    """Probabilidades intradía."""
    probs = {}
    for simb, df in dfs.items():
        prob = None
        if simb == 'SPY' and modelo is not None:
            prob = predecir_probabilidad_lstm(df)
        if prob is None:
            prob = predecir_probabilidad_sma(df)
        probs[simb] = prob
        logger.info(f"🔮 {simb}: {prob*100:.1f}% (intradía)")
    return probs


# ═════════════════════════════════════════════════════════════════════════════
#  CICLO PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def ejecutar_ciclo():
    """Un ciclo completo del bot horario."""
    global _ultima_ejecucion

    ahora = datetime.now()

    # Evitar ejecuciones muy seguidas
    if _ultima_ejecucion is not None:
        diff_min = (ahora - _ultima_ejecucion).total_seconds() / 60
        if diff_min < 5:
            logger.info(f"⏸️  Último ciclo hace {diff_min:.0f}min. Mínimo 5min entre ciclos.")
            return
    _ultima_ejecucion = ahora

    logger.info("=" * 55)
    logger.info("🔄 CICLO HORARIO — Pipeline multi-activo intradía...")
    logger.info(f"📈 Activos: {', '.join(SIMBOLOS)}")
    logger.info(f"📐 Kelly: {FRACCION_KELLY*100:.0f}% | Vol target: {VOLATILIDAD_OBJETIVO*100:.0f}%")
    logger.info("=" * 55)

    # 1. Verificar pausa
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

    # 3. Datos intradía
    dfs = obtener_todos_los_datos()
    if dfs is None:
        return

    # 4. Sanity Check
    sanity = ejecutar_sanity_check(dfs)
    if not sanity['paso']:
        logger.error("❌ Sanity Check FALLÓ — abortando.")
        return

    # 5. Regímen
    regimen = evaluar_factor_riesgo(dfs)
    factor_riesgo = regimen['factor_riesgo']

    # 6. Predicciones
    probabilidades = obtener_probabilidades(dfs)

    # 7. Pipeline Kelly completo
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

    # 8. Guardar resumen
    os.makedirs(os.path.dirname(RESUMEN_PATH), exist_ok=True)
    try:
        with open(RESUMEN_PATH, 'w') as f:
            json.dump(resumen, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo guardar resumen: {e}")

    trades_hoy = obtener_estado_trades_hoy()
    logger.info(f"📊 Trades hoy: {trades_hoy}")
    logger.info("✅ CICLO HORARIO COMPLETADO.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  HORARIO
# ═════════════════════════════════════════════════════════════════════════════

def en_horario_mercado():
    """Lun–vie, 9:30–16:00 ET ≈ 13:30–20:00 UTC en verano.
    Usamos un margen amplio: 13:00–21:00 UTC."""
    ahora = datetime.now()
    if ahora.weekday() >= 5:
        return False
    return 13 <= ahora.hour <= 21


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    modo_loop = '--loop' in sys.argv

    logger.info(f"🚀 BOT HORARIO INICIADO — "
                f"{'Loop infinito (cada ' + str(REBALANCEO_MINUTOS) + 'min)' if modo_loop else 'Un solo ciclo'}")

    if modo_loop:
        while True:
            try:
                if en_horario_mercado():
                    ejecutar_ciclo()
                else:
                    logger.info("🌙 Fuera de horario intradía. Esperando...")

                logger.info(f"😴 Próximo ciclo en {REBALANCEO_MINUTOS}min...")
                time.sleep(REBALANCEO_MINUTOS * 60)

            except KeyboardInterrupt:
                logger.info("🛑 Bot detenido por el usuario.")
                break
            except Exception as e:
                logger.error(f"💥 Error inesperado: {e}")
                time.sleep(300)
    else:
        ejecutar_ciclo()