"""
retrain_pipeline.py — Pipeline de Re-entrenamiento Semestral LSTM

Propósito:
  Mantener el cerebro del bot actualizado. Cada 6 meses este script:
    1. Descarga 5 años de datos frescos de Yahoo Finance
    2. Entrena un nuevo modelo LSTM desde cero
    3. Backtestea el modelo nuevo contra el actual en datos OOS
    4. Solo guarda el nuevo modelo si SUPERA al actual en métricas estrictas
    5. Envía reporte por Telegram con resultados

Ejecución:
  python retrain_pipeline.py                    # Un solo ciclo
  python retrain_pipeline.py --force             # Forzar retrain (ignora fecha)
  python retrain_pipeline.py --ticker AAPL       # Entrenar para un ticker específico

Configuración de cron (cada 6 meses el día 1):
  0 0 1 1,7 * cd /home/roberto/proyectos/alpacamodels && python retrain_pipeline.py >> logs/retrain.log 2>&1
"""

import os
import sys
import json
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ─── ML ─────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ─── Yahoo Finance (yfinance para datos independientes de Alpaca) ───────────
try:
    import yfinance as yf
    YFINANCE_DISPONIBLE = True
except ImportError:
    YFINANCE_DISPONIBLE = False
    logging.warning("⚠️ yfinance no instalado — pip install yfinance")

# ─── Configuración ──────────────────────────────────────────────────────────
DIR_PROYECTO = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(DIR_PROYECTO, "logs")
RUTA_MODELOS = os.path.join(DIR_PROYECTO, "strategies", "models")
RUTA_REPORTES = os.path.join(DIR_PROYECTO, "data", "retrain")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RUTA_MODELOS, exist_ok=True)
os.makedirs(RUTA_REPORTES, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "retrain.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("retrain")

load_dotenv()

# ─── Parámetros de entrenamiento ────────────────────────────────────────────
SIMBOLOS_DIARIO = ['SPY', 'AAPL', 'GLD', 'XOM']
ANIOS_HISTORIA = 5               # 5 años de datos
VENTANA_TEMPORAL = 15            # 15 velas para predecir la siguiente
EPOCHS = 30
BATCH_SIZE = 32
TEST_SPLIT = 0.80                # 80% train, 20% test

# Para backtesting: ventana OOS dentro de los datos de test
DIAS_OOS = 180                   # 6 meses de out-of-sample

# Guardar solo si mejora por lo menos este margen
MEJORA_MINIMA_ACCURACY = 0.01    # +1% de accuracy
MEJORA_MINIMA_LOG_LOSS = -0.05   # log loss más bajo

# Path de estado (para saber cuándo fue el último retrain)
ESTADO_PATH = os.path.join(RUTA_REPORTES, "estado_retrain.json")

# Telegram
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


def enviar_telegram(mensaje):
    """Envía reporte por Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info(f"📱 [Telegram no configurado] {mensaje[:100]}...")
        return
    try:
        import urllib.request
        datos = json.dumps({
            'chat_id': TELEGRAM_CHAT_ID,
            'text': f"🤖 *Re-Train LSTM* 🤖\n\n{mensaje}",
            'parse_mode': 'Markdown'
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data=datos, headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=10):
            logger.info("📱 Reporte enviado por Telegram")
    except Exception as e:
        logger.warning(f"⚠️ Telegram falló: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  DESCARGA DE DATOS (Yahoo Finance)
# ═════════════════════════════════════════════════════════════════════════════

def descargar_datos_yahoo(simbolo, anios=ANIOS_HISTORIA):
    """
    Descarga datos históricos de Yahoo Finance.
    Independiente de Alpaca — fuente de datos neutral para re-entrenamiento.
    """
    if not YFINANCE_DISPONIBLE:
        logger.error("❌ yfinance no instalado. Usa 'pip install yfinance'")
        return None

    try:
        fecha_fin = datetime.now()
        fecha_ini = fecha_fin - timedelta(days=anios * 365)

        ticker = yf.Ticker(simbolo)
        df = ticker.history(start=fecha_ini.strftime('%Y-%m-%d'),
                            end=fecha_fin.strftime('%Y-%m-%d'))

        if df.empty:
            logger.error(f"❌ {simbolo}: Yahoo devolvió DataFrame vacío")
            return None

        # Renombrar columnas a minúsculas (consistencia con Alpaca)
        df.columns = [c.lower() for c in df.columns]

        # Eliminar zonas horarias del index
        df.index = pd.to_datetime(df.index).tz_localize(None)

        logger.info(f"📥 {simbolo}: {len(df)} filas descargadas (Yahoo)")
        return df

    except Exception as e:
        logger.error(f"❌ Error descargando {simbolo} de Yahoo: {e}")
        return None


def guardar_datos_csv(df, simbolo):
    """Guarda los datos descargados para referencia."""
    path = os.path.join(RUTA_REPORTES, f"{simbolo}_historial.csv")
    try:
        df.to_csv(path)
        logger.info(f"💾 Datos guardados: {path}")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo guardar CSV: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO LSTM
# ═════════════════════════════════════════════════════════════════════════════

def crear_secuencias(datos, etiquetas, pasos_temporales=VENTANA_TEMPORAL):
    """Transforma datos 2D en tensores 3D (muestras, pasos, features)."""
    X, y = [], []
    for i in range(len(datos) - pasos_temporales):
        X.append(datos[i:(i + pasos_temporales)])
        y.append(etiquetas[i + pasos_temporales])
    return np.array(X), np.array(y)


def preparar_datos(df):
    """
    Prepara el DataFrame para entrenamiento LSTM.
    Features: open, high, low, close, volume, Retorno, Volatilidad
    Target: 1 si sube mañana, 0 si baja o se mantiene.
    """
    df = df.copy()
    df = df.sort_index()

    # Features derivadas
    df['Retorno'] = df['close'].pct_change()
    df['Volatilidad'] = df['Retorno'].rolling(window=10).std()

    # Target: predicción binaria
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    df.dropna(inplace=True)

    features = ['open', 'high', 'low', 'close', 'volume', 'Retorno', 'Volatilidad']

    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(df[features])
    etiquetas = df['Target'].values

    return datos_escalados, etiquetas, scaler, df


def construir_modelo(input_shape):
    """Arquitectura LSTM probada (2 capas con Dropout)."""
    modelo = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])
    return modelo


def entrenar_lstm(df, simbolo):
    """
    Entrena un modelo LSTM completo.

    Retorna
    -------
    dict : {
        'modelo': modelo Keras,
        'scaler': MinMaxScaler,
        'accuracy': float,
        'log_loss': float,
        'n_train': int,
        'n_test': int
    }
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"🎯 ENTRENANDO LSTM para {simbolo}")
    logger.info('='*50)

    # 1. Preparar datos
    datos, etiquetas, scaler, df = preparar_datos(df)
    logger.info(f"   Datos preparados: {len(datos)} muestras, {datos.shape[1]} features")

    # 2. Crear secuencias
    X, y = crear_secuencias(datos, etiquetas)
    logger.info(f"   Tensores: X={X.shape}, y={y.shape}")

    if len(X) < 100:
        logger.error(f"❌ {simbolo}: muy pocas muestras ({len(X)}). No se puede entrenar.")
        return None

    # 3. División cronológica
    corte = int(len(X) * TEST_SPLIT)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]
    logger.info(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Construir y entrenar
    modelo = construir_modelo((X.shape[1], X.shape[2]))
    logger.info("   Entrenando...")

    historial = modelo.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0  # silencioso en logs
    )

    # 5. Evaluar
    y_pred_prob = modelo.predict(X_test, verbose=0).flatten()
    y_pred_cls = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_cls)
    ll = log_loss(y_test, y_pred_prob)

    # Últimas épocas para log
    loss_final = historial.history['loss'][-1]
    val_loss_final = historial.history['val_loss'][-1]

    logger.info(f"   ✅ {simbolo} — "
                f"Accuracy={acc*100:.2f}% | "
                f"LogLoss={ll:.4f} | "
                f"TrainLoss={loss_final:.4f} | "
                f"ValLoss={val_loss_final:.4f}")

    return {
        'modelo': modelo,
        'scaler': scaler,
        'accuracy': acc,
        'log_loss': ll,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


# ═════════════════════════════════════════════════════════════════════════════
#  BACKTESTING COMPARATIVO
# ═════════════════════════════════════════════════════════════════════════════

def backtestear_modelo(df, modelo_nuevo, scaler_nuevo, modelo_actual, scaler_actual,
                        simbolo, ventana_oos=DIAS_OOS):
    """
    Backtestea el modelo nuevo vs el actual en una ventana OOS.

    Retorna
    -------
    dict : resultados comparativos
    """
    logger.info(f"\n📊 Backtesting OOS ({ventana_oos} días) para {simbolo}...")

    # Tomar los últimos ventana_oos días como OOS
    datos, etiquetas, _, _ = preparar_datos(df)
    X, y = crear_secuencias(datos, etiquetas)

    if len(X) <= ventana_oos:
        logger.warning(f"⚠️ No hay suficientes datos para OOS ({len(X)} ≤ {ventana_oos})")
        # Usar el 20% final como OOS
        ventana_oos = int(len(X) * 0.2)
        logger.info(f"   Usando {ventana_oos} muestras como OOS")

    X_oos = X[-ventana_oos:]
    y_oos = y[-ventana_oos:]

    if len(X_oos) < 10:
        logger.warning(f"⚠️ Muy pocas muestras OOS ({len(X_oos)}). Saltando backtest.")
        return None

    # Predecir con modelo nuevo
    try:
        y_prob_nuevo = modelo_nuevo.predict(X_oos, verbose=0).flatten()
        y_cls_nuevo = (y_prob_nuevo > 0.5).astype(int)
        acc_nuevo = accuracy_score(y_oos, y_cls_nuevo)
        ll_nuevo = log_loss(y_oos, y_prob_nuevo)
    except Exception as e:
        logger.error(f"❌ Error en predicción del modelo nuevo: {e}")
        return None

    # Predecir con modelo actual
    acc_actual = 0.0
    ll_actual = float('inf')
    if modelo_actual is not None and scaler_actual is not None:
        try:
            # Escalar con el scaler del modelo actual
            features = ['open', 'high', 'low', 'close', 'volume', 'Retorno', 'Volatilidad']
            df_temp = df.tail(ventana_oos + VENTANA_TEMPORAL + 10).copy()
            df_temp['Retorno'] = df_temp['close'].pct_change()
            df_temp['Volatilidad'] = df_temp['Retorno'].rolling(window=10).std()
            df_temp.dropna(inplace=True)

            if len(df_temp) >= VENTANA_TEMPORAL + ventana_oos:
                escalados = scaler_actual.transform(df_temp[features])
                X_actual, y_actual = crear_secuencias(
                    escalados,
                    np.where(df_temp['close'].shift(-1) > df_temp['close'], 1, 0).values
                )
                X_oos_actual = X_actual[-ventana_oos:] if len(X_actual) >= ventana_oos else X_actual
                y_oos_actual = y_actual[-ventana_oos:] if len(y_actual) >= ventana_oos else y_actual

                if len(X_oos_actual) > 0:
                    y_prob_actual = modelo_actual.predict(X_oos_actual, verbose=0).flatten()
                    y_cls_actual = (y_prob_actual > 0.5).astype(int)
                    acc_actual = accuracy_score(y_oos_actual, y_cls_actual)
                    ll_actual = log_loss(y_oos_actual, y_prob_actual)
        except Exception as e:
            logger.warning(f"⚠️ Error backtesteando modelo actual: {e}")

    resultados = {
        'modelo_nuevo': {
            'accuracy': acc_nuevo,
            'log_loss': ll_nuevo
        },
        'modelo_actual': {
            'accuracy': acc_actual,
            'log_loss': ll_actual
        },
        'diferencia_accuracy': acc_nuevo - acc_actual,
        'diferencia_log_loss': ll_nuevo - ll_actual,
        'n_muestras_oos': len(X_oos)
    }

    logger.info(f"   Modelo NUEVO: Accuracy={acc_nuevo*100:.2f}%, LogLoss={ll_nuevo:.4f}")
    logger.info(f"   Modelo ACTUAL: Accuracy={acc_actual*100:.2f}%, LogLoss={ll_actual:.4f}")
    logger.info(f"   Diferencia: ΔAcc={acc_nuevo-acc_actual:+.4f}, ΔLL={ll_nuevo-ll_actual:+.4f}")

    return resultados


# ═════════════════════════════════════════════════════════════════════════════
#  GUARDAR MODELO
# ═════════════════════════════════════════════════════════════════════════════

def guardar_modelo(modelo, scaler, simbolo, resultados_backtest):
    """Guarda modelo solo si pasa los thresholds de mejora."""
    nombre_base = f"lstm_{simbolo.lower()}_modelo"

    # Verificar si hay modelo actual
    modelo_actual_path = os.path.join(RUTA_MODELOS, f"{nombre_base}.keras")
    scaler_actual_path = os.path.join(RUTA_MODELOS, f"scaler_{simbolo.lower()}.pkl")

    modelo_actual_existe = os.path.exists(modelo_actual_path)

    # Decidir si guardar
    guardar = False
    razon = ""

    if not modelo_actual_existe:
        # No hay modelo anterior → guardar
        guardar = True
        razon = "Primer modelo — guardando"
    elif resultados_backtest is None:
        # No se pudo backtestear → guardar con precaución
        guardar = True
        razon = "Backtest no disponible — guardando con precaución"
    else:
        diff_acc = resultados_backtest['diferencia_accuracy']
        diff_ll = resultados_backtest['diferencia_log_loss']
        acc_nuevo = resultados_backtest['modelo_nuevo']['accuracy']

        if diff_acc >= MEJORA_MINIMA_ACCURACY or diff_ll <= MEJORA_MINIMA_LOG_LOSS:
            guardar = True
            razon = (f"Mejora detectada: ΔAcc={diff_acc:+.4f} "
                     f"(umbral={MEJORA_MINIMA_ACCURACY:+.4f})")
        elif acc_nuevo > 0.52:
            # Accuracy decente aunque no mejore al anterior → guardar como respaldo
            guardar = True
            razon = f"Accuracy aceptable ({acc_nuevo*100:.1f}%) — guardando respaldo"
        else:
            guardar = False
            razon = (f"No mejora al modelo actual "
                     f"(ΔAcc={diff_acc:+.4f}, mínimo={MEJORA_MINIMA_ACCURACY:+.4f})")

    if guardar:
        # Path definitivo
        modelo_path = os.path.join(RUTA_MODELOS, f"{nombre_base}.keras")
        scaler_path = os.path.join(RUTA_MODELOS, f"scaler_{simbolo.lower()}.pkl")

        modelo.save(modelo_path)
        joblib.dump(scaler, scaler_path)

        logger.info(f"✅ MODELO GUARDADO: {modelo_path}")
        logger.info(f"   Razón: {razon}")

        return {
            'guardado': True,
            'razon': razon,
            'modelo_path': modelo_path,
            'scaler_path': scaler_path
        }
    else:
        logger.info(f"⏭️  Modelo NO guardado: {razon}")
        return {
            'guardado': False,
            'razon': razon,
            'modelo_path': None,
            'scaler_path': None
        }


# ═════════════════════════════════════════════════════════════════════════════
#  ESTADO DEL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def cargar_estado():
    """Carga el estado del último retrain."""
    if os.path.exists(ESTADO_PATH):
        try:
            with open(ESTADO_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def guardar_estado(estado):
    """Guarda el estado del retrain."""
    try:
        with open(ESTADO_PATH, 'w') as f:
            json.dump(estado, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo guardar estado: {e}")


def debe_retrain(forzar=False):
    """
    Decide si es momento de re-entrenar.
    Retorna True si:
      - `--force` está activo
      - No hay registro de retrain previo
      - Pasaron más de 180 días desde el último retrain exitoso
    """
    if forzar:
        logger.info("⏩ Modo FORCE: re-entrenando sin importar fecha.")
        return True

    estado = cargar_estado()
    ultimo_retrain = estado.get('ultimo_retrain')

    if ultimo_retrain is None:
        logger.info("🆕 Primer retrain — no hay registro previo.")
        return True

    ultima_fecha = datetime.fromisoformat(ultimo_retrain)
    diff_dias = (datetime.now() - ultima_fecha).days

    if diff_dias >= 180:
        logger.info(f"📅 Último retrain: {ultima_fecha.date()} ({diff_dias} días atrás). "
                    "Re-entrenando...")
        return True

    logger.info(f"⏸️  Último retrain hace {diff_dias} días. Próximo en {180-diff_dias} días.")
    return False


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def ejecutar_retrain(forzar=False, ticker_especifico=None):
    """
    Ejecuta el pipeline completo de re-entrenamiento.

    Parámetros
    ----------
    forzar : bool
        Ignorar fecha del último retrain.
    ticker_especifico : str | None
        Entrenar solo para un ticker (ej. 'AAPL').
    """
    logger.info("=" * 55)
    logger.info("🧠 PIPELINE DE RE-ENTRENAMIENTO LSTM")
    logger.info(f"📅 {datetime.now().isoformat()}")
    logger.info("=" * 55)

    if not YFINANCE_DISPONIBLE:
        logger.error("❌ yfinance no instalado. pip install yfinance")
        return

    if not debe_retrain(forzar=forzar):
        logger.info("✅ No es necesario re-entrenar todavía.")
        return

    tickers = [ticker_especifico] if ticker_especifico else SIMBOLOS_DIARIO
    resultados_generales = {
        'ultimo_retrain': datetime.now().isoformat(),
        'resultados': {}
    }

    for simbolo in tickers:
        logger.info(f"\n{'─'*50}")
        logger.info(f"📈 Procesando {simbolo}...")

        # 1. Descargar datos
        df = descargar_datos_yahoo(simbolo)
        if df is None:
            logger.error(f"❌ No se pudieron obtener datos para {simbolo}")
            continue

        guardar_datos_csv(df, simbolo)

        # 2. Entrenar modelo nuevo
        resultado_entrenamiento = entrenar_lstm(df, simbolo)
        if resultado_entrenamiento is None:
            logger.error(f"❌ Entrenamiento falló para {simbolo}")
            continue

        # 3. Cargar modelo actual (si existe)
        modelo_actual = None
        scaler_actual = None
        nombre_base = f"lstm_{simbolo.lower()}_modelo"
        modelo_path = os.path.join(RUTA_MODELOS, f"{nombre_base}.keras")
        scaler_path = os.path.join(RUTA_MODELOS, f"scaler_{simbolo.lower()}.pkl")

        if os.path.exists(modelo_path):
            try:
                modelo_actual = load_model(modelo_path)
                scaler_actual = joblib.load(scaler_path)
                logger.info(f"   Modelo actual cargado: {modelo_path}")
            except Exception as e:
                logger.warning(f"   No se pudo cargar modelo actual: {e}")

        # 4. Backtest comparativo
        resultados_bt = backtestear_modelo(
            df,
            resultado_entrenamiento['modelo'],
            resultado_entrenamiento['scaler'],
            modelo_actual, scaler_actual,
            simbolo
        )

        # 5. Decidir guardado
        resultado_guardado = guardar_modelo(
            resultado_entrenamiento['modelo'],
            resultado_entrenamiento['scaler'],
            simbolo,
            resultados_bt
        )

        # 6. Registrar resultados
        resultados_generales['resultados'][simbolo] = {
            'accuracy': resultado_entrenamiento['accuracy'],
            'log_loss': resultado_entrenamiento['log_loss'],
            'n_train': resultado_entrenamiento['n_train'],
            'n_test': resultado_entrenamiento['n_test'],
            'backtest': resultados_bt,
            'guardado': resultado_guardado
        }

        # Limpiar memoria
        import gc
        gc.collect()

    # 7. Guardar estado global
    guardar_estado(resultados_generales)

    # 8. Reporte final
    reporte = f"📊 *Reporte de Re-Entrenamiento*\n"
    reporte += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    for simb, res in resultados_generales['resultados'].items():
        reporte += f"*{simb}:*\n"
        reporte += f"  Accuracy: {res['accuracy']*100:.1f}%\n"
        reporte += f"  LogLoss: {res['log_loss']:.4f}\n"
        if res['guardado']['guardado']:
            reporte += f"  ✅ Guardado: {res['guardado']['razon']}\n"
        else:
            reporte += f"  ⏭️  No guardado: {res['guardado']['razon']}\n"
        if res.get('backtest') and res['backtest']:
            diff_acc = res['backtest']['diferencia_accuracy']
            reporte += f"  ΔAccuracy: {diff_acc:+.2%}\n"

    logger.info(f"\n{'='*55}")
    logger.info(reporte)
    logger.info('='*55)

    enviar_telegram(reporte)

    return resultados_generales


if __name__ == "__main__":
    forzar = '--force' in sys.argv
    ticker = None

    for arg in sys.argv:
        if arg.startswith('--ticker='):
            ticker = arg.split('=')[1].upper()

    logger.info(f"🚀 RETRAIN PIPELINE — forzar={forzar}, ticker={ticker or 'todos'}")

    try:
        ejecutar_retrain(forzar=forzar, ticker_especifico=ticker)
    except KeyboardInterrupt:
        logger.info("🛑 Retrain interrumpido por el usuario.")
    except Exception as e:
        logger.error(f"💥 Error en retrain: {e}", exc_info=True)