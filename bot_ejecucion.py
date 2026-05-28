import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from tensorflow.keras.models import load_model

# Suprimir logs molestos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Configuración Inicial y Credenciales
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

SIMBOLO = 'SPY'
FRACCION_KELLY = 0.1138  # 11.38% de capital calculado matemáticamente

print("Iniciando Bot de Ejecución Autónoma... 🤖")

# 2. Conectar a la API y cargar el "Cerebro"
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

try:
    modelo = load_model('models/lstm_spy_modelo.keras')
    scaler = joblib.load('models/scaler_spy.pkl')
    print("✅ Modelo LSTM y Escalador cargados correctamente.")
except Exception as e:
    print(f"❌ Error al cargar los modelos. ¿Estás seguro de que están en la carpeta 'models/'? Detalle: {e}")
    exit()

def obtener_datos_recientes():
    """Descarga los últimos 40 días para tener suficiente margen para calcular la volatilidad y extraer los últimos 15 días limpios."""
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - timedelta(days=60) # Pedimos 60 días calendario para asegurar 40 días hábiles
    
    start_str = fecha_inicio.strftime('%Y-%m-%d')
    end_str = fecha_fin.strftime('%Y-%m-%d')
    
    df = api.get_bars(SIMBOLO, tradeapi.TimeFrame.Day, start=start_str, end=end_str, feed='iex', adjustment='all').df
    df.index = df.index.tz_localize(None) # Limpiar zona horaria para evitar warnings
    return df

def predecir_movimiento():
    df = obtener_datos_recientes()
    
    # Recrear la misma Ingeniería de Características del entrenamiento
    df['Retorno'] = df['close'].pct_change()
    df['Volatilidad'] = df['Retorno'].rolling(window=10).std()
    df.dropna(inplace=True)
    
    # Extraer EXACTAMENTE los últimos 15 días (nuestra ventana de tiempo / pasos temporales)
    ultimos_15_dias = df.tail(15).copy()
    
    features = ['open', 'high', 'low', 'close', 'volume', 'Retorno', 'Volatilidad']
    
    # Escalar los datos con los mismos "lentes" del entrenamiento
    datos_escalados = scaler.transform(ultimos_15_dias[features])
    
    # Transformar a Tensor 3D: (1 muestra, 15 pasos de tiempo, 7 variables)
    tensor_entrada = np.array([datos_escalados])
    
    # Hacer la predicción
    probabilidad = modelo.predict(tensor_entrada, verbose=0)[0][0]
    return probabilidad, ultimos_15_dias.iloc[-1]['close']

def ejecutar_estrategia():
    print(f"\nAnalizando el mercado para {SIMBOLO}...")
    probabilidad_subida, ultimo_precio = predecir_movimiento()
    
    print(f"Último precio de cierre: ${ultimo_precio:.2f}")
    print(f"Probabilidad de que el mercado SUBA mañana: {probabilidad_subida * 100:.2f}%")
    
    # Revisar estado de la cuenta
    cuenta = api.get_account()
    efectivo_disponible = float(cuenta.cash)
    
    # Revisar si ya tenemos acciones compradas de este símbolo
    posicion_actual = 0
    try:
        posicion = api.get_position(SIMBOLO)
        posicion_actual = float(posicion.qty)
    except:
        pass # Si lanza error, significa que no tenemos ninguna posición abierta
        
    print("-" * 40)
    
    # Lógica de Decisión Autónoma
    if probabilidad_subida > 0.50:
        print("📈 Señal: ALCISTA. Evaluando compra...")
        if posicion_actual == 0:
            # Calcular cuánto dinero usar según el Criterio de Kelly
            monto_a_invertir = efectivo_disponible * FRACCION_KELLY
            print(f"Ordenando compra por valor de ${monto_a_invertir:.2f} USD...")
            
            # Enviar orden de compra (usamos notional para comprar fracciones de acción exactas)
            api.submit_order(
                symbol=SIMBOLO,
                notional=monto_a_invertir,
                side='buy',
                type='market',
                time_in_force='day'
            )
            print("✅ ¡Orden de compra ejecutada en Alpaca!")
        else:
            print(f"Ya tienes una posición abierta de {posicion_actual} acciones. Manteniendo posición.")
            
    else:
        print("📉 Señal: BAJISTA. Evaluando venta...")
        if posicion_actual > 0:
            print(f"Vendiendo posición actual de {posicion_actual} acciones para proteger capital...")
            api.submit_order(
                symbol=SIMBOLO,
                qty=posicion_actual,
                side='sell',
                type='market',
                time_in_force='day'
            )
            print("✅ ¡Orden de venta (liquidación) ejecutada en Alpaca!")
        else:
            print("No tienes posiciones abiertas. Manteniéndonos en efectivo por seguridad.")
            
    print("=" * 40)
    print("Operación finalizada. El bot volverá a dormir.")

if __name__ == "__main__":
    ejecutar_estrategia()
