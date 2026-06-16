import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
PORTAFOLIO = ['AAPL', 'JPM', 'XOM', 'GLD']
VENTANA_HORAS = 15

def obtener_datos_horarios(simbolo, anos=3):
    print(f"[{simbolo}] Descargando datos históricos...")
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - timedelta(days=365 * anos)
    
    df = api.get_bars(
        simbolo, tradeapi.TimeFrame.Hour, 
        start=fecha_inicio.strftime('%Y-%m-%d'), end=fecha_fin.strftime('%Y-%m-%d'), 
        feed='iex', adjustment='all'
    ).df
    df.index = df.index.tz_localize(None)
    
    df['Retorno'] = df['close'].pct_change()
    df['Volatilidad'] = df['Retorno'].rolling(window=10).std()
    df['Direccion'] = (df['Retorno'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df

def crear_dataset(X_data, y_data, ventana):
    X, y = [], []
    for i in range(ventana, len(X_data)):
        X.append(X_data[i-ventana:i])
        y.append(y_data[i])
    return np.array(X), np.array(y)

def entrenar_modelo_activo(simbolo):
    print(f"\n{'='*40}\nIniciando pipeline para {simbolo}\n{'='*40}")
    df = obtener_datos_horarios(simbolo)
    
    features = ['open', 'high', 'low', 'close', 'volume', 'Retorno', 'Volatilidad'] # Estrictamente 7
    target = 'Direccion'
    
    data_x = df[features].values
    data_y = df[target].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_x_escalada = scaler.fit_transform(data_x)
    
    X, y = crear_dataset(data_x_escalada, data_y, VENTANA_HORAS)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    modelo = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    modelo.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)
    
    os.makedirs('models', exist_ok=True)
    modelo.save(f'models/lstm_{simbolo}_1H.keras')
    joblib.dump(scaler, f'models/scaler_{simbolo}_1H.pkl')
    print(f"✅ [{simbolo}] Guardado (Tensor Shape: {X_train.shape})\n")

if __name__ == "__main__":
    for activo in PORTAFOLIO:
        entrenar_modelo_activo(activo)
