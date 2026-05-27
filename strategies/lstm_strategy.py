import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import joblib # <- para guardar el modelo

# Suprimir advertencias molestas de TensorFlow en la terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def crear_secuencias(datos, etiquetas, pasos_temporales=15):
    """
    Transforma datos 2D en Tensores 3D (Muestras, Pasos de Tiempo, Características)
    """
    X, y = [], []
    for i in range(len(datos) - pasos_temporales):
        X.append(datos[i:(i + pasos_temporales)])
        y.append(etiquetas[i + pasos_temporales])
    return np.array(X), np.array(y)

def entrenar_lstm(ruta_csv):
    print("1. Cargando y preparando datos...")
    df = pd.read_csv(ruta_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    # Ingeniería de Características Básica
    df['Retorno'] = df['close'].pct_change()
    df['Volatilidad'] = df['Retorno'].rolling(window=10).std()
    
    # Target: 1 si sube mañana, 0 si baja o se mantiene
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)

    # Seleccionar features
    features = ['open', 'high', 'low', 'close', 'volume', 'Retorno', 'Volatilidad']
    
    # Las Redes Neuronales ODIAN los números grandes. Necesitamos escalar todo entre 0 y 1.
    scaler = MinMaxScaler()
    datos_escalados = scaler.fit_transform(df[features])
    etiquetas = df['Target'].values

    # 2. Construir los Tensores 3D
    # Usaremos 15 días de historia para predecir el día 16
    PASOS_TEMPORALES = 15
    X, y = crear_secuencias(datos_escalados, etiquetas, PASOS_TEMPORALES)

    # 3. División Cronológica (Train / Test)
    corte = int(len(X) * 0.8)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    print(f"Forma del Tensor de Entrenamiento: {X_train.shape}")
    print(f"(Muestras: {X_train.shape[0]}, Días mirando al pasado: {X_train.shape[1]}, Variables: {X_train.shape[2]})")

    # 4. Arquitectura de la Red Neuronal (LSTM)
    print("\n2. Construyendo la Red Neuronal LSTM...")
    modelo = Sequential([
        # Primera capa LSTM que lee la secuencia
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2), # Apagamos el 20% de las neuronas aleatoriamente para evitar overfitting
        
        # Segunda capa LSTM que consolida el aprendizaje
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Capa final de decisión (Neurona Sigmoide que da una probabilidad entre 0 y 1)
        Dense(units=1, activation='sigmoid')
    ])

    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 5. Entrenamiento
    print("\n3. Iniciando Entrenamiento (Epochs)...")
    # Epoch = Una pasada completa por todos los datos
    historial = modelo.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32, # Procesa los datos en bloques de 32 para optimizar memoria
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 6. Evaluación Final
    print("\n" + "="*40)
    print("   RESULTADOS DEL DEEP LEARNING (LSTM)   ")
    print("="*40)
    perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión Final en datos no vistos (Test Accuracy): {precision * 100:.2f}%")
    print("="*40)
    print("\n4. Guardando el modelo y el escalador...")
    
    # Nos aseguramos de que la carpeta exista antes de guardar nada
    os.makedirs('models', exist_ok=True)
    
    # Guardar la red neuronal (usamos la extensión moderna .keras)
    modelo.save('models/lstm_spy_modelo.keras')
    
    # Guardar el objeto matemático del escalador
    joblib.dump(scaler, 'models/scaler_spy.pkl')
    
    print("✅ Cerebro y Lentes guardados exitosamente en la carpeta 'models/'")

if __name__ == "__main__":
    # Ajusta esta ruta a tu ruta absoluta si es necesario, o usa la relativa si ejecutas desde la raíz
    entrenar_lstm('/home/roberto/proyectos/alpacamodels/data/SPY_5y_daily.csv')
