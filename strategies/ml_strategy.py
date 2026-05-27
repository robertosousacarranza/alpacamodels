import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def calcular_indicadores(df):
    """
    Ingeniería de Características (Feature Engineering).
    Transformamos los precios crudos en variables matemáticas que el modelo pueda entender.
    """
    print("Calculando indicadores matemáticos...")
    
    # 1. Retorno diario
    df['Retorno'] = df['close'].pct_change()
    
    # 2. Volatilidad (Desviación estándar de los retornos en 10 días)
    df['Volatilidad_10d'] = df['Retorno'].rolling(window=10).std()
    
    # 3. Momento (Retorno acumulado de los últimos 5 días)
    df['Momento_5d'] = df['close'].pct_change(periods=5)
    
    # 4. RSI (Relative Strength Index) manual con pandas (14 días)
    delta = df['close'].diff()
    ganancia = delta.where(delta > 0, 0).rolling(window=14).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganancia / perdida
    df['RSI_14d'] = 100 - (100 / (1 + rs))
    
    # 5. Distancia a la Media Móvil Simple de 50 días
    sma_50 = df['close'].rolling(window=50).mean()
    df['Distancia_SMA50'] = (df['close'] - sma_50) / sma_50

    return df

def preparar_datos_y_entrenar(ruta_csv):
    # Cargar datos
    df = pd.read_csv(ruta_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    # Calcular features
    df = calcular_indicadores(df)

    # 6. VARIABLE OBJETIVO (Target)
    # 1 si el precio de cierre de MAÑANA es mayor que el de HOY, 0 si es menor o igual.
    # Usamos shift(-1) para mirar un día en el futuro.
    df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

    # Limpiar datos nulos generados por los rolling windows y el shift
    df.dropna(inplace=True)

    # Seleccionar las columnas para entrenar (X) y la etiqueta (y)
    features = ['Retorno', 'Volatilidad_10d', 'Momento_5d', 'RSI_14d', 'Distancia_SMA50']
    X = df[features]
    y = df['Target']

    # 7. DIVISIÓN DE DATOS (Train/Test Split)
    # ¡CRÍTICO en series de tiempo! No podemos usar train_test_split con shuffle=True, 
    # porque mezclaríamos el futuro con el pasado (data leakage). Cortamos cronológicamente.
    corte = int(len(df) * 0.8) # 80% para entrenar, 20% para probar
    
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y.iloc[:corte], y.iloc[corte:]

    print(f"\nEntrenando modelo con {len(X_train)} días y probando con {len(X_test)} días...")

    # 8. ENTRENAMIENTO DEL MODELO (Random Forest)
    modelo = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,        # Limitamos la profundidad para evitar overfitting
        random_state=42, 
        class_weight='balanced'
    )
    modelo.fit(X_train, y_train)

    # 9. PREDICCIÓN Y EVALUACIÓN
    predicciones = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, predicciones)
    
    print("\n" + "="*40)
    print("   RESULTADOS DEL MACHINE LEARNING   ")
    print("="*40)
    print(f"Precisión General (Accuracy): {accuracy * 100:.2f}%\n")
    print("Reporte de Clasificación Detallado:")
    print(classification_report(y_test, predicciones))
    print("="*40)

if __name__ == "__main__":
    preparar_datos_y_entrenar('/home/roberto/proyectos/alpacamodels/data/SPY_5y_daily.csv')
