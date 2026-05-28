import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta

# 1. Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# 2. Inicializar conexión
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

def obtener_datos_historicos(simbolo, años=5):
    print(f"Descargando datos diarios de {simbolo} de los últimos {años} años...")
    
    # 3. Calcular las fechas dinámicamente
    fecha_fin = datetime.now() - timedelta(days=1)
    fecha_inicio = fecha_fin - timedelta(days=años*365)
    
    # Convertir a formato de cadena que Alpaca entiende (YYYY-MM-DD)
    start_str = fecha_inicio.strftime('%Y-%m-%d')
    end_str = fecha_fin.strftime('%Y-%m-%d')

    try:
       # 4. Solicitar las barras (velas) a la API de Alpaca
        barras = api.get_bars(
            simbolo, 
            tradeapi.TimeFrame.Day, 
            start=start_str, 
            end=end_str, 
            feed='iex',         # <--- ESTA ES LA SOLUCIÓN
            adjustment='all' 
        ).df 

        # 5. Guardar el DataFrame como CSV en nuestra carpeta de datos
        ruta_archivo = f"/home/roberto/proyectos/alpacamodels/data/{simbolo}_5y_daily.csv"
        barras.to_csv(ruta_archivo)
        
        print(f"✅ ¡Datos guardados exitosamente en {ruta_archivo}!")
        print(f"📊 Total de días de mercado extraídos: {len(barras)}")
        return barras

    except Exception as e:
        print(f"❌ Error al obtener los datos: {e}")
        return None

if __name__ == "__main__":
    # Ejecutamos la función para el S&P 500
    df = obtener_datos_historicos('SPY')
    
    if df is not None:
        print("\nPrimeras filas del DataFrame:")
        print(df.head())
