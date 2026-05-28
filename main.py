import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# 1. Cargar las variables de seguridad desde el archivo .env
load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# 2. Inicializar la conexión con Alpaca
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

def verificar_conexion():
    try:
        # Pedimos a Alpaca la información de tu cuenta
        cuenta = api.get_account()
        
        print("¡Conexión exitosa a Alpaca! 🦙")
        print("-" * 40)
        print(f"Estado de la cuenta: {cuenta.status}")
        print(f"Poder adquisitivo (Buying Power): ${cuenta.buying_power}")
        print(f"Dinero en efectivo (Cash): ${cuenta.cash}")
        print("-" * 40)
        
    except Exception as e:
        print("❌ Error al conectar con Alpaca. Revisa tus credenciales en el archivo .env")
        print(f"Detalle: {e}")

if __name__ == "__main__":
    verificar_conexion()
