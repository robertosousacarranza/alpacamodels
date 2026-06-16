import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()
api = tradeapi.REST(
    os.getenv('ALPACA_API_KEY'), 
    os.getenv('ALPACA_SECRET_KEY'), 
    os.getenv('ALPACA_BASE_URL')
)

# 1. Obtenemos el objeto cuenta
account = api.get_account()

# 2. Imprimimos los valores clave para el diagnóstico
print(f"--- Diagnóstico de Liquidez ---")
print(f"Estado de la cuenta: {account.status}")
print(f"Equity (Capital Total): ${account.equity}")
print(f"Cash (Efectivo Neto): ${account.cash}")
print(f"Buying Power: ${account.buying_power}")
print(f"Reg T Margin: {account.regt_buying_power}")
print(f"Day Trading Buying Power: {account.daytrading_buying_power}")
