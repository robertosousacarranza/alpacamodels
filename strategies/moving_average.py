import pandas as pd
import numpy as np

def simular_cruce_medias(ruta_csv, ventana_corta=50, ventana_larga=200):
    # 1. Cargar datos y asegurar orden cronológico
    df = pd.read_csv("/home/roberto/proyectos/alpacamodels/data/SPY_5y_daily.csv")
    
    # Alpaca usa 'timestamp' como columna de fecha. La convertimos a índice
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    
    print(f"Analizando {len(df)} días de mercado...")

    # 2. Calcular Medias Móviles Simples (SMA)
    df['SMA_Corta'] = df['close'].rolling(window=ventana_corta).mean()
    df['SMA_Larga'] = df['close'].rolling(window=ventana_larga).mean()

    # 3. Generar Señales Matemáticas
    # 1 = Comprado (Tendencia alcista), 0 = Fuera del mercado (Efectivo)
    df['senal'] = np.where(df['SMA_Corta'] > df['SMA_Larga'], 1, 0)
    
    # IMPORTANTE: Desplazar la señal un día hacia adelante (shift). 
    # Si la señal se genera hoy al cierre, compramos mañana a primera hora. Evita el sesgo de mirar al futuro.
    df['posicion'] = df['senal'].shift(1)

    # 4. Calcular Retornos Diarios
    df['retorno_mercado'] = df['close'].pct_change()
    df['retorno_estrategia'] = df['retorno_mercado'] * df['posicion']

    # 5. Calcular Retornos Acumulados (Rendimiento Compuesto)
    # Reemplazamos NaNs con 0 para que la multiplicación empiece limpia
    df['rendimiento_mercado_acum'] = (1 + df['retorno_mercado'].fillna(0)).cumprod() - 1
    df['rendimiento_estrategia_acum'] = (1 + df['retorno_estrategia'].fillna(0)).cumprod() - 1

    # 6. Extraer Métricas Finales
    rendimiento_final_mercado = df['rendimiento_mercado_acum'].iloc[-1] * 100
    rendimiento_final_estrategia = df['rendimiento_estrategia_acum'].iloc[-1] * 100
    
    # Calcular Sharpe Ratio básico (simplificado, asumiendo tasa libre de riesgo = 0)
    sharpe_mercado = (df['retorno_mercado'].mean() / df['retorno_mercado'].std()) * np.sqrt(252)
    sharpe_estrategia = (df['retorno_estrategia'].mean() / df['retorno_estrategia'].std()) * np.sqrt(252)

    print("\n" + "="*40)
    print("      RESULTADOS DEL BACKTESTING (5 AÑOS)      ")
    print("="*40)
    print(f"Estrategia: Cruce de SMA {ventana_corta} / SMA {ventana_larga}")
    print(f"Rendimiento Mercado (Buy & Hold): {rendimiento_final_mercado:.2f}%")
    print(f"Rendimiento de tu Algoritmo:      {rendimiento_final_estrategia:.2f}%")
    print("-" * 40)
    print(f"Sharpe Ratio Mercado:             {sharpe_mercado:.2f}")
    print(f"Sharpe Ratio Algoritmo:           {sharpe_estrategia:.2f}")
    print("="*40)

    # Guardar resultados analizados para el futuro
    df.to_csv('/home/roberto/proyectos/alpacamodels/data/SPY_backtest_results.csv')
    print("\nAnálisis detallado guardado en 'data/SPY_backtest_results.csv'")

if __name__ == "__main__":
    simular_cruce_medias('data/SPY_5y_daily.csv')
