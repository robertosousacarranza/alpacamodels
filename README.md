# 🦙 Alpaca Models — Algotrading con Machine Learning

Sistema de trading algorítmico que usa modelos de Machine Learning (Random Forest y LSTM) para predecir movimientos del SPY y ejecutar órdenes automáticas a través de la API de Alpaca (Paper Trading).

## 📁 Estructura del Proyecto

```
alpacamodels/
├── main.py                 # Verificación de conexión con Alpaca
├── bot_ejecucion.py        # Bot principal (loop continuo con logging y bitácora)
├── bot_trading.py          # Punto de entrada para ejecución por cron (1 ciclo)
├── requirements.txt        # Dependencias del proyecto
├── .env                    # Credenciales de Alpaca (NO subir a Git)
├── .gitignore              # Archivos ignorados por Git
├── README.md               # Este archivo
│
├── strategies/             # Estrategias de trading
│   ├── moving_average.py      # Cruce de medias SMA (backtesting clásico)
│   ├── ml_strategy.py         # Random Forest con feature engineering
│   ├── lstm_strategy.py       # Red LSTM para predicción de series temporales
│   ├── requirements.txt       # Dependencias extra para ML
│   └── models/                # Modelos entrenados
│       ├── lstm_spy_modelo.keras
│       └── scaler_spy.pkl
│
├── models/                 # (placeholder — los modelos reales están en strategies/models/)
├── data/                   # Datos históricos descargados
│   └── operaciones.csv     # Bitácora de operaciones ejecutadas
│
├── utils/
│   └── get_data.py         # Descarga de datos históricos desde Alpaca
│
├── logs/                   # Archivos de log del bot (se crea automáticamente)
│   └── bot.log
│
└── venv/                   # Entorno virtual (ignorado por Git)
```

## 🚀 Cómo Usar

### 1. Configurar credenciales

Copia tu archivo `.env` con tus claves de Alpaca:

```
ALPACA_API_KEY=tu_api_key
ALPACA_SECRET_KEY=tu_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

> ⚠️ **IMPORTANTE**: Usa Paper Trading (entorno de pruebas) mientras ajustas tu estrategia. Nunca compartas tus claves.

### 2. Instalar dependencias

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r strategies/requirements.txt  # Para ML / LSTM
```

### 3. Descargar datos históricos

```bash
python utils/get_data.py
```

Esto descarga 5 años de velas diarias del SPY y las guarda en `data/`.

### 4. Probar la conexión

```bash
python main.py
```

### 5. Ejecutar el bot

**Modo continuo** (loop infinito — recomendado para servidor):
```bash
python bot_ejecucion.py
```

**Modo un ciclo** (para cron / programación horaria):
```bash
python bot_trading.py
```

## 🧠 Estrategias Incluidas

| Estrategia | Archivo | Tipo |
|---|---|---|
| Cruce de Medias Móviles (SMA 50/200) | `strategies/moving_average.py` | Clásica |
| Random Forest con indicadores | `strategies/ml_strategy.py` | Machine Learning |
| Red LSTM (15 días de ventana) | `strategies/lstm_strategy.py` | Deep Learning |

Cada estrategia incluye su propio backtesting.

## 📊 Logging y Bitácora

El bot genera automáticamente:
- **`logs/bot.log`** — Registro detallado con fechas, niveles (INFO/WARN/ERROR).
- **`data/operaciones.csv`** — Historial de cada orden ejecutada (compra/venta, precio, monto, razón).

## 🔐 Seguridad

- Las credenciales están en `.env` (protegido por `.gitignore`).
- Nunca subas tu `.env` o archivos `.csv` / `.h5` / `.pkl` a Git.

## 🐳 Docker (próximamente)

Próximamente: `docker-compose.yml` para ejecutar el bot con un solo comando.

---

Hecho con 🤖 y 🦙 por Roberto Sousa Carranza.
