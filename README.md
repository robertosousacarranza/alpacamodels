# 📈 Quantitative Trading Portfolio: Multi-Asset LSTM Architecture

Un ecosistema automatizado de trading algorítmico desplegado en la nube de GCP. El sistema utiliza redes neuronales profundas (LSTM) para predecir la dirección probabilística de un portafolio de activos diversificados, ejecutando órdenes autónomas mediante la API de Alpaca bajo una estricta gestión matemática del riesgo.

---

## 🧠 Evolución de la Arquitectura (Multi-Timeframe & Multi-Asset)

El proyecto evolucionó de un modelo base enfocado en el S&P 500 a un **portafolio ortogonal de 4 activos** diseñado para minimizar la correlación cruzada y diversificar el riesgo sectorial:
1. **Tecnología (Alta Beta):** Apple (`AAPL`)
2. **Sector Financiero (Sensibilidad a Tasas):** JPMorgan Chase (`JPM`)
3. **Energía (Materias Primas):** ExxonMobil (`XOM`)
4. **Refugio Seguro (Safe Haven):** ETF de Oro Físico (`GLD`)

Para capturar la volatilidad en distintas frecuencias, la arquitectura opera con **dos motores independientes paralelos**:

* ⏱️ **Motor Táctico Intradía (Velas de 1 Hora):** Un cronjob ejecuta un escaneo secuencial a la hora en punto para capitalizar las fluctuaciones intradiarias. Entrenado con más de 8,800 secuencias tensoriales por activo.
* 📅 **Motor Estratégico Diario (Velas de 1 Día):** Diseñado para tendencias macroeconómicas, evalúa el cierre del mercado y toma posiciones al día siguiente aislando el ruido estocástico del corto plazo.

---

## ⚙️ Ingeniería de Software e Infraestructura Cloud

Desplegar 8 modelos de Deep Learning (4 diarios, 4 horarios) en un entorno con recursos limitados (1 GB RAM en Google Cloud) requirió soluciones de ingeniería de bajo nivel:

* **Multiplexación por División de Tiempo (Sequential Execution):** En lugar de paralelizar procesos pesados, el bot maestro itera sobre la lista de activos. Analiza una empresa, envía la orden y ejecuta `K.clear_session()` para purgar la memoria RAM antes de cargar la red neuronal del siguiente activo, evitando el colapso del servidor (OOM).
* **Asignación Dinámica de Capital (Live API Integration):** El sistema consulta en tiempo real el *Buying Power* disponible en Alpaca y lo divide equitativamente entre los activos del portafolio.
* **Precisión de Ejecución Institucional:** Implementación de envolturas matemáticas para garantizar el redondeo a 2 decimales en el cálculo de fracciones nocionales, cumpliendo con los estándares de enrutamiento de órdenes del *broker*.

---

## 🛡️ Gestión de Riesgo (Risk Failsafes)

1.  **Freno de Emergencia por Max Drawdown:** Si el portafolio global sufre una caída superior al 5% desde el capital base inicial, el sistema lanza una instrucción global de `Market Sell` liquidando todo el portafolio para refugiar el dinero en dólares y aborta la ejecución con `sys.exit(1)`.
2.  **Dimensionamiento Ajustado por Fricción (Half-Kelly):** Se utiliza una aproximación conservadora del Criterio de Kelly (5.6% del capital asignado por operación) para maximizar el crecimiento compuesto mientras se limita la varianza y se absorbe el *slippage*.
3.  **Filtrado Estricto de Integridad:** El pipeline valida la recepción milimétrica de las ventanas temporales exigidas (15 timesteps). Si la API devuelve datos corruptos por cierres de mercado o días festivos, el bot de ese activo entra en estado de reposo temporal protegiendo el modelo predictivo.

---

## 🚀 Despliegue en Producción (Linux Cron)

El entorno en Google Cloud (Ubuntu) automatiza el pipeline completo:
* `01 9-14 * * 1-5`: Despierta al motor horario durante la ventana operativa de NY.
* `0 8 * * 1-5`: Despierta al motor diario para tomar posiciones estructurales.
Todo el ecosistema opera silenciosamente registrando sus inferencias probabilísticas y decisiones de capital en *logs* de sistema auditables.

---

## 👤 Sobre el Autor

**Roberto Sousa Carranza**
*Matemático | Análisis Cuantitativo y Ciencia de Datos*

Desarrollo soluciones algorítmicas combinando rigor matemático, Machine Learning e ingeniería de datos en la nube. Este portafolio es una demostración técnica de modelado predictivo y gestión de riesgos aplicada a mercados estocásticos complejos.

---

## 🦙 Estructura del Proyecto

```
alpacamodels/
├── bot_ejecucion.py        # Bot principal (loop continuo, multi-activo, optimización dinámica)
├── bot_trading.py          # Punto de entrada para cron (1 ciclo)
├── main.py                 # Verificación de conexión con Alpaca
├── requirements.txt        # Dependencias del proyecto
├── .env                    # Credenciales de Alpaca (NO subir a Git)
├── .gitignore              # Archivos ignorados por Git
├── README.md               # Este archivo
│
├── strategies/
│   ├── moving_average.py      # Cruce de medias SMA (backtesting clásico)
│   ├── ml_strategy.py         # Random Forest con feature engineering
│   ├── lstm_strategy.py       # Red LSTM para predicción de series temporales
│   ├── portfolio_optimizer.py # Matriz de covarianza + optimización Sharpe (SciPy)
│   ├── requirements.txt       # Dependencias extra para ML
│   └── models/                # Modelos entrenados
│       ├── lstm_spy_modelo.keras
│       └── scaler_spy.pkl
│
├── models/                 # (placeholder)
├── data/                   # Datos históricos descargados
│   └── operaciones.csv     # Bitácora de operaciones ejecutadas
├── utils/
│   └── get_data.py         # Descarga de datos históricos desde Alpaca
├── logs/                   # Logs del bot (se crea automáticamente)
└── venv/                   # Entorno virtual (ignorado por Git)
```

## 🚀 Cómo Usar

### 1. Configurar credenciales
```ini
ALPACA_API_KEY=tu_api_key
ALPACA_SECRET_KEY=tu_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 2. Instalar dependencias
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r strategies/requirements.txt
```

### 3. Descargar datos históricos
```bash
python utils/get_data.py
```

### 4. Probar conexión
```bash
python main.py
```

### 5. Ejecutar el bot
```bash
python bot_ejecucion.py   # Modo continuo (loop infinito)
python bot_trading.py     # Modo un ciclo (para cron)
```

## 📊 Logging y Bitácora

El bot genera automáticamente:
- **`logs/bot.log`** — Registro detallado con timestamps y niveles (INFO/WARN/ERROR).
- **`data/operaciones.csv`** — Historial de cada orden ejecutada.

## 🔐 Seguridad

- Credenciales en `.env` (protegido por `.gitignore`).
- Nunca subas `.env`, `*.csv`, `*.h5`, `*.pkl`, `logs/` a Git.
