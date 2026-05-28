# 📈 Quantitative Trading Bot: LSTM Neural Network Architecture

Un sistema automatizado de trading algorítmico (*end-to-end*) desplegado en la nube, diseñado para predecir la dirección diaria del ETF S&P 500 (SPY) utilizando Deep Learning y ejecutar órdenes autónomas mediante la API de Alpaca.

Este proyecto demuestra la implementación completa de un pipeline de Machine Learning financiero: desde la ingesta de datos y la ingeniería de características, hasta el modelado predictivo, la estricta gestión matemática del riesgo en producción y el despliegue de infraestructura cloud.

---

## 🧠 Arquitectura del Sistema y Tech Stack

El bot opera bajo una arquitectura modular y autónoma alojada en Google Cloud Platform (GCP).

* **Lenguaje:** Python 3.11+
* **Ingesta de Datos:** `alpaca-trade-api`, `pandas`
* **Machine Learning (Feature Engineering):** `scikit-learn` (MinMaxScaler, pipelines de transformación)
* **Deep Learning:** `TensorFlow` / `Keras` (Redes Neuronales Recurrentes LSTM)
* **Infraestructura (Cloud):** Google Cloud Compute Engine (Ubuntu), automatización vía `cron`, gestión de memoria Swap optimizada para entornos de bajos recursos.

---

## ⚙️ Metodología y Rigor Matemático

### 1. Ingesta y Feature Engineering
El modelo no consume precios crudos. Se calculan indicadores técnicos y estadísticos dinámicos para capturar la inercia del mercado:
* Retornos porcentuales diarios.
* Volatilidad histórica (desviación estándar móvil de 10 días).
* Normalización de datos en rangos [0, 1] para optimizar la convergencia del gradiente descendente durante el entrenamiento.

### 2. Memoria Secuencial (Arquitectura LSTM)
Dado que los mercados financieros son procesos estocásticos con memoria, se implementó una red **Long Short-Term Memory (LSTM)**. 
* **Tensores 3D:** Los datos se estructuran en ventanas de tiempo de 15 días continuos para que la red analice la trayectoria antes de emitir una predicción probabilística sobre el día 16.
* **Mitigación de Overfitting:** Implementación de capas `Dropout` (0.2) entre las capas LSTM.

---

## 🛡️ Gestión de Riesgo y Seguridad Institucional (Failsafes)

Un pilar central de esta arquitectura es la protección del capital contra anomalías del mercado o errores de infraestructura. El sistema cuenta con tres disyuntores de seguridad programados directamente en el flujo de ejecución:

1. **Freno de Emergencia por Max Drawdown:**
   El bot evalúa la salud global del portafolio en cada iteración. Si se detecta una pérdida acumulada que supera el límite de riesgo estricto (5% del capital base), el algoritmo ejecuta una orden de mercado (Market Sell) para liquidar cualquier exposición abierta, protegiendo el capital restante y suspendiendo operaciones futuras de forma automática.

2. **Dimensionamiento Ajustado por Fricción (Half-Kelly):**
   Para mitigar el impacto del *slippage* (deslizamiento de precio) inherente en el mercado real, el tamaño de la posición no es binario. Se utiliza un **Criterio de Kelly Conservador (Half-Kelly)**, asignando un máximo del 5.6% del capital disponible por operación para maximizar el crecimiento compuesto limitando la varianza.

3. **Validación Estricta de Integridad de Datos:**
   Antes de que el tensor sea alimentado a la red neuronal, el sistema purga datos anómalos. Si la respuesta de la API contiene valores nulos (`NaN`) o si existen huecos en la ventana temporal requerida, la ejecución se aborta en seco para evitar inferencias sesgadas y proteger el portafolio de errores de red.

---

## 🚀 Despliegue en Producción (CI/CD)

El sistema está alojado en una instancia de Google Cloud y opera con autonomía total:
1.  **Automatización:** Un proceso `cron` despierta al bot de lunes a viernes a las 8:00 AM, una vez que la volatilidad de apertura del mercado ha sido absorbida.
2.  **Inferencia en Vivo:** Descarga y limpia datos recientes, carga los pesos del modelo `.keras` pre-entrenado y calcula la probabilidad direccional probabilística.
3.  **Ejecución:** Envía de forma asíncrona la orden ajustada por riesgo al broker (Alpaca) y registra la transacción en un log del sistema antes de regresar al estado de reposo para optimizar recursos computacionales.

---

## 👤 Sobre el Autor

**Roberto Sousa Carranza**
*Matemático | Universidad Autónoma del Estado de México (UAEMéx)*

Con un enfoque profundo en el análisis cuantitativo y la ciencia de datos, desarrollo soluciones algorítmicas que traducen problemas complejos en modelos matemáticos y de Machine Learning ejecutables. Este proyecto fue construido como una demostración técnica de capacidades analíticas avanzadas, infraestructura cloud y gestión de riesgos, orientado a aplicaciones en el sector bancario e institucional.
