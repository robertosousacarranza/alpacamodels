# 📈 Quantitative Trading Bot: LSTM Neural Network Architecture

Un sistema automatizado de trading algorítmico (*end-to-end*) desplegado en la nube, diseñado para predecir la dirección diaria del ETF S&P 500 (SPY) utilizando Deep Learning y ejecutar órdenes autónomas mediante la API de Alpaca.

Este proyecto demuestra la implementación completa de un pipeline de Machine Learning financiero: desde la ingesta de datos y la ingeniería de características, hasta el modelado predictivo, la gestión matemática del riesgo y el despliegue en producción.

---

## 🧠 Arquitectura del Sistema y Tech Stack

El bot opera bajo una arquitectura modular y autónoma alojada en Google Cloud Platform (GCP).

* **Lenguaje:** Python 3.11+
* **Ingesta de Datos:** `alpaca-trade-api`, `pandas`
* **Machine Learning (Feature Engineering):** `scikit-learn` (MinMaxScaler, pipelines de transformación)
* **Deep Learning:** `TensorFlow` / `Keras` (Redes Neuronales Recurrentes LSTM)
* **Infraestructura (Cloud):** Google Cloud Compute Engine (Ubuntu 22.04), `cron` para automatización de tareas, gestión de memoria Swap.

---

## ⚙️ Metodología y Rigor Matemático

### 1. Ingesta y Feature Engineering
El modelo no consume precios crudos. Se calculan indicadores técnicos y estadísticos dinámicos para capturar la inercia del mercado:
* Retornos porcentuales diarios.
* Volatilidad histórica (desviación estándar móvil de 10 días).
* Normalización de datos en rangos [0, 1] para optimizar la convergencia del gradiente descendente durante el entrenamiento.

### 2. Memoria Secuencial (Arquitectura LSTM)
Dado que los mercados financieros son procesos estocásticos con memoria, se descartaron modelos de clasificación plana (como Random Forest) en favor de una red **Long Short-Term Memory (LSTM)**. 
* **Tensores 3D:** Los datos se estructuran en ventanas de tiempo de 15 días continuos para que la red interprete la trayectoria antes de emitir una predicción probabilística sobre el día 16.
* **Mitigación de Overfitting:** Implementación de capas `Dropout` (0.2) entre las capas LSTM.

### 3. Gestión de Riesgo (Criterio de Kelly Simplificado)
La ejecución de órdenes no es binaria ni arbitraria. El tamaño de la posición se determina matemáticamente utilizando una variante del Criterio de Kelly. Basado en la precisión (*Accuracy*) comprobada del modelo en el conjunto de prueba (Test Data), el algoritmo asigna un porcentaje específico de capital (ej. 11.38%) por operación para maximizar el crecimiento compuesto a largo plazo y minimizar el riesgo de ruina.

---

## 🚀 Despliegue en Producción (CI/CD)

El sistema está alojado en una instancia de Google Cloud y opera con autonomía total:
1.  **Automatización:** Un proceso `cron` despierta al bot de lunes a viernes a las 8:00 AM (tiempo del centro de México), cuando la volatilidad de apertura del mercado ha sido absorbida.
2.  **Inferencia en vivo:** Descarga los últimos 40 días, aísla la ventana tensorial pertinente, carga los pesos del modelo `.keras` pre-entrenado y calcula la probabilidad direccional.
3.  **Ejecución:** Envía de forma asíncrona la orden ajustada por riesgo al broker (Alpaca) y registra la transacción en un log del sistema antes de entrar en reposo.

---

## 👤 Sobre el Autor

**Roberto Sousa Carranza**
*Matemático | Universidad Autónoma del Estado de México (UAEMéx)*

Con un enfoque profundo en el análisis cuantitativo y la ciencia de datos, desarrollo soluciones algorítmicas que traducen problemas complejos en modelos matemáticos y de Machine Learning ejecutables. Este proyecto forma parte de un portafolio avanzado diseñado para el sector bancario y el análisis de datos a gran escala.
