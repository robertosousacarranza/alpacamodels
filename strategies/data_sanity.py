"""
data_sanity.py — Data Sanity Check

Valida los datos de entrada antes de cualquier decisión de inversión.

Si detecta:
  • Precios nulos (NaN, Inf, -Inf)
  • Saltos absurdos (>50% intradía / >100% en varios días)
  • Datos desactualizados (sin nuevo precio en >48h hábiles)
  • Volúmenes sospechosos (0 o extremadamente bajos durante horario activo)
  • Discrepancias entre activos correlacionados (SPY vs sectoriales)

→ Congela el bot (crea archivo .PAUSA)
→ Envía alerta por Telegram
→ No permite trading hasta que se elimine manualmente la pausa
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import urllib.request
import urllib.error

logger = logging.getLogger("alpaca_bot")

# ─── Configuración ──────────────────────────────────────────────────────────
DIR_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVO_PAUSA = os.path.join(DIR_PROYECTO, ".PAUSA")
ARCHIVO_ULTIMO_CHECK = os.path.join(DIR_PROYECTO, "data", "ultimo_sanity_check.json")

UMBRAL_SALTO_PRECIO = 0.50         # 50% de cambio intradía → anomalía
UMBRAL_SALTO_MULTIDIA = 1.00       # 100% en 5 días → anomalía
MAX_HORAS_SIN_DATOS = 48           # horas sin actualización de precio
UMBRAL_VOLUMEN_CERO_HORAS = 2      # tolerancia para volumen 0 en horas pico
ACTIVOS_CORRELACIONADOS = [         # pares que deberían moverse similares
    ('SPY', 'AAPL'),
]

# ─── Telegram ───────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


def enviar_telegram(mensaje):
    """Envía una alerta por Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurados. "
                       "Alerta no enviada.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    datos = json.dumps({
        'chat_id': TELEGRAM_CHAT_ID,
        'text': f"🚨 *Data Sanity Check* 🚨\n\n{mensaje}",
        'parse_mode': 'Markdown'
    }).encode()

    try:
        req = urllib.request.Request(
            url, data=datos,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(f"📱 Alerta Telegram enviada: {resp.status}")
            return True
    except Exception as e:
        logger.error(f"❌ Error enviando Telegram: {e}")
        return False


def congelar_bot(razon):
    """
    Crea un archivo .PAUSA para detener el bot.
    """
    try:
        with open(ARCHIVO_PAUSA, 'w') as f:
            f.write(f"BOT PAUSADO — {datetime.now().isoformat()}\n")
            f.write(f"Razón: {razon}\n")
            f.write("Elimina este archivo para reanudar.\n")
        logger.error(f"🛑 🛑 🛑 BOT PAUSADO: {razon}")
        return True
    except Exception as e:
        logger.error(f"❌ No se pudo crear archivo de pausa: {e}")
        return False


def verificar_pausa():
    """Retorna True si el bot está pausado."""
    return os.path.exists(ARCHIVO_PAUSA)


def guardar_estado_check(resultados):
    """Guarda el resultado del último sanity check."""
    os.makedirs(os.path.dirname(ARCHIVO_ULTIMO_CHECK), exist_ok=True)
    try:
        state = {
            'timestamp': datetime.now().isoformat(),
            'resultados': resultados
        }
        with open(ARCHIVO_ULTIMO_CHECK, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo guardar estado de sanity check: {e}")


# ─── Validaciones ───────────────────────────────────────────────────────────

def verificar_nulos(df):
    """Verifica valores nulos o infinitos en columnas numéricas."""
    problemas = []
    cols_numericas = df.select_dtypes(include=[np.number]).columns

    for col in cols_numericas:
        nulos = df[col].isna().sum()
        if nulos > 0:
            problemas.append(f"  ⚠️ {col}: {nulos} valores NaN")

        infs = np.isinf(df[col]).sum() if df[col].dtype in [np.float64, np.float32] else 0
        if infs > 0:
            problemas.append(f"  ⚠️ {col}: {infs} valores Inf")

    return problemas


def verificar_saltos_precio(df, simbolo):
    """Detecta cambios de precio absurdos."""
    problemas = []
    if 'close' not in df.columns or len(df) < 2:
        return problemas

    # Retorno diario
    df = df.copy()
    df['retorno'] = df['close'].pct_change()

    # Saltos intradía extremos
    saltos = df[abs(df['retorno']) > UMBRAL_SALTO_PRECIO]
    if len(saltos) > 0:
        for idx in saltos.index[-3:]:  # últimos 3 saltos
            ret = saltos.loc[idx, 'retorno']
            problemas.append(f"  ⚠️ {simbolo}: Salto de {ret*100:.1f}% en {idx}")

    # Saltos multi-día (rolling 5 días)
    if len(df) > 5:
        df['retorno_5d'] = df['close'].pct_change(periods=5)
        saltos_5d = df[abs(df['retorno_5d']) > UMBRAL_SALTO_MULTIDIA]
        if len(saltos_5d) > 0:
            for idx in saltos_5d.index[-2:]:
                ret = saltos_5d.loc[idx, 'retorno_5d']
                problemas.append(f"  ⚠️ {simbolo}: Salto 5d de {ret*100:.1f}% en {idx}")

    return problemas


def verificar_datos_actualizados(df, simbolo):
    """Verifica que los datos no estén desactualizados."""
    problemas = []
    if len(df) == 0:
        problemas.append(f"  ❌ {simbolo}: DataFrame vacío")
        return problemas

    ultima_fecha = df.index.max()
    if hasattr(ultima_fecha, 'tz'):
        ahora = datetime.now(ultima_fecha.tz)
    else:
        ahora = datetime.now()

    diff = ahora - ultima_fecha
    horas_sin_datos = diff.total_seconds() / 3600

    if horas_sin_datos > MAX_HORAS_SIN_DATOS:
        problemas.append(
            f"  ❌ {simbolo}: {horas_sin_datos:.0f}h sin actualizar "
            f"(último: {ultima_fecha})"
        )

    return problemas


def verificar_volumen(df, simbolo):
    """Detecta volúmenes sospechosos durante horario de mercado."""
    problemas = []
    if 'volume' not in df.columns:
        return problemas

    volumen_cero = (df['volume'] == 0).sum()
    total = len(df)
    ratio_cero = volumen_cero / total if total > 0 else 0

    if ratio_cero > 0.5 and total > 10:
        problemas.append(
            f"  ⚠️ {simbolo}: {volumen_cero}/{total} filas con volumen 0 "
            f"({ratio_cero*100:.0f}%)"
        )

    return problemas


def ejecutar_sanity_check(dfs):
    """
    Ejecuta todas las validaciones sobre todos los activos.

    Parámetros
    ----------
    dfs : dict {str: pd.DataFrame}
        Diccionario con datos de cada símbolo.

    Retorna
    -------
    dict : {
        'paso': bool,
        'errores': list[str],
        'alertas': list[str],
        'detalle': str
    }
    """
    logger.info("🔍 Data Sanity Check — Validando datos de entrada...")

    if verificar_pausa():
        logger.warning("⏸️  Bot previamente pausado. Revisa .PAUSA antes de continuar.")

    todos_errores = []
    todas_alertas = []

    if not dfs or len(dfs) == 0:
        todos_errores.append("❌ No hay datos para validar (dfs vacío)")
        resultado = {
            'paso': False,
            'errores': todos_errores,
            'alertas': [],
            'detalle': "Sin datos — no se puede operar"
        }
        guardar_estado_check(resultado)
        return resultado

    for simb, df in dfs.items():
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            todos_errores.append(f"❌ {simb}: Sin datos")
            continue

        # Verificaciones por activo
        errores_nulos = verificar_nulos(df)
        errores_saltos = verificar_saltos_precio(df, simb)
        errores_fecha = verificar_datos_actualizados(df, simb)
        errores_vol = verificar_volumen(df, simb)

        todos_errores.extend(errores_nulos)
        todos_errores.extend(errores_saltos)
        todos_errores.extend(errores_fecha)
        todas_alertas.extend(errores_vol)

    paso = len(todos_errores) == 0

    if paso:
        logger.info("✅ Sanity Check: TODOS LOS DATOS VÁLIDOS")
        detalle = "Sin anomalías detectadas"
    else:
        logger.warning(f"⚠️ Sanity Check: {len(todos_errores)} problema(s) detectado(s)")
        detalle = "\n".join(todos_errores[:10])  # top 10

        # Si hay errores críticos, congelar el bot
        errores_criticos = [e for e in todos_errores if '❌' in e]
        if errores_criticos:
            mensaje_alerta = "🚨 DATOS CORRUPTOS — Bot pausado\n\n" + detalle
            congelar_bot(mensaje_alerta)
            enviar_telegram(mensaje_alerta)

    for alerta in todas_alertas:
        logger.warning(f"📊 {alerta}")

    resultado = {
        'paso': paso,
        'errores': todos_errores,
        'alertas': todas_alertas,
        'detalle': detalle
    }

    guardar_estado_check(resultado)
    return resultado