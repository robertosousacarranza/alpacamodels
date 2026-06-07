"""
bot_trading.py — Punto de entrada para ejecución programada.
Puedes llamar a este script desde cron o systemd para que el bot
se ejecute en horario de mercado sin supervisión.

Uso:
    python bot_trading.py          # Ejecuta un solo ciclo
    python bot_ejecucion.py        # Modo continuo (loop infinito)
"""
from bot_ejecucion import ejecutar_ciclo

if __name__ == "__main__":
    print("🔁 Bot Trading — Modo un solo ciclo")
    ejecutar_ciclo()
