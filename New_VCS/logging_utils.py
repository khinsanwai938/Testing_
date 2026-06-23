"""
logging_utils.py — Flat-file command/event logging for VOX-FLIGHT.
"""

from datetime import datetime


def log_event(raw_text: str, intent: str, confidence: float, outcome: str,
              vehicle: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f" [{vehicle.upper()}]" if vehicle else ""
    with open("voice_flight_commands.log", "a") as f:
        f.write(
            f"[{timestamp}]{tag} TEXT: \"{raw_text}\" | "
            f"INTENT: {intent.upper()} | "
            f"CONF: {confidence:.2f} | "
            f"STATUS: {outcome}\n"
        )
