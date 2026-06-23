"""
main.py — VOX-FLIGHT voice control pipeline.

Supports both drone (ArduCopter) and fixed-wing plane (ArduPlane).
Vehicle type is auto-detected from the MAVLink heartbeat on startup,
or can be forced via the VEHICLE_TYPE constant below.

Architecture:
    nlp_brain.py                → DroneNLPEngine (shared intent matching)
    base_driver.py              → MAVLinkBaseDriver (shared infrastructure)
    drone_driver.py             → MAVLinkDroneDriver (copter-specific)
    plane_driver.py             → MAVLinkPlaneDriver (fixed-wing-specific)
    voice_io.py                 → TTS output + STT mute-window
    stt_engine.py                → RealtimeSTT recorder + listening loop
    confirmation_gate.py        → "Are you sure?" yes/no gate
    command_dispatch_drone.py   → drone intent → action mapping
    command_dispatch_plane.py   → plane intent → action mapping
    logging_utils.py            → flat-file event log
    main.py                     → wiring + vehicle detection (this file)

Every actionable voice command (everything except telemetry queries)
passes through ConfirmationGate: the system asks "are you sure you
want to <action>?", and only proceeds on a clear "yes".
"""

import os
import sys
import time
from typing import Union

from rich.console import Console
from pymavlink import mavutil

from nlp_brain import DroneNLPEngine
from drone_driver import MAVLinkDroneDriver
from plane_driver import MAVLinkPlaneDriver
from voice_io import voice_reply
from stt_engine import run_voice_loop
from confirmation_gate import ConfirmationGate
from command_dispatch_drone import dispatch_drone_command
from command_dispatch_plane import dispatch_plane_command

# ── Windows DLL path fix for PyTorch audio ────────────────────────────────
if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    from torchaudio._extension.utils import _init_dll_path
    _init_dll_path()

# ── Runtime settings ──────────────────────────────────────────────────────
TEST_MODE      = False                   # True = keyboard input instead of voice
CONNECTION_STR = "udp:127.0.0.1:14550"  # MAVLink connection string

# Force a vehicle type ("drone" / "plane") or leave None for auto-detect
VEHICLE_TYPE: str | None = None

console = Console()


# ======================================================================
# Vehicle auto-detection
# ======================================================================

def detect_vehicle_type(connection_string: str) -> str:
    print("[SYSTEM] Auto-detecting vehicle type from heartbeat...")
    master = mavutil.mavlink_connection(connection_string)
    hb = master.wait_heartbeat(timeout=10)
    master.close()

    time.sleep(1.0)  # clear port lock-out

    if hb is None:
        print("[SYSTEM] No heartbeat — defaulting to 'drone'.")
        return "drone"

    # MAVLink Integer IDs:
    # 1       = Standard Fixed-Wing Plane
    # 19–24   = VTOL / QuadPlanes / Tailsitters
    PLANE_IDS = {1, 19, 20, 21, 22, 23, 24}

    if hb.type in PLANE_IDS:
        print(f"[SYSTEM] Detected Plane SITL (MAV_TYPE ID={hb.type}).")
        return "plane"

    print(f"[SYSTEM] Detected Multirotor SITL (MAV_TYPE ID={hb.type}).")
    return "drone"


def create_vehicle(vehicle_type: str, connection_string: str):
    """Instantiate the correct driver class."""
    if vehicle_type == "plane":
        return MAVLinkPlaneDriver(connection_string)
    return MAVLinkDroneDriver(connection_string)


# ======================================================================
# Unified dispatch router (with confirmation gate in front)
# ======================================================================

def make_command_handler(
    brain: DroneNLPEngine,
    vehicle: Union[MAVLinkDroneDriver, MAVLinkPlaneDriver],
    vehicle_type: str,
    gate: ConfirmationGate,
):
    """
    Returns a callable(text) suitable for the STT loop. Routes each
    utterance either to the confirmation gate (if a question is
    pending) or to the normal intent dispatcher (which will, in turn,
    arm the gate for the next utterance if the intent is actionable).
    """
    dispatch_fn = dispatch_plane_command if vehicle_type == "plane" else dispatch_drone_command

    def handle(text: str):
        # If we're mid-confirmation, this utterance IS the yes/no answer.
        if gate.handle_utterance(text):
            return
        dispatch_fn(text, brain, vehicle, gate)

    return handle


# ======================================================================
# Main pipeline
# ======================================================================

def run_central_pipeline():
    os.system('cls' if os.name == 'nt' else 'clear')

    vtype = VEHICLE_TYPE or detect_vehicle_type(CONNECTION_STR)
    console.print(f"\n[bold cyan]Vehicle type: {vtype.upper()}[/bold cyan]")

    brain   = DroneNLPEngine()
    vehicle = create_vehicle(vtype, CONNECTION_STR)
    gate    = ConfirmationGate()

    command_handler = make_command_handler(brain, vehicle, vtype, gate)

    console.print(
        f"\n[bold green]=== VOX-FLIGHT PIPELINE ACTIVE | {vtype.upper()} ===[/bold green]"
    )

    # ── Live voice mode ────────────────────────────────────────────────
    if not TEST_MODE:
        console.print("[bold green]>> RUNNING IN LIVE VOICE MODE <<[/bold green]")
        voice_reply("Ready.")

        try:
            run_voice_loop(console, vtype, command_handler, gate)
        except KeyboardInterrupt:
            console.print("[bold red]Control Loop Stopped. Exiting...[/bold red]")

    # ── Keyboard / test mode ───────────────────────────────────────────
    else:
        console.print("[bold yellow]>> RUNNING IN KEYBOARD TEST MODE <<[/bold yellow]")
        while True:
            try:
                raw = input(f"\n[ TEST | {vtype.upper()} ] Enter command: ").lower().strip()
                if not raw:
                    gate.check_timeout()
                    continue
                command_handler(raw)
            except KeyboardInterrupt:
                console.print("[bold red]Test mode stopped.[/bold red]")
                break

    # ── Cleanup ────────────────────────────────────────────────────────
    vehicle.close_connection()
    console.print("[bold green][SYSTEM] Pipeline terminated cleanly.[/bold green]")


if __name__ == "__main__":
    run_central_pipeline()