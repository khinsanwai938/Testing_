"""
main.py — VOX-FLIGHT voice control pipeline.

Supports both drone (ArduCopter) and fixed-wing plane (ArduPlane).
Vehicle type is auto-detected from the MAVLink heartbeat on startup,
or can be forced via the VEHICLE_TYPE constant below.

Architecture:
    nlp_brain.py      → DroneNLPEngine (shared, all intents)
    base_driver.py    → MAVLinkBaseDriver (shared infrastructure)
    drone_driver.py   → MAVLinkDroneDriver (copter-specific)
    plane_driver.py   → MAVLinkPlaneDriver (fixed-wing-specific)
    main.py           → dispatch_command(), TTS, STT loop (this file)
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Union

import pyttsx3
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from RealtimeSTT import AudioToTextRecorder
from pymavlink import mavutil

from nlp_brain import DroneNLPEngine
from drone_driver import MAVLinkDroneDriver
from plane_driver import MAVLinkPlaneDriver

# ── Windows DLL path fix for PyTorch audio ────────────────────────────────────
if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    from torchaudio._extension.utils import _init_dll_path
    _init_dll_path()

# ── Runtime settings ──────────────────────────────────────────────────────────
TEST_MODE      = True              # True = keyboard input instead of voice
CONNECTION_STR ="udpin:127.0.0.1:14551" # MAVLink connection string

# Force a vehicle type ("drone" / "plane") or leave None for auto-detect
VEHICLE_TYPE: str | None = None

# ── TTS mute state ────────────────────────────────────────────────────────────
_tts_mute_until = 0.0
_tts_lock       = threading.Lock()

TTS_PRE_DELAY  = 0.15   # seconds to mute BEFORE speech starts
TTS_POST_DELAY = 1.20   # seconds to mute AFTER speech ends (echo suppression)





console = Console()


# ======================================================================
# Vehicle auto-detection
# ======================================================================

def detect_vehicle_type(connection_string: str) -> str:
    print("[SYSTEM] Auto-detecting vehicle type from heartbeat...")
    master = mavutil.mavlink_connection(connection_string)
    hb = master.wait_heartbeat(timeout=10)
    master.close()
    
    time.sleep(1.0) # Clear port lock out

    if hb is None:
        print("[SYSTEM] No heartbeat — defaulting to 'drone'.")
        return "drone"

    # MAVLink Integer IDs:
    # 1 = Standard Fixed-Wing Plane
    # 19 through 24 = Standard VTOL / QuadPlanes / Tailsitters
    PLANE_IDS = {1, 19, 20, 21, 22, 23, 24}

    if hb.type in PLANE_IDS:
        print(f"[SYSTEM] Detected Plane SITL (MAV_TYPE ID={hb.type}). Executing plane pipeline...")
        return "plane"
        
    else:
        print(f"[SYSTEM] Detected Multirotor SITL (MAV_TYPE ID={hb.type}). Executing drone pipeline...")
        return "drone"


def create_vehicle(vehicle_type: str, connection_string: str):
    """Instantiate the correct driver class."""
    if vehicle_type == "plane":
        return MAVLinkPlaneDriver(connection_string)
    return MAVLinkDroneDriver(connection_string)


# ======================================================================
# TTS helpers
# ======================================================================

def _mute_for(seconds: float):
    """Extend the STT mute window by `seconds` from now."""
    global _tts_mute_until
    with _tts_lock:
        _tts_mute_until = max(_tts_mute_until, time.time() + seconds)


def _is_muted() -> bool:
    with _tts_lock:
        return time.time() < _tts_mute_until


def voice_reply(text: str):
    print(f"[SYSTEM SPEAK] → \"{text}\"")

    def _speak():
        try:
            _mute_for(TTS_PRE_DELAY)
            time.sleep(TTS_PRE_DELAY)

            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            del engine

            _mute_for(TTS_POST_DELAY)
        except Exception as e:
            print(f"[TTS ERROR] {e}")

    threading.Thread(target=_speak, daemon=True).start()


# ======================================================================
# Logging
# ======================================================================

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


# ======================================================================
# Intent dispatch — DRONE
# ======================================================================

def dispatch_drone_command(
    captured_text: str,
    brain: DroneNLPEngine,
    drone: MAVLinkDroneDriver,
):
    """Parse captured_text and execute the corresponding drone command."""
    intent, confidence = brain.match_intent(captured_text)

    if not intent or intent == "invalid_command":
        log_event(captured_text, "REJECTED", confidence, "INVALID", "drone")
        voice_reply("Command not recognized. Please try again.")
        return

    log_pfx = lambda outcome: log_event(captured_text, intent, confidence, outcome, "drone")

    # ── Emergency (always allowed) ─────────────────────────────────────
    if intent == "emergency_safe":
        action = drone.trigger_emergency_safe_state()
        log_pfx(f"EMERGENCY_{action}")
        voice_reply(
            "Emergency safe state active. Returning home."
            if action == "RTL_ACTIVATED"
            else "Emergency safe state active. Landing immediately."
        )

    # ── Arm / disarm ───────────────────────────────────────────────────
    elif intent == "arm":
        if drone.arm_vehicle():
            log_pfx("ARMED")
            voice_reply("Armed and ready.")
        else:
            log_pfx("ARM_FAILED")
            voice_reply("Arming failed. Check pre-arm conditions.")

    elif intent == "disarm":
        if drone.disarm_vehicle():
            log_pfx("DISARMED")
            voice_reply("Disarmed.")
        else:
            log_pfx("DISARM_FAILED")
            voice_reply("Disarm failed.")

    # ── Takeoff / ascend ───────────────────────────────────────────────
    elif intent in ("takeoff", "ascend"):
        value      = brain.extract_number(captured_text)
        target_alt = float(value) if value else 5.0
        current_alt = drone.get_altitude() or 0.0

        if current_alt < 0.5:
            voice_reply(
                f"Taking off to {int(target_alt)} meters. "
                "Commands locked until altitude is reached."
            )
            if drone.execute_takeoff(target_alt):
                log_pfx(f"TAKEOFF_COMPLETE_ALT_{target_alt}")
            else:
                voice_reply("Takeoff sequence aborted.")
                log_pfx("TAKEOFF_FAILED")
        else:
            voice_reply(f"Ascending an additional {int(target_alt)} meters.")
            if drone.ascend(distance=target_alt):
                log_pfx(f"ALTITUDE_INCREASE_BY_{target_alt}M")
            else:
                voice_reply("Ascent sequence halted.")
                log_pfx("ALTITUDE_INCREASE_FAILED")

    # ── Land / RTL ─────────────────────────────────────────────────────
    elif intent == "land":
        if drone.land():
            log_pfx("LANDED")
            voice_reply("Landed.")
        else:
            voice_reply("Landing timed out.")
            log_pfx("LAND_TIMEOUT")

    elif intent == "rtl":
        if drone.return_to_launch():
            log_pfx("MODE_RTL")
            voice_reply("Returning home.")
        else:
            voice_reply("Return to launch failed.")
            log_pfx("RTL_FAILED")

    # ── Mode shortcuts ─────────────────────────────────────────────────
    elif intent == "loiter":
        if drone.set_loiter():
            log_pfx("MODE_LOITER")
            voice_reply("Loitering.")
        else:
            voice_reply("Loiter blocked.")
            log_pfx("LOITER_BLOCKED")

    elif intent == "position_hold":
        if drone.set_position_hold():
            log_pfx("MODE_POSHOLD")
            voice_reply("Position hold active.")
        else:
            voice_reply("Position hold blocked.")
            log_pfx("POSHOLD_BLOCKED")

    elif intent == "hover":
        drone.hover()
        log_pfx("HOVER")
        voice_reply("Hovering.")

    # ── Lateral movement ───────────────────────────────────────────────
    elif intent == "move_forward":
        value = float(brain.extract_number(captured_text) or 2.0)
        if drone.move_forward(speed=2.0, duration=value):
            log_pfx(f"FORWARD_{value}S")
            voice_reply(f"Moving forward for {value:.0f} seconds.")
        else:
            voice_reply("Forward movement blocked.")
            log_pfx("FORWARD_BLOCKED")

    elif intent == "move_backward":
        value = float(brain.extract_number(captured_text) or 2.0)
        if drone.move_backward(speed=2.0, duration=value):
            log_pfx(f"BACKWARD_{value}S")
            voice_reply(f"Moving backward for {value:.0f} seconds.")
        else:
            voice_reply("Backward movement blocked.")
            log_pfx("BACKWARD_BLOCKED")

    elif intent == "move_left":
        value = float(brain.extract_number(captured_text) or 2.0)
        if drone.move_left(speed=2.0, duration=value):
            log_pfx(f"LEFT_{value}S")
            voice_reply(f"Strafe left for {value:.0f} seconds.")
        else:
            voice_reply("Left movement blocked.")
            log_pfx("LEFT_BLOCKED")

    elif intent == "move_right":
        value = float(brain.extract_number(captured_text) or 2.0)
        if drone.move_right(speed=2.0, duration=value):
            log_pfx(f"RIGHT_{value}S")
            voice_reply(f"Strafe right for {value:.0f} seconds.")
        else:
            voice_reply("Right movement blocked.")
            log_pfx("RIGHT_BLOCKED")

    # ── Altitude (distance-based) ──────────────────────────────────────
    elif intent == "descend":
        value = float(brain.extract_number(captured_text) or 3.0)
        if drone.descend(distance=value):
            log_pfx(f"DESCEND_{value}M")
            voice_reply(f"Descending {value:.0f} meters.")
        else:
            voice_reply("Descend sequence blocked.")
            log_pfx("DESCEND_BLOCKED")

    # ── Yaw / heading ──────────────────────────────────────────────────
    elif intent == "rotate_left":
        value = float(brain.extract_number(captured_text) or 90.0)
        if drone.rotate_left(degrees=value):
            log_pfx(f"ROTATE_LEFT_{value}DEG")
            voice_reply(f"Turning left {value:.0f} degrees.")
        else:
            log_pfx("ROTATE_LEFT_BLOCKED")

    elif intent == "rotate_right":
        value = float(brain.extract_number(captured_text) or 90.0)
        if drone.rotate_right(degrees=value):
            log_pfx(f"ROTATE_RIGHT_{value}DEG")
            voice_reply(f"Turning right {value:.0f} degrees.")
        else:
            log_pfx("ROTATE_RIGHT_BLOCKED")

    elif intent == "set_heading":
        value = float(brain.extract_number(captured_text) or 0.0)
        if drone.set_heading(heading_degrees=value):
            log_pfx(f"SET_HEADING_{value}")
            voice_reply(f"Heading {value:.0f} degrees.")
        else:
            log_pfx("SET_HEADING_FAILED")

    # ── Speed ──────────────────────────────────────────────────────────
    elif intent == "set_speed":
        value = float(brain.extract_number(captured_text) or 5.0)
        if drone.set_groundspeed(value):
            log_pfx(f"SPEED_{value}ms")
            voice_reply(f"Speed {value:.0f} metres per second.")
        else:
            voice_reply("Speed change failed.")
            log_pfx("SPEED_FAILED")

    # ── Waypoints ──────────────────────────────────────────────────────
    elif intent == "save_waypoint":
        if drone.save_waypoint():
            log_pfx("WAYPOINT_SAVED")
            voice_reply("Waypoint saved.")
        else:
            voice_reply("Could not save waypoint.")
            log_pfx("WAYPOINT_SAVE_FAILED")

    elif intent == "goto_waypoint":
        if drone.goto_last_waypoint():
            log_pfx("GOTO_WAYPOINT")
            voice_reply("Navigating to last waypoint.")
        else:
            voice_reply("No waypoint saved or navigation failed.")
            log_pfx("GOTO_WAYPOINT_FAILED")

    elif intent == "export_mission":
        if drone.export_mission():
            log_pfx("MISSION_EXPORTED")
            voice_reply("Mission exported.")
        else:
            voice_reply("Mission export failed — no waypoints saved.")
            log_pfx("EXPORT_FAILED")

    # ── Telemetry queries ──────────────────────────────────────────────
    elif intent == "get_altitude":
        alt = drone.get_altitude()
        if alt is not None:
            log_pfx(f"ALT_QUERY_{alt:.1f}m")
            voice_reply(f"Altitude {alt:.1f} metres.")
        else:
            voice_reply("Altitude unavailable.")

    elif intent == "get_heading":
        hdg = drone.get_heading()
        if hdg is not None:
            log_pfx(f"HDG_QUERY_{hdg:.0f}deg")
            voice_reply(f"Heading {hdg:.0f} degrees.")
        else:
            voice_reply("Heading unavailable.")

    elif intent == "get_battery":
        batt = drone.get_battery()
        if batt and batt.get("remaining_pct", -1) != -1:
            pct = batt["remaining_pct"]
            log_pfx(f"BATT_QUERY_{pct}pct")
            voice_reply(f"Battery {pct} percent.")
        else:
            voice_reply("Battery unavailable.")

    elif intent == "get_location":
        loc = drone.get_location()
        if loc:
            log_pfx("LOC_QUERY")
            voice_reply(
                f"Latitude {loc['lat']:.4f}, "
                f"longitude {loc['lon']:.4f}, "
                f"altitude {loc['alt_relative']:.1f} metres."
            )
        else:
            voice_reply("Location unavailable.")

    elif intent == "get_mode":
        mode = drone.get_current_mode()
        if mode:
            log_pfx(f"MODE_QUERY_{mode}")
            voice_reply(f"Flight mode {mode}.")
        else:
            voice_reply("Flight mode unavailable.")

    elif intent == "get_gps":
        gps = drone.get_gps_status()
        if gps:
            log_pfx(f"GPS_QUERY_{gps['satellites_visible']}sats")
            voice_reply(
                f"GPS fix {gps['fix_type']}, "
                f"{gps['satellites_visible']} satellites."
            )
        else:
            voice_reply("GPS unavailable.")


    elif intent == "start_mission":
        log_pfx("MISSION_START_REQUEST")
        
        # Run the validation check on the drone driver instance
        if drone.start_mission_if_waypoints_exist():
            log_pfx("AUTO_MISSION_ACTIVATED")
            voice_reply("Start mission.")
        else:
            log_pfx("MISSION_START_ABORTED")
            voice_reply("No waypoint.")

    else:
        log_pfx("UNHANDLED_INTENT")
        voice_reply("No action mapped for that command.")


# ======================================================================
# Intent dispatch — PLANE
# ======================================================================

def dispatch_plane_command(
    captured_text: str,
    brain: DroneNLPEngine,
    plane: MAVLinkPlaneDriver,
):
    """Parse captured_text and execute the corresponding plane command."""
    intent, confidence = brain.match_intent(captured_text)

    if not intent or intent == "invalid_command":
        log_event(captured_text, "REJECTED", confidence, "INVALID", "plane")
        voice_reply("Command not recognized. Please try again.")
        return

    log_pfx = lambda outcome: log_event(captured_text, intent, confidence, outcome, "plane")

    # ── Emergency ──────────────────────────────────────────────────────
    if intent == "emergency_safe":
        action = plane.trigger_emergency_safe_state()
        log_pfx(f"EMERGENCY_{action}")
        voice_reply("Emergency safe state active. Returning home.")

    # ── Arm / disarm ───────────────────────────────────────────────────
    elif intent == "arm":
        if plane.arm_vehicle():
            log_pfx("ARMED")
            voice_reply("Armed and ready.")
        else:
            log_pfx("ARM_FAILED")
            voice_reply("Arming failed. Check pre-arm conditions.")

    elif intent == "disarm":
        if plane.disarm_vehicle():
            log_pfx("DISARMED")
            voice_reply("Disarmed.")
        else:
            log_pfx("DISARM_FAILED")
            voice_reply("Disarm failed.")

    # ── Takeoff ────────────────────────────────────────────────────────
    elif intent == "takeoff":
        # 1. Check if we need to arm first
        if not plane._is_armed():
            log_pfx("AUTO-ARMING BEFORE TAKEOFF")
            voice_reply("Executing launch sequence. Arming engines first.")
            
            if not plane.arm_vehicle():
                log_pfx("ARM_FAILED_FOR_TAKEOFF")
                voice_reply("Launch aborted. Arming sequence failed.")
                
                
            # Give the flight controller a moment to register the arm state
            time.sleep(1.5) 

        # 2. Now run the actual takeoff command
        target_alt = brain.extract_number(captured_text) or 10.0
        if plane.execute_takeoff(target_alt):
            log_pfx(f"TAKEOFF_SUCCESS to {target_alt}m")
            voice_reply(f"Taking off to {target_alt} meters.")
        else:
            log_pfx("TAKEOFF_FAILED")
            voice_reply("Takeoff command rejected by autopilot.")

    # ── Land / RTL ─────────────────────────────────────────────────────
    elif intent == "land":
        if plane.land():
            log_pfx("LANDED")
            voice_reply("Landing complete.")
        else:
            voice_reply("Landing timed out.")
            log_pfx("LAND_TIMEOUT")

    elif intent == "rtl":
        if plane.return_to_launch():
            log_pfx("MODE_RTL")
            voice_reply("Returning home.")
        else:
            voice_reply("Return to launch failed.")
            log_pfx("RTL_FAILED")

    # ── Mode shortcuts ─────────────────────────────────────────────────
    elif intent == "loiter":
        if plane.set_loiter():
            log_pfx("MODE_LOITER")
            voice_reply("Loitering — orbiting current position.")
        else:
            voice_reply("Loiter blocked.")
            log_pfx("LOITER_BLOCKED")

    elif intent == "set_cruise":
        if plane.set_cruise():
            log_pfx("MODE_CRUISE")
            voice_reply("Cruise mode active.")
        else:
            voice_reply("Cruise mode failed.")
            log_pfx("CRUISE_FAILED")

    elif intent == "set_fbwa":
        if plane.set_fbwa():
            log_pfx("MODE_FBWA")
            voice_reply("Fly-by-wire A mode active.")
        else:
            voice_reply("FBWA mode failed.")
            log_pfx("FBWA_FAILED")


    elif intent == "set_guided":
        if plane.set_guided():
            log_pfx("MODE_GUIDED")
            voice_reply("Guided mode active.")
        else:
            voice_reply("Guided mode failed.")
            log_pfx("GUIDED_FAILED")

    # ── Climb / descend to absolute altitude ───────────────────────────
    elif intent in ("ascend", "climb_to"):
        value = float(brain.extract_number(captured_text) or 100.0)
        voice_reply(f"Climbing to {int(value)} meters.")
        if plane.climb_to_altitude(value):
            log_pfx(f"CLIMB_TO_{value}M")
        else:
            voice_reply("Climb failed or timed out.")
            log_pfx("CLIMB_FAILED")

    elif intent in ("descend", "descend_to"):
        value = float(brain.extract_number(captured_text) or 50.0)
        voice_reply(f"Descending to {int(value)} meters.")
        if plane.descend_to_altitude(value):
            log_pfx(f"DESCEND_TO_{value}M")
        else:
            voice_reply("Descent failed or timed out.")
            log_pfx("DESCEND_FAILED")

    # ── Turns ──────────────────────────────────────────────────────────
    elif intent == "turn_left":
        value = brain.extract_number(captured_text)   # optional heading
        if plane.turn_left(heading_degrees=value):
            log_pfx(f"TURN_LEFT_{value or 'BANK'}")
            voice_reply(f"Turning left{f' to {int(value)}°' if value else ''}.")
        else:
            voice_reply("Turn left failed.")
            log_pfx("TURN_LEFT_FAILED")

    elif intent == "turn_right":
        value = brain.extract_number(captured_text)
        if plane.turn_right(heading_degrees=value):
            log_pfx(f"TURN_RIGHT_{value or 'BANK'}")
            voice_reply(f"Turning right{f' to {int(value)}°' if value else ''}.")
        else:
            voice_reply("Turn right failed.")
            log_pfx("TURN_RIGHT_FAILED")

    # ── Heading ────────────────────────────────────────────────────────
    elif intent == "set_heading":
        value = float(brain.extract_number(captured_text) or 0.0)
        if plane.set_heading(value):
            log_pfx(f"SET_HEADING_{value}")
            voice_reply(f"Heading {value:.0f} degrees.")
        else:
            voice_reply("Heading change failed.")
            log_pfx("SET_HEADING_FAILED")

    # ── Speed ──────────────────────────────────────────────────────────
    elif intent == "set_speed":
        value = float(brain.extract_number(captured_text) or 15.0)
        if plane.set_airspeed(value):
            log_pfx(f"AIRSPEED_{value}ms")
            voice_reply(f"Airspeed {value:.0f} metres per second.")
        else:
            voice_reply("Airspeed change failed.")
            log_pfx("AIRSPEED_FAILED")
    

    elif intent == "set_manual":
        if plane.set_manual():
            log_pfx("MODE_MANUAL")
            voice_reply("Manual mode.")  # Updated to say exactly 'Manual mode'
        else:
            voice_reply("Manual mode failed.")
            log_pfx("MANUAL_FAILED")

    # ── Waypoints ──────────────────────────────────────────────────────
    elif intent == "save_waypoint":
        if plane.save_waypoint():
            log_pfx("WAYPOINT_SAVED")
            voice_reply("Waypoint saved.")
        else:
            voice_reply("Could not save waypoint.")
            log_pfx("WAYPOINT_SAVE_FAILED")

    elif intent == "goto_waypoint":
        if plane.goto_last_waypoint():
            log_pfx("GOTO_WAYPOINT")
            voice_reply("Navigating to last waypoint.")
        else:
            voice_reply("No waypoint saved or navigation failed.")
            log_pfx("GOTO_WAYPOINT_FAILED")

    elif intent == "export_mission":
        if plane.export_mission():
            log_pfx("MISSION_EXPORTED")
            voice_reply("Mission exported.")
        else:
            voice_reply("Mission export failed — no waypoints saved.")
            log_pfx("EXPORT_FAILED")

    # ── Telemetry queries ──────────────────────────────────────────────
    elif intent == "get_altitude":
        alt = plane.get_altitude()
        if alt is not None:
            log_pfx(f"ALT_QUERY_{alt:.1f}m")
            voice_reply(f"Altitude {alt:.1f} metres.")
        else:
            voice_reply("Altitude unavailable.")

    elif intent == "get_heading":
        hdg = plane.get_heading()
        if hdg is not None:
            log_pfx(f"HDG_QUERY_{hdg:.0f}deg")
            voice_reply(f"Heading {hdg:.0f} degrees.")
        else:
            voice_reply("Heading unavailable.")

    elif intent == "get_battery":
        batt = plane.get_battery()
        if batt and batt.get("remaining_pct", -1) != -1:
            pct = batt["remaining_pct"]
            log_pfx(f"BATT_QUERY_{pct}pct")
            voice_reply(f"Battery {pct} percent.")
        else:
            voice_reply("Battery unavailable.")

    elif intent == "get_location":
        loc = plane.get_location()
        if loc:
            log_pfx("LOC_QUERY")
            voice_reply(
                f"Latitude {loc['lat']:.4f}, "
                f"longitude {loc['lon']:.4f}, "
                f"altitude {loc['alt_relative']:.1f} metres."
            )
        else:
            voice_reply("Location unavailable.")

    elif intent == "get_mode":
        mode = plane.get_current_mode()
        if mode:
            log_pfx(f"MODE_QUERY_{mode}")
            voice_reply(f"Flight mode {mode}.")
        else:
            voice_reply("Flight mode unavailable.")

    elif intent == "get_gps":
        gps = plane.get_gps_status()
        if gps:
            log_pfx(f"GPS_QUERY_{gps['satellites_visible']}sats")
            voice_reply(
                f"GPS fix {gps['fix_type']}, "
                f"{gps['satellites_visible']} satellites."
            )
        else:
            voice_reply("GPS unavailable.")


    elif intent == "set_auto":
        print("[SYSTEM] Voice trigger 'start mission' acknowledged.")
        
        # 1. Query the flight controller directly to see if a Mission Planner mission exists
        if plane.has_mission_uploaded():
            voice_reply("Mission verified in flight controller. Switching to AUTO flight plan.")
            if plane.set_auto():
                log_pfx("MODE_AUTO_ACTIVATED")
            else:
                voice_reply("Failed to switch to AUTO mode.")
                log_pfx("AUTO_MODE_CHANGE_FAILED")
        else:
            # 2. Safety block if you forgot to write/upload the mission points in Mission Planner
            voice_reply("Safety reject. No mission plan is currently loaded in the flight controller.")
            log_pfx("AUTO_REJECTED_NO_MISSION")

    else:
        log_pfx("UNHANDLED_INTENT")
        voice_reply("No action mapped for that command.")


# ======================================================================
# Unified dispatch router
# ======================================================================

def dispatch_command(
    captured_text: str,
    brain: DroneNLPEngine,
    vehicle: Union[MAVLinkDroneDriver, MAVLinkPlaneDriver],
    vehicle_type: str,
):
    if vehicle_type == "plane":
        dispatch_plane_command(captured_text, brain, vehicle)
    else:
        dispatch_drone_command(captured_text, brain, vehicle)


# ======================================================================
# Main pipeline
# ======================================================================

def run_central_pipeline():
    os.system('cls' if os.name == 'nt' else 'clear')

    # ── Determine vehicle type ─────────────────────────────────────────
    vtype = VEHICLE_TYPE or detect_vehicle_type(CONNECTION_STR)
    console.print(f"\n[bold cyan]Vehicle type: {vtype.upper()}[/bold cyan]")

    brain   = DroneNLPEngine()
    vehicle = create_vehicle(vtype, CONNECTION_STR)

    PROMPT_CONTEXT = (
    "arm disarm takeoff land loiter RTL return to launch guided poshold "
    "move forward move backward move left move right hover hold position "
    "ascend descend climb go up go down rotate left rotate right yaw "
    "set heading north south east west compass bearing degrees "
    "go to waypoint navigate coordinates latitude longitude "
    "set airspeed set groundspeed speed metres per second "
    "altitude check heading check battery status GPS status "
    "current mode current location where am I how high "
    "emergency abort safe state trigger emergency stop "
    "save waypoint mark position export mission flight plan "
    "change altitude increase altitude decrease altitude "
    "loiter mode position hold circle spin turn face "
    "meters seconds faster slower"
)



    console.print(
        f"\n[bold green]=== VOX-FLIGHT PIPELINE ACTIVE | {vtype.upper()} ===[/bold green]"
    )

    # ── Live voice mode ────────────────────────────────────────────────
    if not TEST_MODE:
        live = Live(console=console, refresh_per_second=10, screen=False)
        live.start()

        recorder_config = {
            'spinner':                        False,
            'model':                          'tiny.en',
            'compute_type':                   'int8',
            'download_root':                  None,
            'language':                       'en',
            'initial_prompt':                 'PROMPT_CONTEXT',
            'silero_sensitivity':             0.05,
            'webrtc_sensitivity':             3,
            'post_speech_silence_duration':   0.30,
            'min_length_of_recording':        0.5,
            'min_gap_between_recordings':     0,
            'enable_realtime_transcription':  False,
            'silero_deactivity_detection':    True,
            'early_transcription_on_silence': 0,
            'beam_size':                      1,
            'no_log_file':                    True,
            'silero_use_onnx':                True,
            'faster_whisper_vad_filter':      False,
        }

        recorder = AudioToTextRecorder(**recorder_config)

        live.update(Panel(
            Text("Ready for voice commands...", style="green bold"),
            title=f"[bold cyan]VOX-FLIGHT Active — {vtype.upper()}[/bold cyan]",
            border_style="bold green",
        ))

        console.print("[bold green]>> RUNNING IN LIVE VOICE MODE <<[/bold green]")
        voice_reply("Ready.")

        def on_transcription(text: str):
            if _is_muted():
                print(f"[MUTED] Dropped echo: \"{text.strip()}\"")
                return

            text = text.strip().lstrip(".")
            if not text:
                return
            text = text[0].upper() + text[1:]

            live.update(Panel(
                Text().append("Command: ", style="bold green").append(text, style="bold yellow"),
                title=f"[bold cyan]Voice Control — {vtype.upper()}[/bold cyan]",
                border_style="bold cyan",
            ))

            print(f'[STT] Transcribed: "{text}"')
            dispatch_command(text, brain, vehicle, vtype)

        try:
            while True:
                recorder.text(on_transcription)
        except KeyboardInterrupt:
            live.stop()
            console.print("[bold red]Control Loop Stopped. Exiting...[/bold red]")

    # ── Keyboard / test mode ───────────────────────────────────────────
    else:
        console.print("[bold yellow]>> RUNNING IN KEYBOARD TEST MODE <<[/bold yellow]")
        while True:
            try:
                # This line right here prints out the [ TEST | PLANE ] label:
                raw = input(f"\n[ TEST | {vtype.upper()} ] Enter command: ").lower().strip()
                if not raw:
                    continue
                dispatch_command(raw, brain, vehicle, vtype)
            except KeyboardInterrupt:
                console.print("[bold red]Test mode stopped.[/bold red]")
                break

    # ── Cleanup ────────────────────────────────────────────────────────
    vehicle.close_connection()
    console.print("[bold green][SYSTEM] Pipeline terminated cleanly.[/bold green]")


if __name__ == "__main__":
    run_central_pipeline()
