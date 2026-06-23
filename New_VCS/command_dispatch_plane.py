"""
command_dispatch_plane.py — Intent → action mapping for ArduPlane aircraft.

Mirrors command_dispatch_drone.py's structure: telemetry queries answer
immediately, every other intent is routed through ConfirmationGate first.
"""

import time

from plane_driver import MAVLinkPlaneDriver
from nlp_brain import DroneNLPEngine
from voice_io import voice_reply
from logging_utils import log_event
from confirmation_gate import ConfirmationGate, needs_confirmation


# ── Human-readable labels for confirmation prompts ─────────────────────────
_PLANE_LABELS = {
    "emergency_safe": "trigger the emergency safe state",
    "arm":            "arm the aircraft",
    "disarm":         "disarm the aircraft",
    "takeoff":        "take off",
    "land":           "land",
    "rtl":            "return to launch",
    "loiter":         "loiter",
    "set_cruise":     "switch to cruise mode",
    "set_fbwa":       "switch to fly-by-wire A mode",
    "set_guided":     "switch to guided mode",
    "set_manual":     "switch to manual mode",
    "set_auto":       "start the autonomous mission",
    "ascend":         "climb",
    "climb_to":       "climb",
    "descend":        "descend",
    "descend_to":     "descend",
    "turn_left":      "turn left",
    "turn_right":     "turn right",
    "set_heading":    "change heading",
    "set_speed":      "change airspeed",
    "save_waypoint":  "save a waypoint",
    "goto_waypoint":  "navigate to the last waypoint",
    "export_mission": "export the mission",
}


def _label_for(intent: str) -> str:
    return _PLANE_LABELS.get(intent, intent.replace("_", " "))


def dispatch_plane_command(
    captured_text: str,
    brain: DroneNLPEngine,
    plane: MAVLinkPlaneDriver,
    gate: ConfirmationGate,
):
    """
    Parse captured_text. Telemetry queries answer immediately.
    Everything else is queued behind a voice confirmation gate.
    """
    intent, confidence = brain.match_intent(captured_text)

    if not intent or intent == "invalid_command":
        log_event(captured_text, "REJECTED", confidence, "INVALID", "plane")
        voice_reply("Command not recognized. Please try again.")
        return

    log_pfx = lambda outcome: log_event(captured_text, intent, confidence, outcome, "plane")

    # ── Telemetry — answer immediately, no confirmation needed ─────────
    if not needs_confirmation(intent):
        _run_telemetry_query(intent, plane, log_pfx)
        return

    # ── Everything else — gate behind "are you sure?" ──────────────────
    label = _label_for(intent)

    def _execute():
        _run_action(intent, captured_text, brain, plane, log_pfx)

    gate.request_confirmation(label, _execute)


def _run_telemetry_query(intent: str, plane: MAVLinkPlaneDriver, log_pfx):
    if intent == "get_altitude":
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


def _run_action(
    intent: str,
    captured_text: str,
    brain: DroneNLPEngine,
    plane: MAVLinkPlaneDriver,
    log_pfx,
):
    """Executed only after the user has voice-confirmed 'yes'."""

    # ── Emergency ────────────────────────────────────────────────────
    if intent == "emergency_safe":
        action = plane.trigger_emergency_safe_state()
        log_pfx(f"EMERGENCY_{action}")
        voice_reply("Emergency safe state active. Returning home.")

    # ── Arm / disarm ─────────────────────────────────────────────────
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

    # ── Takeoff ──────────────────────────────────────────────────────
    elif intent == "takeoff":
        if not plane._is_armed():
            log_pfx("AUTO_ARMING_BEFORE_TAKEOFF")
            voice_reply("Executing launch sequence. Arming engines first.")

            if not plane.arm_vehicle():
                log_pfx("ARM_FAILED_FOR_TAKEOFF")
                voice_reply("Launch aborted. Arming sequence failed.")
                return

            time.sleep(1.5)   # let the FC register the arm state

        target_alt = brain.extract_number(captured_text) or 10.0
        if plane.execute_takeoff(target_alt):
            log_pfx(f"TAKEOFF_SUCCESS_TO_{target_alt}M")
            voice_reply(f"Taking off to {target_alt} meters.")
        else:
            log_pfx("TAKEOFF_FAILED")
            voice_reply("Takeoff command rejected by autopilot.")

    # ── Land / RTL ───────────────────────────────────────────────────
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

    # ── Mode shortcuts ───────────────────────────────────────────────
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

    elif intent == "set_manual":
        if plane.set_manual():
            log_pfx("MODE_MANUAL")
            voice_reply("Manual mode.")
        else:
            voice_reply("Manual mode failed.")
            log_pfx("MANUAL_FAILED")

    elif intent == "set_auto":
        if plane.has_mission_uploaded():
            voice_reply("Mission verified in flight controller. Switching to AUTO flight plan.")
            if plane.set_auto():
                log_pfx("MODE_AUTO_ACTIVATED")
            else:
                voice_reply("Failed to switch to AUTO mode.")
                log_pfx("AUTO_MODE_CHANGE_FAILED")
        else:
            voice_reply("Safety reject. No mission plan is currently loaded in the flight controller.")
            log_pfx("AUTO_REJECTED_NO_MISSION")

    # ── Climb / descend to absolute altitude ─────────────────────────
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

    # ── Turns ────────────────────────────────────────────────────────
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

    # ── Heading ──────────────────────────────────────────────────────
    elif intent == "set_heading":
        value = float(brain.extract_number(captured_text) or 0.0)
        if plane.set_heading(value):
            log_pfx(f"SET_HEADING_{value}")
            voice_reply(f"Heading {value:.0f} degrees.")
        else:
            voice_reply("Heading change failed.")
            log_pfx("SET_HEADING_FAILED")

    # ── Speed ────────────────────────────────────────────────────────
    elif intent == "set_speed":
        value = float(brain.extract_number(captured_text) or 15.0)
        if plane.set_airspeed(value):
            log_pfx(f"AIRSPEED_{value}ms")
            voice_reply(f"Airspeed {value:.0f} metres per second.")
        else:
            voice_reply("Airspeed change failed.")
            log_pfx("AIRSPEED_FAILED")

    # ── Waypoints ────────────────────────────────────────────────────
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

    else:
        log_pfx("UNHANDLED_INTENT")
        voice_reply("No action mapped for that command.")
