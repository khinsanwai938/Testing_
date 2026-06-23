"""
command_dispatch_drone.py — Intent → action mapping for ArduCopter drones.

Every actionable intent is wrapped in a confirmation request via
ConfirmationGate.request_confirmation() rather than executed immediately.
Telemetry queries (get_altitude, get_battery, ...) bypass the gate and
answer immediately, since they are read-only and harmless.
"""

from drone_driver import MAVLinkDroneDriver
from nlp_brain import DroneNLPEngine
from voice_io import voice_reply
from logging_utils import log_event
from confirmation_gate import ConfirmationGate, needs_confirmation


# ── Human-readable labels for confirmation prompts ─────────────────────────
_DRONE_LABELS = {
    "emergency_safe":  "trigger the emergency safe state",
    "arm":             "arm the motors",
    "disarm":          "disarm the motors",
    "takeoff":         "take off",
    "ascend":          "ascend",
    "land":            "land",
    "rtl":             "return to launch",
    "loiter":          "switch to loiter",
    "position_hold":   "switch to position hold",
    "hover":           "hover in place",
    "move_forward":    "move forward",
    "move_backward":   "move backward",
    "move_left":       "move left",
    "move_right":      "move right",
    "descend":         "descend",
    "rotate_left":     "rotate left",
    "rotate_right":    "rotate right",
    "set_heading":     "change heading",
    "set_speed":       "change speed",
    "save_waypoint":   "save a waypoint",
    "goto_waypoint":   "navigate to the last waypoint",
    "export_mission":  "export the mission",
    "start_mission":   "start the autonomous mission",
}


def _label_for(intent: str, captured_text: str) -> str:
    return _DRONE_LABELS.get(intent, intent.replace("_", " "))


def dispatch_drone_command(
    captured_text: str,
    brain: DroneNLPEngine,
    drone: MAVLinkDroneDriver,
    gate: ConfirmationGate,
):
    """
    Parse captured_text. Telemetry queries answer immediately.
    Everything else is queued behind a voice confirmation gate.
    """
    intent, confidence = brain.match_intent(captured_text)

    if not intent or intent == "invalid_command":
        log_event(captured_text, "REJECTED", confidence, "INVALID", "drone")
        voice_reply("Command not recognized. Please try again.")
        return

    log_pfx = lambda outcome: log_event(captured_text, intent, confidence, outcome, "drone")

    # ── Telemetry — answer immediately, no confirmation needed ─────────
    if not needs_confirmation(intent):
        _run_telemetry_query(intent, drone, log_pfx)
        return

    # ── Everything else — gate behind "are you sure?" ──────────────────
    label = _label_for(intent, captured_text)

    def _execute():
        _run_action(intent, captured_text, brain, drone, log_pfx)

    gate.request_confirmation(label, _execute)


def _run_telemetry_query(intent: str, drone: MAVLinkDroneDriver, log_pfx):
    if intent == "get_altitude":
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


def _run_action(
    intent: str,
    captured_text: str,
    brain: DroneNLPEngine,
    drone: MAVLinkDroneDriver,
    log_pfx,
):
    """Executed only after the user has voice-confirmed 'yes'."""

    # ── Emergency ────────────────────────────────────────────────────
    if intent == "emergency_safe":
        action = drone.trigger_emergency_safe_state()
        log_pfx(f"EMERGENCY_{action}")
        voice_reply(
            "Emergency safe state active. Returning home."
            if action == "RTL_ACTIVATED"
            else "Emergency safe state active. Landing immediately."
        )

    # ── Arm / disarm ─────────────────────────────────────────────────
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

    # ── Takeoff / ascend ─────────────────────────────────────────────
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

    # ── Land / RTL ───────────────────────────────────────────────────
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

    # ── Mode shortcuts ───────────────────────────────────────────────
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

    # ── Lateral movement ─────────────────────────────────────────────
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

    # ── Altitude (distance-based) ────────────────────────────────────
    elif intent == "descend":
        value = float(brain.extract_number(captured_text) or 3.0)
        if drone.descend(distance=value):
            log_pfx(f"DESCEND_{value}M")
            voice_reply(f"Descending {value:.0f} meters.")
        else:
            voice_reply("Descend sequence blocked.")
            log_pfx("DESCEND_BLOCKED")

    # ── Yaw / heading ────────────────────────────────────────────────
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

    # ── Speed ────────────────────────────────────────────────────────
    elif intent == "set_speed":
        value = float(brain.extract_number(captured_text) or 5.0)
        if drone.set_groundspeed(value):
            log_pfx(f"SPEED_{value}ms")
            voice_reply(f"Speed {value:.0f} metres per second.")
        else:
            voice_reply("Speed change failed.")
            log_pfx("SPEED_FAILED")

    # ── Waypoints ────────────────────────────────────────────────────
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

    # ── Autonomous mission ───────────────────────────────────────────
    elif intent == "start_mission":
        log_pfx("MISSION_START_REQUEST")
        if drone.start_mission_if_waypoints_exist():
            log_pfx("AUTO_MISSION_ACTIVATED")
            voice_reply("Starting mission.")
        else:
            log_pfx("MISSION_START_ABORTED")
            voice_reply("No waypoints saved. Mission not started.")

    else:
        log_pfx("UNHANDLED_INTENT")
        voice_reply("No action mapped for that command.")
