"""
drone_driver.py — ArduCopter / multirotor driver.

Extends MAVLinkBaseDriver with:
  - Altitude gate (LOCKED state during climb)
  - Hover watchdog (battery monitor + zero-velocity heartbeat)
  - Hover timer (auto-hold after reaching altitude)
  - Takeoff, ascend, descend, lateral velocity commands
  - Yaw / heading control
  - Loiter / POSHOLD mode shortcuts
  - Waypoint save / go-to / export
"""

import time
import threading
from typing import Optional
from pymavlink import mavutil

from base_driver import MAVLinkBaseDriver


class MAVLinkDroneDriver(MAVLinkBaseDriver):

    # ── Altitude-gate tuning ─────────────────────────────────────────────
    ALTITUDE_TOLERANCE = 0.5     # metres — "close enough" to target
    HOVER_TIMEOUT      = 10.0    # seconds before auto-LOITER after reaching alt

    _LOG_PREFIX = "[DRONE]"

    # ======================================================================
    # Construction
    # ======================================================================

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        super().__init__(connection_string)

        # ── Altitude gate ───────────────────────────────────────────────
        self._target_altitude: Optional[float] = None
        self._altitude_monitor: Optional[threading.Thread] = None

        # ── Waypoint store ──────────────────────────────────────────────
        self._waypoints: list[dict] = []   # list of {"lat", "lon", "alt"}

        # ── Hover watchdog (battery + zero-velocity hold) ───────────────
        self._hover_heartbeat_active = True
        self._hover_thread = threading.Thread(
            target=self._production_hover_watchdog, daemon=True
        )
        self._hover_thread.start()

    # ======================================================================
    # Altitude gate — internal
    # ======================================================================

    def _start_altitude_monitor(self, target_altitude: float):
        """
        Spin a daemon thread that watches altitude and transitions
        LOCKED → READY once the vehicle arrives, then starts the hover timer.
        """
        self._target_altitude = target_altitude
        with self._state_lock:
            self._state = self._STATE_LOCKED

        def _monitor():
            print(
                f"{self._LOG_PREFIX} 🔒 Altitude gate ACTIVE — commands locked until "
                f"{target_altitude} m is confirmed."
            )
            while True:
                with self._state_lock:
                    if self._state not in (self._STATE_LOCKED, self._STATE_READY):
                        return  # emergency / disarm — stop watching

                alt = self._get_altitude()
                if alt is not None and alt >= target_altitude - self.ALTITUDE_TOLERANCE:
                    with self._state_lock:
                        if self._state == self._STATE_LOCKED:
                            self._state = self._STATE_READY
                            print(
                                f"{self._LOG_PREFIX} ✅ Target altitude {target_altitude} m reached "
                                f"(current: {alt:.1f} m). Commands now accepted."
                            )
                    self._start_hover_timer()
                    return

                time.sleep(0.5)

        self._altitude_monitor = threading.Thread(target=_monitor, daemon=True)
        self._altitude_monitor.start()

    # ======================================================================
    # Hover timer — internal
    # ======================================================================

    def _start_hover_timer(self):
        if self._hover_timer is not None:
            self._hover_timer.cancel()

        print(f"{self._LOG_PREFIX} Hover timer initialized ({self.HOVER_TIMEOUT}s).")

        def _speak_warning():
            with self._state_lock:
                if self._state == self._STATE_READY:
                    print(f"{self._LOG_PREFIX} Warning: No command detected. "
                          "Engaging auto-hover fallback in 2 seconds...")

        def _engage_hover_hold():
            with self._state_lock:
                if self._state == self._STATE_READY:
                    self._state = self._STATE_HOVER
            print(f"{self._LOG_PREFIX} Safeguard active: streaming zero-velocity hold.")

        if self.HOVER_TIMEOUT > 3.0:
            warning_timer = threading.Timer(3.0, _speak_warning)
            warning_timer.daemon = True
            warning_timer.start()

        self._hover_timer = threading.Timer(self.HOVER_TIMEOUT, _engage_hover_hold)
        self._hover_timer.daemon = True
        self._hover_timer.start()

    def reset_hover_timeout(self):
        """Reset the idle-hover countdown — call when user gives a command."""
        with self._state_lock:
            if self._state != self._STATE_READY:
                return
        print(f"{self._LOG_PREFIX} Action detected — resetting hover countdown.")
        self._start_hover_timer()

    # ======================================================================
    # Hover watchdog — internal (battery monitor + active hover heartbeat)
    # ======================================================================

    def _production_hover_watchdog(self):
        """
        Background watchdog: silent battery tracking and zero-velocity hover hold.
        Triggers emergency RTL on critical battery conditions.
        """
        has_warned_low     = False
        has_warned_anomaly = False

        while self._hover_heartbeat_active:
            with self._state_lock:
                current_state = self._state

            batt = self.get_battery()
            if batt is not None:
                voltage   = batt.get("voltage_v") or 12.6
                remaining = batt.get("remaining_pct", -1)

                if (voltage < self.BATTERY_MIN_VOLTAGE) or (
                        remaining != -1 and remaining < self.BATTERY_MIN_PCT
                        and voltage < 11.1):
                    print(f"\n[CRITICAL] 🔋 BATTERY EXHAUSTION DETECTED: "
                          f"{voltage:.2f}V, {remaining}%.")
                    if self.is_flying():
                        print("[CRITICAL] EMERGENCY SAFETY ACTION: Aborting flight!")
                        self.trigger_emergency_safe_state()
                        break

                elif remaining == 0 and voltage >= self.BATTERY_MIN_VOLTAGE:
                    if not has_warned_anomaly:
                        print(f"\n{self._LOG_PREFIX} ⚠️ Telemetry anomaly: 0% capacity "
                              f"but voltage is SAFE at {voltage:.2f}V. "
                              "Flight continuing — adjust capacity in Mission Planner later.")
                        has_warned_anomaly = True
                    has_warned_low = False

                elif remaining != -1 and remaining < self.BATTERY_WARN_PCT:
                    if not has_warned_low:
                        print(f"\n{self._LOG_PREFIX} ⚠️ Low battery: "
                              f"{remaining}% ({voltage:.2f}V).")
                        has_warned_low = True
                    has_warned_anomaly = False

                else:
                    has_warned_low     = False
                    has_warned_anomaly = False

            # ── Zero-velocity hover heartbeat ───────────────────────────
            if current_state == self._STATE_HOVER:
                try:
                    self.send_body_translation(0.0, 0.0, 0.0)
                except Exception as e:
                    print(f"{self._LOG_PREFIX} Hover heartbeat error: {e}")

            time.sleep(0.5)

    # ======================================================================
    # Takeoff
    # ======================================================================

    def execute_takeoff(self, target_altitude: float = 5.0) -> bool:
        """
        Command takeoff to target_altitude metres and engage the altitude gate.
        Motors must already be armed.

        Blocks until altitude is reached or timeout / stuck-altitude fires.
        """
        if not self._is_armed():
            print(f"{self._LOG_PREFIX} CRITICAL: Takeoff rejected — motors must be armed first.")
            return False

        print(f"{self._LOG_PREFIX} Taking off to {target_altitude:.1f} m — "
              "commands locked until altitude is confirmed.")

        success = self._send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, p7=target_altitude
        )
        if not success:
            print(f"{self._LOG_PREFIX} Takeoff command rejected.")
            return False

        self._start_altitude_monitor(target_altitude)

        TAKEOFF_TIMEOUT = 30
        STUCK_THRESHOLD = 0.05
        STUCK_WINDOW    = 8

        deadline    = time.time() + TAKEOFF_TIMEOUT
        stuck_start = time.time()
        last_alt    = 0.0

        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                print(f"{self._LOG_PREFIX} Waiting for altitude data...")
                time.sleep(1)
                continue

            print(f"{self._LOG_PREFIX} Altitude: {alt:.2f} m")

            if alt >= (target_altitude - self.ALTITUDE_TOLERANCE):
                print(f"{self._LOG_PREFIX} Target altitude reached.")
                return True

            if abs(alt - last_alt) > STUCK_THRESHOLD:
                last_alt    = alt
                stuck_start = time.time()
            elif time.time() - stuck_start > STUCK_WINDOW:
                print(f"{self._LOG_PREFIX} Takeoff stuck at {alt:.2f} m — aborting.")
                return False

            time.sleep(1)

        print(f"{self._LOG_PREFIX} Takeoff timed out after {TAKEOFF_TIMEOUT} s.")
        return False

    # ======================================================================
    # Velocity / motion primitives
    # ======================================================================

    def send_body_translation(self, vx: float, vy: float, vz: float):
        """
        Send a SET_POSITION_TARGET_LOCAL_NED velocity command (body/NED frame, m/s).

        Args:
            vx: Forward velocity  (positive = forward).
            vy: Right velocity    (positive = right).
            vz: Down velocity     (positive = down — MAVLink NED convention).
        """
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0,
        )

    def _move_direction(self, vx: float, vy: float, vz: float,
                        speed: float, duration: float):
        """Republish a velocity command for `duration` seconds, then hover."""
        end = time.time() + duration
        while time.time() < end:
            self.send_body_translation(vx * speed, vy * speed, vz * speed)
            time.sleep(0.2)
        self.hover()

    def move_forward(self, speed: float = 2.0, duration: float = 2.0) -> bool:
        """
        Fly forward at `speed` m/s for `duration` seconds, then hover.

        Voice triggers: "go forward", "fly forward", "move forward X seconds"
        """
        if not self._is_command_allowed("move_forward"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Moving forward at {speed} m/s for {duration} s...")
        self._move_direction(1.0, 0.0, 0.0, speed, duration)
        return True

    def move_backward(self, speed: float = 2.0, duration: float = 2.0) -> bool:
        """
        Fly backward at `speed` m/s for `duration` seconds, then hover.

        Voice triggers: "go back", "fly backward", "move backward X seconds"
        """
        if not self._is_command_allowed("move_backward"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Moving backward at {speed} m/s for {duration} s...")
        self._move_direction(-1.0, 0.0, 0.0, speed, duration)
        return True

    def move_left(self, speed: float = 2.0, duration: float = 2.0) -> bool:
        """
        Strafe left at `speed` m/s for `duration` seconds, then hover.

        Voice triggers: "go left", "move left", "fly left X seconds"
        """
        if not self._is_command_allowed("move_left"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Moving left at {speed} m/s for {duration} s...")
        self._move_direction(0.0, -1.0, 0.0, speed, duration)
        return True

    def move_right(self, speed: float = 2.0, duration: float = 2.0) -> bool:
        """
        Strafe right at `speed` m/s for `duration` seconds, then hover.

        Voice triggers: "go right", "move right", "fly right X seconds"
        """
        if not self._is_command_allowed("move_right"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Moving right at {speed} m/s for {duration} s...")
        self._move_direction(0.0, 1.0, 0.0, speed, duration)
        return True

    def ascend(self, distance: float = 5.0, speed: float = 1.5) -> bool:
        """
        Climb `distance` metres relative to current altitude at `speed` m/s.

        Voice triggers: "go up", "ascend", "climb X meters",
                        "go up X meters", "increase altitude"
        """
        if not self._is_command_allowed("ascend"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot ascend — drone is not airborne.")
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            print(f"{self._LOG_PREFIX} Cannot read current altitude.")
            return False

        target_alt = current_alt + distance
        print(f"{self._LOG_PREFIX} Ascending {distance} m to {target_alt:.1f} m...")

        timeout  = distance / speed + 10
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self._state == self._STATE_EMERGENCY:
                print(f"{self._LOG_PREFIX} Ascent halted — emergency state!")
                break

            alt = self._get_altitude()
            if alt is None:
                time.sleep(0.3)
                continue

            if alt >= target_alt - 0.3:
                self.hover()
                print(f"{self._LOG_PREFIX} Ascent complete — altitude {alt:.2f} m.")
                return True

            self.send_body_translation(0.0, 0.0, -speed)   # NED: negative Z = up
            time.sleep(0.2)

        self.hover()
        print(f"{self._LOG_PREFIX} Ascent timed out or interrupted.")
        return False

    def descend(self, distance: float = 5.0, speed: float = 1.0) -> bool:
        """
        Descend `distance` metres relative to current altitude at `speed` m/s.
        Stops at 1 m minimum to avoid unintentional landing.

        Voice triggers: "go down", "descend", "drop X meters",
                        "decrease altitude", "lower"
        """
        if not self._is_command_allowed("descend"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot descend — drone is not airborne.")
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            print(f"{self._LOG_PREFIX} Cannot read current altitude.")
            return False

        MIN_SAFE_ALT = 1.0
        target_alt   = max(current_alt - distance, MIN_SAFE_ALT)
        actual_drop  = current_alt - target_alt

        if actual_drop <= 0:
            print(f"{self._LOG_PREFIX} Already at or below {MIN_SAFE_ALT} m — not descending.")
            return False

        print(f"{self._LOG_PREFIX} Descending {actual_drop:.1f} m to {target_alt:.1f} m...")

        timeout  = actual_drop / speed + 10
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self._state == self._STATE_EMERGENCY:
                print(f"{self._LOG_PREFIX} Descent halted — emergency state!")
                break

            alt = self._get_altitude()
            if alt is None:
                time.sleep(0.3)
                continue

            if alt <= target_alt + 0.3:
                self.hover()
                print(f"{self._LOG_PREFIX} Descent complete — altitude {alt:.2f} m.")
                return True

            self.send_body_translation(0.0, 0.0, speed)    # NED: positive Z = down
            time.sleep(0.2)

        self.hover()
        print(f"{self._LOG_PREFIX} Descent timed out or interrupted.")
        return False

    def hover(self, duration: float = 0.0) -> bool:
        """
        Immediately stop all translational motion and hold position.

        Voice triggers: "stop", "hover", "stay", "hold", "freeze", "stop moving"
        """
        print(f"{self._LOG_PREFIX} Hovering — zeroing velocity.")
        self.send_body_translation(0.0, 0.0, 0.0)

        if duration > 0:
            deadline = time.time() + duration
            while time.time() < deadline:
                self.send_body_translation(0.0, 0.0, 0.0)
                time.sleep(0.2)
        return True

    # ======================================================================
    # Yaw / heading
    # ======================================================================

    def rotate_left(self, degrees: float = 90.0) -> bool:
        """
        Yaw counter-clockwise by `degrees`.

        Voice triggers: "turn left", "rotate left", "yaw left X degrees", "spin left"
        """
        if not self._is_command_allowed("rotate_left"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot rotate — drone is not airborne.")
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Rotating left {degrees}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(degrees), p2=20.0, p3=-1.0, p4=1.0,
        )

    def rotate_right(self, degrees: float = 90.0) -> bool:
        """
        Yaw clockwise by `degrees`.

        Voice triggers: "turn right", "rotate right", "yaw right X degrees", "spin right"
        """
        if not self._is_command_allowed("rotate_right"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot rotate — drone is not airborne.")
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Rotating right {degrees}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(degrees), p2=20.0, p3=1.0, p4=1.0,
        )

    def set_heading(self, heading_degrees: float) -> bool:
        """
        Point to an absolute compass bearing (0–360°).

        Voice triggers: "face north/south/east/west",
                        "turn to X degrees", "heading X"
        """
        if not self._is_command_allowed("set_heading"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot set heading — drone is not airborne.")
            return False
        self._accept_command()
        heading_degrees %= 360.0
        print(f"{self._LOG_PREFIX} Setting heading to {heading_degrees:.0f}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(heading_degrees), p2=20.0, p3=1.0, p4=0.0,
        )

    # ======================================================================
    # Flight mode shortcuts
    # ======================================================================

    def set_loiter(self) -> bool:
        """
        Switch to LOITER — GPS-assisted position hold with yaw control.

        Voice triggers: "loiter", "hold position", "hover in place"
        """
        if not self._is_command_allowed("set_loiter"):
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Switching to LOITER mode...")
        return self._set_mode("LOITER")

    def set_position_hold(self) -> bool:
        """
        Switch to POSHOLD — full position, velocity, and altitude hold.

        Voice triggers: "position hold", "pos hold", "lock position"
        """
        if not self._is_command_allowed("set_position_hold"):
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Switching to POSHOLD mode...")
        return self._set_mode("POSHOLD")

    # ======================================================================
    # Waypoint management
    # ======================================================================

    def save_waypoint(self) -> bool:
        """
        Save the current GPS position as a named waypoint.

        Voice triggers: "save waypoint", "mark current location",
                        "save this spot", "mark position"
        """
        loc = self.get_location()
        if loc is None:
            print(f"{self._LOG_PREFIX} Cannot save waypoint — location unavailable.")
            return False

        wp = {
            "index": len(self._waypoints),
            "lat":   loc["lat"],
            "lon":   loc["lon"],
            "alt":   loc["alt_relative"],
        }
        self._waypoints.append(wp)
        print(f"{self._LOG_PREFIX} Waypoint {wp['index']} saved: "
              f"({wp['lat']:.6f}, {wp['lon']:.6f}) @ {wp['alt']:.1f} m")
        return True

    def goto_last_waypoint(self) -> bool:
        """
        Fly to the most recently saved waypoint.

        Voice triggers: "go to waypoint", "fly to waypoint",
                        "navigate to waypoint", "head to waypoint"
        """
        if not self._waypoints:
            print(f"{self._LOG_PREFIX} No waypoints saved.")
            return False
        wp = self._waypoints[-1]
        print(f"{self._LOG_PREFIX} Flying to last waypoint {wp['index']} "
              f"({wp['lat']:.6f}, {wp['lon']:.6f}) @ {wp['alt']:.1f} m...")
        return self.goto_waypoint(wp["lat"], wp["lon"], wp["alt"])

    def export_mission(self) -> bool:
        """
        Export all saved waypoints to a plain-text mission file.

        Voice triggers: "save flight plan", "export waypoints",
                        "export mission", "save mission text"
        """
        if not self._waypoints:
            print(f"{self._LOG_PREFIX} No waypoints to export.")
            return False

        filename = "drone_mission.waypoints"
        with open(filename, "w") as f:
            f.write("QGC WPL 110\n")
            for i, wp in enumerate(self._waypoints):
                # QGC waypoint format: index current_wp frame command param1-4 lat lon alt autocontinue
                f.write(
                    f"{i}\t0\t3\t16\t0\t0\t0\t0\t"
                    f"{wp['lat']:.7f}\t{wp['lon']:.7f}\t{wp['alt']:.2f}\t1\n"
                )
        print(f"{self._LOG_PREFIX} Mission exported → {filename} "
              f"({len(self._waypoints)} waypoints)")
        return True

    def start_mission_if_waypoints_exist(self) -> bool:
        """
        Switch to AUTO and run the saved waypoint mission, but only if
        at least one waypoint has been saved this session.

        Voice triggers: "start mission", "begin mission", "run mission"
        """
        if not self._waypoints:
            print(f"{self._LOG_PREFIX} Cannot start mission — no waypoints saved.")
            return False
        print(f"{self._LOG_PREFIX} Starting mission with {len(self._waypoints)} waypoint(s)...")
        return self._set_mode("AUTO")

    # ======================================================================
    # Emergency — override to add LOITER pause before RTL/LAND
    # ======================================================================

    def trigger_emergency_safe_state(self) -> str:
        """
        Cancel all pending timers, pause in LOITER, then RTL or LAND.

        Returns "RTL_ACTIVATED" or "LAND_ACTIVATED".
        """
        print(f"[CRITICAL] {self._LOG_PREFIX} EMERGENCY SAFE STATE TRIGGERED!")

        if self._hover_timer is not None:
            self._hover_timer.cancel()
        with self._state_lock:
            self._state = self._STATE_EMERGENCY

        self._set_mode("LOITER")
        time.sleep(1)

        alt = self._get_altitude()
        if alt is not None and alt > 3.0:
            self._set_mode("RTL")
            return "RTL_ACTIVATED"

        self._set_mode("LAND")
        return "LAND_ACTIVATED"

    # ======================================================================
    # Connection lifecycle — override to stop hover watchdog
    # ======================================================================

    def close_connection(self):
        self._hover_heartbeat_active = False
        super().close_connection()
