"""
plane_driver.py — ArduPlane / fixed-wing driver.

Extends MAVLinkBaseDriver with:
  - TAKEOFF / CRUISE / FBWA mode management
  - Airspeed and throttle control
  - Banking turn commands (left / right)
  - Climb / descend via pitch attitude
  - Loiter (circle hold) and AUTO mission modes
  - Waypoint save / go-to / export
"""

import time
import threading
from typing import Optional
from pymavlink import mavutil

from base_driver import MAVLinkBaseDriver


class MAVLinkPlaneDriver(MAVLinkBaseDriver):

    # ── Fixed-wing speed defaults ────────────────────────────────────────
    DEFAULT_CRUISE_SPEED   = 15.0   # m/s — typical ArduPlane cruise
    DEFAULT_CLIMB_SPEED    = 12.0   # m/s — airspeed during climb
    DEFAULT_LOITER_RADIUS  = 50.0   # m   — loiter circle radius

    # ── Altitude tolerance for climb/descend tracking ────────────────────
    ALTITUDE_TOLERANCE = 2.0        # metres (planes are less precise than copters)

    _LOG_PREFIX = "[PLANE]"

    # ======================================================================
    # Construction
    # ======================================================================

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        super().__init__(connection_string)

        # Planes don't use the copter altitude-gate but share state machine
        self._target_altitude: Optional[float] = None

        # Waypoint store (same format as drone)
        self._waypoints: list[dict] = []

        # Background fuel / battery monitor
        self._monitor_active = True
        self._monitor_thread = threading.Thread(
            target=self._plane_monitor, daemon=True
        )
        self._monitor_thread.start()

    # ======================================================================
    # Background monitor
    # ======================================================================

    def _plane_monitor(self):
        """
        Silent background loop: battery / fuel monitoring.
        Triggers emergency RTL on critical battery.
        """
        has_warned_low     = False
        has_warned_anomaly = False

        while self._monitor_active:
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
                              f"but voltage is SAFE at {voltage:.2f}V.")
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

            time.sleep(5.0)   # planes check less frequently than copters

    # ======================================================================
    # Pre-arm — plane skips EKF, uses GPS only
    # ======================================================================

    def _wait_for_armable(self, timeout: float = 30) -> bool:
        """
        Block until GPS fix ≥ 3 (planes don't require EKF_STATUS_REPORT).
        """
        print(f"{self._LOG_PREFIX} Waiting for GPS fix...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            gps = self._get_cached("GPS_RAW_INT")
            if gps and gps.fix_type >= 3:
                print(f"{self._LOG_PREFIX} ✅ GPS 3-D fix — vehicle is armable.")
                return True
            if gps:
                print(f"{self._LOG_PREFIX} GPS not ready — fix_type={gps.fix_type}, "
                      f"sats={gps.satellites_visible}")
            else:
                print(f"{self._LOG_PREFIX} Waiting for GPS_RAW_INT...")
            time.sleep(1)

        print(f"{self._LOG_PREFIX} Pre-arm GPS timeout after {timeout:.0f} s.")
        return False

    # ======================================================================
    # Arming / takeoff — plane-specific sequence
    # ======================================================================

    def arm_vehicle(self) -> bool:
        """
        Arm the plane: battery check → GPS fix → TAKEOFF mode → arm.
        """
        if self._is_armed():
            print(f"{self._LOG_PREFIX} Vehicle is already armed.")
            return True

        safe, reason = self._check_battery_safe()
        if not safe:
            print(f"{self._LOG_PREFIX} Cannot arm — battery safety block: {reason}")
            return False

        if not self._wait_for_armable(timeout=30):
            print(f"{self._LOG_PREFIX} Cannot arm — GPS check failed.")
            return False

        # Planes typically launch from TAKEOFF or FBWA mode
        print(f"{self._LOG_PREFIX} Setting TAKEOFF mode...")
        if not self._set_mode("TAKEOFF"):
            print(f"{self._LOG_PREFIX} TAKEOFF mode unavailable — trying FBWA...")
            if not self._set_mode("FBWA"):
                print(f"{self._LOG_PREFIX} Could not enter launch mode — aborting arm.")
                return False

        time.sleep(0.5)

        print(f"{self._LOG_PREFIX} Sending arm command...")
        with self._cache_lock:
            self._message_cache.pop("COMMAND_ACK", None)

        result = self._send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            p1=1, p2=21196,
        )

        if result and self._wait_until_armed(timeout=10):
            print(f"{self._LOG_PREFIX} Armed successfully.")
            return True

        print(f"{self._LOG_PREFIX} Arming failed — check Mission Planner for PreArm errors.")
        return False

    def execute_takeoff(self, target_altitude: float = 30.0) -> bool:
        """
        Command a fixed-wing hand-launch or runway takeoff to target_altitude.

        Voice triggers: "take off", "launch", "start flying", "lift off"
        """
        if not self._is_armed():
            print(f"{self._LOG_PREFIX} CRITICAL: Takeoff rejected — arm first.")
            return False

        print(f"{self._LOG_PREFIX} Commanding fixed-wing takeoff to {target_altitude:.0f} m...")

        # MAV_CMD_NAV_TAKEOFF for fixed wing: pitch angle, empty, empty, yaw, lat, lon, alt
        success = self._send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            p1=15.0,           # pitch degrees
            p7=target_altitude
        )
        if not success:
            print(f"{self._LOG_PREFIX} Takeoff command rejected by FC.")
            return False

        # Monitor altitude climb
        TAKEOFF_TIMEOUT = 60    # planes take longer to climb
        deadline        = time.time() + TAKEOFF_TIMEOUT
        last_alt        = 0.0
        stuck_start     = time.time()

        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                time.sleep(1)
                continue

            print(f"{self._LOG_PREFIX} Altitude: {alt:.1f} m")

            if alt >= target_altitude - self.ALTITUDE_TOLERANCE:
                print(f"{self._LOG_PREFIX} ✅ Cruise altitude reached. Switching to CRUISE mode...")
                self._set_mode("CRUISE")
                with self._state_lock:
                    self._state = self._STATE_FLYING
                return True

            if abs(alt - last_alt) > 0.2:
                last_alt    = alt
                stuck_start = time.time()
            elif time.time() - stuck_start > 15:
                print(f"{self._LOG_PREFIX} Climb stuck at {alt:.1f} m — aborting.")
                return False

            time.sleep(1)

        print(f"{self._LOG_PREFIX} Takeoff timed out after {TAKEOFF_TIMEOUT} s.")
        return False

    # ======================================================================
    # Flight mode shortcuts
    # ======================================================================

    def set_cruise(self) -> bool:
        """
        Switch to CRUISE — fixed-wing speed and heading hold.

        Voice triggers: "cruise mode", "set cruise", "fly cruise"
        """
        print(f"{self._LOG_PREFIX} Switching to CRUISE mode...")
        return self._set_mode("CRUISE")

    def set_fbwa(self) -> bool:
        """
        Switch to FBWA (Fly-By-Wire-A) — pilot controls attitude limits.

        Voice triggers: "manual mode", "fly by wire", "fbwa"
        """
        print(f"{self._LOG_PREFIX} Switching to FBWA mode...")
        return self._set_mode("FBWA")

    def set_auto(self) -> bool:
        """
        Switch to AUTO — follow uploaded waypoint mission.

        Voice triggers: "auto mode", "start mission", "follow mission"
        """
        print(f"{self._LOG_PREFIX} Switching to AUTO mode...")
        success = self._set_mode("AUTO")
        if success:
            with self._state_lock:
                self._state = self._STATE_FLYING
        return success

    def set_loiter(self) -> bool:
        """
        Switch to LOITER — orbit a GPS point at DEFAULT_LOITER_RADIUS.

        Voice triggers: "loiter", "hold position", "circle", "orbit"
        """
        if not self._is_command_allowed("set_loiter"):
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Switching to LOITER mode...")
        return self._set_mode("LOITER")

    def set_guided(self) -> bool:
        """
        Switch to GUIDED — allows MAVLink-commanded navigation.

        Voice triggers: "guided mode", "enable guided"
        """
        print(f"{self._LOG_PREFIX} Switching to GUIDED mode...")
        return self._set_mode("GUIDED")

    def set_manual(self) -> bool:
        """
        Switch to MANUAL — full pilot control, no stabilization.

        Voice triggers: "manual mode", "switch to manual"
        """
        print(f"{self._LOG_PREFIX} Switching to MANUAL mode...")
        return self._set_mode("MANUAL")

    def has_mission_uploaded(self) -> bool:
        """
        Return True if the flight controller reports at least one
        mission waypoint loaded (via MISSION_COUNT), independent of
        this session's in-memory _waypoints list.
        """
        with self._cache_lock:
            self._message_cache.pop("MISSION_COUNT", None)

        self.master.mav.mission_request_list_send(
            self.master.target_system, self.master.target_component
        )

        deadline = time.time() + 3
        while time.time() < deadline:
            msg = self._get_cached("MISSION_COUNT")
            if msg is not None:
                print(f"{self._LOG_PREFIX} Flight controller reports {msg.count} mission item(s).")
                return msg.count > 0
            time.sleep(0.1)

        print(f"{self._LOG_PREFIX} No MISSION_COUNT response — assuming no mission loaded.")
        return False

    # ======================================================================
    # Attitude / climb / descend commands
    # ======================================================================

    def climb_to_altitude(self, target_altitude: float, speed: float = None) -> bool:
        """
        Climb to an absolute altitude above home in GUIDED mode.

        Voice triggers: "climb to X meters", "go up to X meters",
                        "ascend to altitude X"
        """
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot climb — plane is not airborne.")
            return False

        if speed is None:
            speed = self.DEFAULT_CLIMB_SPEED

        current_alt = self._get_altitude()
        if current_alt is None:
            print(f"{self._LOG_PREFIX} Cannot read altitude.")
            return False

        if target_altitude <= current_alt:
            print(f"{self._LOG_PREFIX} Already at or above {target_altitude:.1f} m.")
            return False

        print(f"{self._LOG_PREFIX} Climbing from {current_alt:.1f} m to {target_altitude:.1f} m "
              f"at {speed:.1f} m/s airspeed...")

        if not self._set_mode("GUIDED"):
            return False

        self.set_airspeed(speed)

        loc = self.get_location()
        if loc is None:
            print(f"{self._LOG_PREFIX} Cannot read location for climb command.")
            return False

        # Command current lat/lon at new altitude
        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_111111111000,
            int(loc["lat"] * 1e7),
            int(loc["lon"] * 1e7),
            target_altitude,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )

        # Wait for altitude
        deadline = time.time() + 120
        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                time.sleep(1)
                continue
            print(f"{self._LOG_PREFIX} Altitude: {alt:.1f} m")
            if alt >= target_altitude - self.ALTITUDE_TOLERANCE:
                print(f"{self._LOG_PREFIX} ✅ Target altitude {target_altitude:.1f} m reached.")
                return True
            time.sleep(1)

        print(f"{self._LOG_PREFIX} Climb timed out.")
        return False

    def descend_to_altitude(self, target_altitude: float) -> bool:
        """
        Descend to an absolute altitude above home in GUIDED mode.

        Voice triggers: "descend to X meters", "go down to X meters",
                        "reduce altitude to X"
        """
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot descend — plane is not airborne.")
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            print(f"{self._LOG_PREFIX} Cannot read altitude.")
            return False

        MIN_SAFE_ALT = 50.0   # planes need more clearance than copters
        target_altitude = max(target_altitude, MIN_SAFE_ALT)

        if target_altitude >= current_alt:
            print(f"{self._LOG_PREFIX} Already at or below {target_altitude:.1f} m.")
            return False

        print(f"{self._LOG_PREFIX} Descending from {current_alt:.1f} m "
              f"to {target_altitude:.1f} m...")

        if not self._set_mode("GUIDED"):
            return False

        loc = self.get_location()
        if loc is None:
            return False

        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_111111111000,
            int(loc["lat"] * 1e7),
            int(loc["lon"] * 1e7),
            target_altitude,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )

        deadline = time.time() + 120
        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                time.sleep(1)
                continue
            print(f"{self._LOG_PREFIX} Altitude: {alt:.1f} m")
            if alt <= target_altitude + self.ALTITUDE_TOLERANCE:
                print(f"{self._LOG_PREFIX} ✅ Target altitude {target_altitude:.1f} m reached.")
                return True
            time.sleep(1)

        print(f"{self._LOG_PREFIX} Descent timed out.")
        return False

    # ======================================================================
    # Turning commands
    # ======================================================================

    def turn_left(self, heading_degrees: float = None, bank_angle: float = 30.0) -> bool:
        """
        Bank left to an absolute heading, or just set a left bank via MAV_CMD_DO_SET_ROI.
        Uses MAV_CMD_CONDITION_YAW when a heading is specified.

        Voice triggers: "turn left", "bank left", "left turn"
        """
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot turn — plane is not airborne.")
            return False
        if heading_degrees is not None:
            print(f"{self._LOG_PREFIX} Turning left to {heading_degrees:.0f}°...")
            return self._send_command_long(
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                p1=float(heading_degrees), p2=0.0, p3=-1.0, p4=0.0,  # absolute CCW
            )
        # No heading specified — just command a standard rate turn (bank)
        print(f"{self._LOG_PREFIX} Initiating left bank at {bank_angle:.0f}° in CRUISE...")
        self._set_mode("CRUISE")
        return True

    def turn_right(self, heading_degrees: float = None, bank_angle: float = 30.0) -> bool:
        """
        Bank right to an absolute heading, or just set a right bank.

        Voice triggers: "turn right", "bank right", "right turn"
        """
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot turn — plane is not airborne.")
            return False
        if heading_degrees is not None:
            print(f"{self._LOG_PREFIX} Turning right to {heading_degrees:.0f}°...")
            return self._send_command_long(
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                p1=float(heading_degrees), p2=0.0, p3=1.0, p4=0.0,   # absolute CW
            )
        print(f"{self._LOG_PREFIX} Initiating right bank at {bank_angle:.0f}° in CRUISE...")
        self._set_mode("CRUISE")
        return True

    def set_heading(self, heading_degrees: float) -> bool:
        """
        Fly to a specific compass bearing (0–360°).

        Voice triggers: "set heading X", "face north/south/east/west",
                        "heading X degrees"
        """
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot set heading — plane is not airborne.")
            return False
        heading_degrees %= 360.0
        print(f"{self._LOG_PREFIX} Setting heading to {heading_degrees:.0f}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(heading_degrees), p2=0.0, p3=1.0, p4=0.0,
        )

    # ======================================================================
    # Waypoint management (QGC format, same as drone)
    # ======================================================================

    def save_waypoint(self) -> bool:
        """
        Save the current GPS position as a named waypoint.

        Voice triggers: "save waypoint", "mark current location", "save this spot"
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
        Fly to the most recently saved waypoint in GUIDED mode.

        Voice triggers: "go to waypoint", "fly to waypoint",
                        "navigate to waypoint"
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
        Export all saved waypoints to a QGC-compatible .waypoints file.

        Voice triggers: "save flight plan", "export waypoints",
                        "export mission", "save mission text"
        """
        if not self._waypoints:
            print(f"{self._LOG_PREFIX} No waypoints to export.")
            return False

        filename = "plane_mission.waypoints"
        with open(filename, "w") as f:
            f.write("QGC WPL 110\n")
            for i, wp in enumerate(self._waypoints):
                f.write(
                    f"{i}\t0\t3\t16\t0\t0\t0\t0\t"
                    f"{wp['lat']:.7f}\t{wp['lon']:.7f}\t{wp['alt']:.2f}\t1\n"
                )
        print(f"{self._LOG_PREFIX} Mission exported → {filename} "
              f"({len(self._waypoints)} waypoints)")
        return True

    # ======================================================================
    # Emergency — plane goes straight to RTL (no LOITER pause)
    # ======================================================================

    def trigger_emergency_safe_state(self) -> str:
        """
        Immediately activate RTL. No LOITER pause — planes must keep flying.

        Returns "RTL_ACTIVATED".
        """
        print(f"[CRITICAL] {self._LOG_PREFIX} EMERGENCY SAFE STATE TRIGGERED!")

        if self._hover_timer is not None:
            self._hover_timer.cancel()
        with self._state_lock:
            self._state = self._STATE_EMERGENCY

        self._set_mode("RTL")
        return "RTL_ACTIVATED"

    # ======================================================================
    # Connection lifecycle
    # ======================================================================

    def close_connection(self):
        self._monitor_active = False
        super().close_connection()
