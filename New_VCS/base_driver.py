"""
base_driver.py — Shared MAVLink infrastructure for all vehicle types.

Provides: connection, message cache, ACK handling, mode changes,
battery safety, pre-arm checks, arming/disarming, telemetry queries,
emergency safe state, and connection lifecycle.

Subclassed by:
    drone_driver.py  → MAVLinkDroneDriver
    plane_driver.py  → MAVLinkPlaneDriver
"""

import math
import time
import threading
from typing import Optional
from pymavlink import mavutil
from pymavlink.mavutil import mavfile


MIN_GPS_FIX       = 3
HEARTBEAT_TIMEOUT = 10


class MAVLinkBaseDriver:

    # ── Battery safety thresholds ─────────────────────────────────────────
    BATTERY_MIN_PCT     = 20     # % — block arming below this
    BATTERY_WARN_PCT    = 30     # % — warn but still allow arming
    BATTERY_MIN_VOLTAGE = 10.5   # V  — absolute floor (3S LiPo = 3.5 V/cell)

    # ── Internal state labels ─────────────────────────────────────────────
    _STATE_IDLE      = "IDLE"
    _STATE_LOCKED    = "LOCKED"
    _STATE_READY     = "READY"
    _STATE_FLYING    = "FLYING"
    _STATE_HOVER     = "HOVER"
    _STATE_EMERGENCY = "EMERGENCY"

    # ── Subclass must declare a log prefix ───────────────────────────────
    _LOG_PREFIX = "[BASE]"

    # ======================================================================
    # Construction / connection
    # ======================================================================

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        print(f"{self._LOG_PREFIX} Connecting to vehicle on: {connection_string}")
        self.master: mavfile = mavutil.mavlink_connection(connection_string)  # type: ignore

        print(f"{self._LOG_PREFIX} Waiting for heartbeat...")
        msg = self.master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
        if msg is None:
            raise ConnectionError(
                f"No heartbeat received within {HEARTBEAT_TIMEOUT} s. "
                "Is the SITL or flight controller running?"
            )
        print(
            f"{self._LOG_PREFIX} Heartbeat received — system {self.master.target_system}, "
            f"component {self.master.target_component}"
        )

        # ── Message cache ──────────────────────────────────────────────
        self._message_cache: dict = {}
        self._cache_lock = threading.Lock()
        self._listener_running = True
        self._listener_thread = threading.Thread(
            target=self._message_listener, daemon=True
        )
        self._listener_thread.start()

        # ── Safety state machine ───────────────────────────────────────
        self._state           = self._STATE_IDLE
        self._state_lock      = threading.Lock()
        self._hover_timer: Optional[threading.Timer] = None

    # ======================================================================
    # Background message listener
    # ======================================================================

    def _message_listener(self):
        """Continuously read MAVLink messages and keep the cache fresh."""
        while self._listener_running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue
                with self._cache_lock:
                    self._message_cache[msg.get_type()] = msg
            except OSError:
                break
            except Exception as exc:
                print(f"{self._LOG_PREFIX} Cache error: {exc}")

    def _get_cached(self, msg_type: str):
        """Return the latest cached message of the given type, or None."""
        with self._cache_lock:
            return self._message_cache.get(msg_type)

    # ======================================================================
    # Low-level MAVLink helpers
    # ======================================================================

    def _send_command_long(self, command, p1=0.0, p2=0.0, p3=0.0,
                           p4=0.0, p5=0.0, p6=0.0, p7=0.0) -> bool:
        """Send MAV_CMD via COMMAND_LONG and wait for ACK."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            command, 0,
            p1, p2, p3, p4, p5, p6, p7,
        )
        return self._wait_command_ack(command)

    def _wait_command_ack(self, command, timeout: float = 10) -> bool:
        """Block until COMMAND_ACK for the given command ID arrives."""
        with self._cache_lock:
            self._message_cache.pop("COMMAND_ACK", None)

        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._get_cached("COMMAND_ACK")
            if msg and msg.command == command:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    return True
                print(f"{self._LOG_PREFIX} Command {command} rejected — result code {msg.result}")
                return False
            time.sleep(0.05)

        print(f"{self._LOG_PREFIX} Timeout waiting for ACK of command {command}")
        return False

    def _set_mode(self, mode_name: str) -> bool:
        """Change the flight mode by name (e.g. 'GUIDED', 'LAND', 'RTL')."""
        mode_mapping = self.master.mode_mapping()
        if not mode_mapping:
            raise RuntimeError("Failed to retrieve mode mapping from vehicle.")
        if mode_name not in mode_mapping:
            raise ValueError(f"Mode '{mode_name}' is not available on this vehicle.")

        mode_id = mode_mapping[mode_name]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

        deadline = time.time() + 5
        while time.time() < deadline:
            msg = self._get_cached("HEARTBEAT")
            if msg and msg.custom_mode == mode_id:
                return True
            time.sleep(0.1)

        print(f"{self._LOG_PREFIX} Mode change to {mode_name} not confirmed in time.")
        return False

    # ======================================================================
    # Telemetry helpers
    # ======================================================================

    def _get_altitude(self) -> Optional[float]:
        """Return current relative altitude in metres, or None."""
        msg = self._get_cached("GLOBAL_POSITION_INT")
        return msg.relative_alt / 1000.0 if msg else None

    def _get_global_position(self):
        """Return latest GLOBAL_POSITION_INT message, or None."""
        return self._get_cached("GLOBAL_POSITION_INT")

    def _is_armed(self) -> bool:
        """Return True if the vehicle reports as armed."""
        msg = self._get_cached("HEARTBEAT")
        if msg:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        return False

    def _wait_until_armed(self, timeout: float = 10) -> bool:
        """Poll HEARTBEAT until armed flag is set, with direct-stream fallback."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_armed():
                return True
            msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                with self._cache_lock:
                    self._message_cache['HEARTBEAT'] = msg
                if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    return True
            time.sleep(0.2)
        return False

    def _wait_for_armable(self, timeout: float = 30) -> bool:
        """
        Block until pre-arm checks pass (GPS 3-D fix + EKF attitude/velocity/position).
        Prints per-check status every second.
        """
        print(f"{self._LOG_PREFIX} Waiting for pre-arm checks (GPS + EKF)...")
        deadline = time.time() + timeout
        gps_ok = ekf_ok = False

        while time.time() < deadline:
            # ── GPS check ─────────────────────────────────────────────
            gps = self._get_cached("GPS_RAW_INT")
            if gps:
                if gps.fix_type >= MIN_GPS_FIX:
                    gps_ok = True
                else:
                    fix_names = {0: "NO_GPS", 1: "NO_FIX", 2: "2D_FIX",
                                 3: "3D_FIX", 4: "DGPS", 5: "RTK_FLOAT",
                                 6: "RTK_FIXED"}
                    print(
                        f"{self._LOG_PREFIX} GPS not ready — "
                        f"fix_type={fix_names.get(gps.fix_type, gps.fix_type)}, "
                        f"sats={gps.satellites_visible}"
                    )
            else:
                print(f"{self._LOG_PREFIX} GPS — no GPS_RAW_INT message received yet.")

            # ── EKF check ─────────────────────────────────────────────
            ekf = self._get_cached("EKF_STATUS_REPORT")
            if ekf:
                attitude_ok = bool(ekf.flags & 0x01)
                velocity_ok = bool(ekf.flags & 0x02)
                position_ok = bool(ekf.flags & 0x04)
                if attitude_ok and velocity_ok and position_ok:
                    ekf_ok = True
                else:
                    print(
                        f"{self._LOG_PREFIX} EKF not ready — "
                        f"attitude={'OK' if attitude_ok else 'WAIT'}, "
                        f"velocity={'OK' if velocity_ok else 'WAIT'}, "
                        f"position={'OK' if position_ok else 'WAIT'} "
                        f"(raw flags=0x{ekf.flags:02X})"
                    )
            else:
                print(f"{self._LOG_PREFIX} EKF — no EKF_STATUS_REPORT yet "
                      "(normal for first ~5 s on SITL).")

            if gps_ok and ekf_ok:
                print(f"{self._LOG_PREFIX} ✅ GPS + EKF ready — vehicle is armable.")
                return True

            time.sleep(1)

        print(
            f"{self._LOG_PREFIX} Pre-arm timeout after {timeout:.0f} s — "
            f"GPS={'OK' if gps_ok else 'FAILED'}, EKF={'OK' if ekf_ok else 'FAILED'}"
        )
        return False

    # ======================================================================
    # Safety state machine — shared
    # ======================================================================

    def _is_command_allowed(self, command_name: str) -> bool:
        """Return True if a movement command is allowed in the current state."""
        with self._state_lock:
            if self._state == self._STATE_LOCKED:
                alt = self._get_altitude()
                remaining = (
                    f"{self._target_altitude - alt:.1f} m remaining"
                    if alt is not None else "altitude unknown"
                )
                print(
                    f"{self._LOG_PREFIX} '{command_name}' BLOCKED — vehicle has not "
                    f"reached target altitude. {remaining}. Please wait."
                )
                return False
        return True

    def _accept_command(self):
        """Cancel the hover timer and transition state to FLYING."""
        if self._hover_timer is not None:
            self._hover_timer.cancel()
            self._hover_timer = None
        with self._state_lock:
            if self._state in (self._STATE_READY, self._STATE_HOVER):
                self._state = self._STATE_FLYING
                print(f"{self._LOG_PREFIX} Command accepted — state → FLYING.")

    # ======================================================================
    # Battery safety
    # ======================================================================

    def _check_battery_safe(self) -> tuple[bool, str]:
        """
        Read battery status and decide if it is safe to arm.
        Returns (safe: bool, reason: str).
        """
        batt = self.get_battery()

        if batt is None:
            print(f"{self._LOG_PREFIX} Battery data unavailable — proceeding without check.")
            return True, "NO_DATA"

        voltage   = batt.get("voltage_v")
        remaining = batt.get("remaining_pct", -1)

        if voltage is not None and voltage < self.BATTERY_MIN_VOLTAGE:
            reason = (
                f"BATTERY_CRITICAL_VOLTAGE: {voltage:.2f} V "
                f"(minimum {self.BATTERY_MIN_VOLTAGE} V)"
            )
            print(f"{self._LOG_PREFIX} Arming BLOCKED — {reason}")
            return False, reason

        if remaining != -1:
            if remaining < self.BATTERY_MIN_PCT:
                reason = f"BATTERY_TOO_LOW: {remaining}% (minimum {self.BATTERY_MIN_PCT}%)"
                print(f"{self._LOG_PREFIX} Arming BLOCKED — {reason}")
                return False, reason

            if remaining < self.BATTERY_WARN_PCT:
                print(
                    f"{self._LOG_PREFIX} Battery warning: {remaining}% remaining "
                    f"(below {self.BATTERY_WARN_PCT}% threshold). Proceeding — plan a short flight."
                )
                return True, f"BATTERY_WARNING_{remaining}pct"

        v_str = f"{voltage:.2f} V" if voltage is not None else "unknown V"
        p_str = f"{remaining}%" if remaining != -1 else "unknown %"
        print(f"{self._LOG_PREFIX} Battery OK — {v_str}, {p_str}")
        return True, f"BATTERY_OK_{p_str}"

    # ======================================================================
    # Arming / disarming
    # ======================================================================

    def arm_vehicle(self) -> bool:
        """
        Arm the motors using the correct ArduPilot sequence:
        battery check → pre-arm sensors → GUIDED mode → arm command → confirm.
        """
        if self._is_armed():
            print(f"{self._LOG_PREFIX} Vehicle is already armed.")
            return True

        safe, reason = self._check_battery_safe()
        if not safe:
            print(f"{self._LOG_PREFIX} Cannot arm — battery safety block: {reason}")
            return False

        if not self._wait_for_armable(timeout=30):
            print(f"{self._LOG_PREFIX} Cannot arm — pre-arm checks failed.")
            return False

        print(f"{self._LOG_PREFIX} Setting GUIDED mode...")
        if not self._set_mode("GUIDED"):
            print(f"{self._LOG_PREFIX} Could not enter GUIDED mode — aborting arm.")
            return False

        time.sleep(1.0)

        print(f"{self._LOG_PREFIX} Sending arm command...")
        with self._cache_lock:
            self._message_cache.pop("COMMAND_ACK", None)

        result = self._send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            p1=1, p2=21196,   # force-arm magic number — skips soft checks on SITL
        )

        if result and self._wait_until_armed(timeout=10):
            print(f"{self._LOG_PREFIX} Armed successfully.")
            return True

        print(f"{self._LOG_PREFIX} Arming failed — check Mission Planner for PreArm errors.")
        return False

    def disarm_vehicle(self) -> bool:
        """Disarm the motors and reset the safety state to IDLE."""
        if not self._is_armed():
            print(f"{self._LOG_PREFIX} Vehicle is already disarmed.")
            return False

        if self._hover_timer is not None:
            self._hover_timer.cancel()
        with self._state_lock:
            self._state = self._STATE_IDLE

        print(f"{self._LOG_PREFIX} Disarming...")
        result = self._send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=0
        )

        if result:
            deadline = time.time() + 5
            while time.time() < deadline:
                if not self._is_armed():
                    print(f"{self._LOG_PREFIX} Disarmed.")
                    return True
                time.sleep(0.5)

        print(f"{self._LOG_PREFIX} Disarm failed.")
        return False

    # ======================================================================
    # Public — status / state
    # ======================================================================

    def get_safety_state(self) -> str:
        """Return the current safety state string."""
        with self._state_lock:
            return self._state

    def is_flying(self) -> bool:
        """Return True if the vehicle is airborne (altitude > 0.3 m)."""
        alt = self._get_altitude()
        return False if alt is None else alt > 0.3

    # ======================================================================
    # Public — status queries (voice feedback)
    # ======================================================================

    def get_altitude(self) -> Optional[float]:
        """
        Return current relative altitude in metres, or None.

        Voice triggers: "what is my altitude", "how high am I",
                        "altitude check", "current altitude"
        """
        alt = self._get_altitude()
        if alt is not None:
            print(f"{self._LOG_PREFIX} Current altitude: {alt:.2f} m")
        return alt

    def get_heading(self) -> Optional[float]:
        """
        Return current compass heading in degrees (0–360), or None.

        Voice triggers: "what direction am I facing", "what is my heading",
                        "compass heading", "which way am I pointing"
        """
        msg = self._get_global_position()
        if msg:
            heading = msg.hdg / 100.0
            print(f"{self._LOG_PREFIX} Current heading: {heading:.1f}°")
            return heading
        return None

    def get_groundspeed(self) -> Optional[float]:
        """
        Return current ground speed in m/s, or None.

        Voice triggers: "how fast am I going", "current speed", "ground speed"
        """
        msg = self._get_global_position()
        if msg:
            vx    = msg.vx / 100.0
            vy    = msg.vy / 100.0
            speed = math.sqrt(vx ** 2 + vy ** 2)
            print(f"{self._LOG_PREFIX} Current ground speed: {speed:.2f} m/s")
            return speed
        return None

    def get_battery(self) -> Optional[dict]:
        """
        Return battery status as a dict, or None if unavailable.

        Returns:
            { "voltage_v": float, "current_a": float, "remaining_pct": int }

        Voice triggers: "check battery", "battery level", "battery status"
        """
        msg = self._get_cached("BATTERY_STATUS")
        if msg:
            voltage   = msg.voltages[0] / 1000.0 if msg.voltages[0] != 65535 else None
            current   = msg.current_battery / 100.0 if msg.current_battery != -1 else -1
            remaining = msg.battery_remaining
            return {"voltage_v": voltage, "current_a": current, "remaining_pct": remaining}

        msg = self._get_cached("SYS_STATUS")
        if msg:
            voltage   = msg.voltage_battery / 1000.0
            current   = msg.current_battery / 100.0 if msg.current_battery != -1 else -1
            remaining = msg.battery_remaining
            return {"voltage_v": voltage, "current_a": current, "remaining_pct": remaining}

        print(f"{self._LOG_PREFIX} Battery status unavailable.")
        return None

    def get_gps_status(self) -> Optional[dict]:
        """
        Return GPS fix status, or None.

        Returns:
            { "fix_type": int, "satellites_visible": int, "hdop": float }

        Voice triggers: "check GPS", "GPS status", "satellite count"
        """
        msg = self._get_cached("GPS_RAW_INT")
        if msg:
            fix_names = {0: "no fix", 1: "no fix", 2: "2D", 3: "3D",
                         4: "DGPS", 5: "RTK float", 6: "RTK fixed"}
            result = {
                "fix_type":           msg.fix_type,
                "satellites_visible": msg.satellites_visible,
                "hdop":               msg.eph / 100.0,
            }
            print(
                f"{self._LOG_PREFIX} GPS: {fix_names.get(msg.fix_type, str(msg.fix_type))}, "
                f"{msg.satellites_visible} satellites, HDOP {result['hdop']:.1f}"
            )
            return result

        print(f"{self._LOG_PREFIX} GPS status unavailable.")
        return None

    def get_current_mode(self) -> Optional[str]:
        """
        Return the vehicle's current flight mode name, or None.

        Voice triggers: "what mode am I in", "current flight mode", "which mode"
        """
        msg = self._get_cached("HEARTBEAT")
        if msg:
            mode_name = mavutil.mode_string_v10(msg)
            print(f"{self._LOG_PREFIX} Current mode: {mode_name}")
            return mode_name
        print(f"{self._LOG_PREFIX} Could not read current mode.")
        return None

    def get_location(self) -> Optional[dict]:
        """
        Return current GPS position and altitude, or None.

        Returns:
            { "lat": float, "lon": float,
              "alt_relative": float, "alt_asl": float }

        Voice triggers: "where am I", "current position", "GPS coordinates"
        """
        msg = self._get_global_position()
        if msg:
            result = {
                "lat":          msg.lat / 1e7,
                "lon":          msg.lon / 1e7,
                "alt_relative": msg.relative_alt / 1000.0,
                "alt_asl":      msg.alt / 1000.0,
            }
            print(
                f"{self._LOG_PREFIX} Location: ({result['lat']:.6f}, {result['lon']:.6f}), "
                f"alt {result['alt_relative']:.1f} m AGL"
            )
            return result

        print(f"{self._LOG_PREFIX} Location unavailable.")
        return None

    # ======================================================================
    # Public — emergency
    # ======================================================================

    def trigger_emergency_safe_state(self) -> str:
        """
        Cancel all pending timers, override the altitude gate, and
        switch to the safest available mode.

        Returns "RTL_ACTIVATED" or "LAND_ACTIVATED".
        """
        print(f"[CRITICAL] {self._LOG_PREFIX} EMERGENCY SAFE STATE TRIGGERED!")

        if self._hover_timer is not None:
            self._hover_timer.cancel()
        with self._state_lock:
            self._state = self._STATE_EMERGENCY

        self._set_mode("RTL")
        time.sleep(1)

        alt = self._get_altitude()
        if alt is not None and alt > 3.0:
            return "RTL_ACTIVATED"

        self._set_mode("LAND")
        return "LAND_ACTIVATED"

    # ======================================================================
    # Speed control (shared by both vehicles)
    # ======================================================================

    def set_airspeed(self, speed_ms: float) -> bool:
        """
        Set the target airspeed (m/s).

        Voice triggers: "set airspeed to X", "airspeed X metres per second"
        """
        print(f"{self._LOG_PREFIX} Setting airspeed to {speed_ms:.1f} m/s...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            p1=0.0, p2=float(speed_ms), p3=-1.0,
        )

    def set_groundspeed(self, speed_ms: float) -> bool:
        """
        Set the target ground speed (m/s).

        Voice triggers: "set speed to X", "fly at X metres per second",
                        "go faster", "slow down"
        """
        print(f"{self._LOG_PREFIX} Setting groundspeed to {speed_ms:.1f} m/s...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            p1=1.0, p2=float(speed_ms), p3=-1.0,
        )

    # ======================================================================
    # GPS navigation (shared)
    # ======================================================================

    def goto_waypoint(self, lat: float, lon: float,
                      alt: Optional[float] = None) -> bool:
        """
        Fly to a GPS coordinate in GUIDED mode.
        If `alt` is None the vehicle maintains its current altitude.

        Voice triggers: "go to coordinates X Y", "navigate to waypoint",
                        "fly to latitude X longitude Y"
        """
        if not self._is_command_allowed("goto_waypoint"):
            return False
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot navigate — vehicle is not airborne.")
            return False

        if alt is None:
            alt = self._get_altitude() or 10.0
            print(f"{self._LOG_PREFIX} No altitude specified — maintaining {alt:.1f} m.")

        self._accept_command()
        print(f"{self._LOG_PREFIX} Navigating to ({lat:.6f}, {lon:.6f}) at {alt:.1f} m...")

        if not self._set_mode("GUIDED"):
            print(f"{self._LOG_PREFIX} Cannot switch to GUIDED — navigation aborted.")
            return False

        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_111111111000,
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )
        return True

    def return_to_launch(self) -> bool:
        """
        Switch to RTL — climbs to RTL altitude, flies home, lands.

        Voice triggers: "come home", "return to launch", "go home", "RTL"
        """
        print(f"{self._LOG_PREFIX} Returning to launch...")
        success = self._set_mode("RTL")
        if not success:
            print(f"{self._LOG_PREFIX} Failed to switch to RTL mode.")
        return success

    def land(self) -> bool:
        """
        Land at the current position.
        Blocks until altitude < 0.3 m or a 60 s timeout.

        Voice triggers: "land", "land now", "set down"
        """
        print(f"{self._LOG_PREFIX} Landing...")
        if not self._set_mode("LAND"):
            print(f"{self._LOG_PREFIX} Failed to switch to LAND mode.")
            return False

        deadline = time.time() + 60
        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                time.sleep(1)
                continue
            print(f"{self._LOG_PREFIX} Descending — altitude: {alt:.2f} m")
            if alt < 0.3:
                print(f"{self._LOG_PREFIX} Landed.")
                return True
            time.sleep(1)

        print(f"{self._LOG_PREFIX} Landing timed out.")
        return False

    def change_flight_mode(self, mode_name: str) -> bool:
        """Change to an arbitrary ArduPilot flight mode by name."""
        if not self._is_command_allowed(f"change_flight_mode({mode_name})"):
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Changing flight mode to: {mode_name}")
        success = self._set_mode(mode_name)
        if not success:
            print(f"{self._LOG_PREFIX} Failed to change mode to {mode_name}.")
        return success

    # ======================================================================
    # Connection lifecycle
    # ======================================================================

    def close_connection(self):
        """Shut down the listener thread and close the MAVLink connection."""
        print(f"{self._LOG_PREFIX} Closing vehicle connection...")
        if self._hover_timer is not None:
            self._hover_timer.cancel()
        self._listener_running = False
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2)
        if self.master:
            self.master.close()
