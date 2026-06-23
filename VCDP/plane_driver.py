"""
plane_driver.py — ArduPlane / fixed-wing driver.

Extends MAVLinkBaseDriver with:
  - TAKEOFF / CRUISE / FBWA mode management
  - Airspeed and throttle control
  - Banking turn commands (left / right)
  - Climb / descend via pitch attitude
  - Loiter (circle hold) and AUTO mission modes
  - Waypoint save / go-to / export
  - Dynamic Mission Planner Waypoint Verification
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

    def __init__(self, connection_string: str = "udp:127.0.0.1:14551"):
        super().__init__(connection_string)

        # Planes don't use the copter altitude-gate but share state machine
        self._target_altitude: Optional[float] = None

        # Waypoint store (saved locally via script)
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
    # Arming / takeoff / disarming
    # ======================================================================

    def arm_vehicle(self) -> bool:
        """
        Arm the plane safely. Stays in ground mode (FBWA) to prevent accidental
        throttle blade spinning.
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

        # Start in FBWA ground state for safety
        print(f"{self._LOG_PREFIX} Setting FBWA mode for ground staging...")
        if not self._set_mode("FBWA"):
            print(f"{self._LOG_PREFIX} Could not enter safety mode — aborting arm.")
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
            print(f"{self._LOG_PREFIX} Armed successfully. Standing by on ground.")
            return True

        print(f"{self._LOG_PREFIX} Arming failed — check Mission Planner for PreArm errors.")
        return False

    def disarm_vehicle(self) -> bool:
        
        # 1. Check current relative altitude
        current_alt = self.get_altitude() # Returns relative altitude in meters
        
        if current_alt is None:
            print(f"{self._LOG_PREFIX} Disarm rejected: Altitude data unavailable.")
            return False
            
        # Allow a tiny margin of error (e.g., less than 1.0 meter) for sensor noise on the ground
        if current_alt > 1.0:
            print(f"{self._LOG_PREFIX} DISARM DENIED: Aircraft is at {current_alt:.2f}m. Must be near 0m to disarm.")
            return False

        if not self._is_armed():
            print(f"{self._LOG_PREFIX} Vehicle is already disarmed.")
            return True

        # 2. Force the plane into MANUAL mode on the ground to clear flight state gates
        print(f"{self._LOG_PREFIX} Altitude verified at {current_alt:.2f}m. Forcing MANUAL mode to disarm...")
        
        # Maps to ArduPlane specific mode switches via internal master driver
        self.set_flight_mode("MANUAL")
        time.sleep(0.5) 

        # 3. Send the MAVLink disarm command
        print(f"{self._LOG_PREFIX} Sending disarm command...")
        try:
            self._master.mav.command_long_send(
                self._target_system,
                self._target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,        # Confirmation
                0,        # 0 = Disarm
                21196,    # Force disarm override code if supported by firmware
                0, 0, 0, 0, 0  # Unused parameters
            )
            command_sent = True
        except Exception as e:
            print(f"{self._LOG_PREFIX} Failed to send MAVLink packet: {e}")
            command_sent = False

        # 4. Await verification from state tracking thread loop
        if command_sent and self._wait_until_disarmed(timeout=5.0):
            print(f"{self._LOG_PREFIX} Disarmed successfully.")
            with self._state_lock:
                self._state = self._STATE_GROUND
            return True

        print(f"{self._LOG_PREFIX} Disarming failed or timeout waiting for state transition.")
        return False

    def _wait_until_disarmed(self, timeout: float = 5.0) -> bool:
        """
        Blocks until the vehicle reports a disarmed state from heartbeat, or times out.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._is_armed():
                return True
            time.sleep(0.1)
        return False

    def execute_takeoff(self, target_altitude: float = 30.0) -> bool:
        """
        Command a fixed-wing hand-launch or runway takeoff to target_altitude.
        """
        if not self._is_armed():
            print(f"{self._LOG_PREFIX} CRITICAL: Takeoff rejected — arm first.")
            return False

        # Switch to active TAKEOFF mode to enable launch throttle suppression logic
        self._set_mode("TAKEOFF")
        time.sleep(0.2)

        print(f"{self._LOG_PREFIX} Commanding fixed-wing takeoff to {target_altitude:.0f} m...")

        success = self._send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            p1=15.0,           # launch pitch angle
            p7=target_altitude
        )
        if not success:
            print(f"{self._LOG_PREFIX} Takeoff command rejected by FC.")
            return False

        # Monitor altitude climb
        TAKEOFF_TIMEOUT = 60
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
                print(f"{self._LOG_PREFIX} Cruise altitude reached. Switching to CRUISE mode...")
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

        print(f"{self._LOG_PREFIX} Takeoff timed out.")
        return False
    

    # ======================================================================
    # NEW: Waypoint Validation Logic (Mission Planner Connectivity)
    # ======================================================================

    # ======================================================================
    # Mission Planner Waypoint Verification
    # ======================================================================

    def has_mission_uploaded(self) -> bool:

        print(f"{self._LOG_PREFIX} Querying flight controller for active missions...")
        
        # Clear old cached MISSION_COUNT messages to ensure fresh data
        with self._cache_lock:
            self._message_cache.pop("MISSION_COUNT", None)
            
        # Send MAVLink request for the mission list count
        self.master.mav.mission_request_list_send(
            self.master.target_system, 
            self.master.target_component
        )
        
        # Wait up to 3 seconds for the flight controller to respond
        deadline = time.time() + 3.0
        while time.time() < deadline:
            msg = self._get_cached("MISSION_COUNT")
            if msg:
                count = msg.count
                # ArduPilot always loads WP 0 as Home. A valid mission requires at least WP 1.
                if count > 1:
                    print(f"{self._LOG_PREFIX}Mission verified! {count - 1} flight waypoints found in autopilot memory.")
                    return True
                else:
                    print(f"{self._LOG_PREFIX}Mission verification failed: Only Home position (WP 0) is loaded.")
                    return False
            time.sleep(0.2)
            
        print(f"{self._LOG_PREFIX}  Mission check timed out. No response from flight controller.")
        return False

    # ======================================================================
    # Flight mode shortcuts
    # ======================================================================

    def set_cruise(self) -> bool:
        print(f"{self._LOG_PREFIX} Switching to CRUISE mode...")
        return self._set_mode("CRUISE")

    def set_fbwa(self) -> bool:
        print(f"{self._LOG_PREFIX} Switching to FBWA mode...")
        return self._set_mode("FBWA")

    def set_auto(self) -> bool:
        print(f"{self._LOG_PREFIX} Switching to AUTO mode...")
        success = self._set_mode("AUTO")
        if success:
            with self._state_lock:
                self._state = self._STATE_FLYING
        return success

    def set_loiter(self) -> bool:
        if not self._is_command_allowed("set_loiter"):
            return False
        self._accept_command()
        print(f"{self._LOG_PREFIX} Switching to LOITER mode...")
        return self._set_mode("LOITER")

    def set_guided(self) -> bool:
        print(f"{self._LOG_PREFIX} Switching to GUIDED mode...")
        return self._set_mode("GUIDED")

    # ======================================================================
    # Attitude / climb / descend commands
    # ======================================================================

    def climb_to_altitude(self, target_altitude: float, speed: float = None) -> bool:
        if not self.is_flying():
            print(f"{self._LOG_PREFIX} Cannot climb — plane is not airborne.")
            return False

        if speed is None:
            speed = self.DEFAULT_CLIMB_SPEED

        current_alt = self._get_altitude()
        if current_alt is None:
            return False

        if target_altitude <= current_alt:
            return False

        if not self._set_mode("GUIDED"):
            return False

        self.set_airspeed(speed)
        loc = self.get_location()
        if loc is None:
            return False

        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_111111111000,
            int(loc["lat"] * 1e7), int(loc["lon"] * 1e7), target_altitude,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

        deadline = time.time() + 120
        while time.time() < deadline:
            alt = self._get_altitude()
            if alt and alt >= target_altitude - self.ALTITUDE_TOLERANCE:
                print(f"{self._LOG_PREFIX} Target altitude reached.")
                return True
            time.sleep(1)
        return False

    def descend_to_altitude(self, target_altitude: float) -> bool:
        if not self.is_flying():
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            return False

        MIN_SAFE_ALT = 50.0
        target_altitude = max(target_altitude, MIN_SAFE_ALT)

        if not self._set_mode("GUIDED"):
            return False

        loc = self.get_location()
        if loc is None:
            return False

        self.master.mav.set_position_target_global_int_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_111111111000,
            int(loc["lat"] * 1e7), int(loc["lon"] * 1e7), target_altitude,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

        deadline = time.time() + 120
        while time.time() < deadline:
            alt = self._get_altitude()
            if alt and alt <= target_altitude + self.ALTITUDE_TOLERANCE:
                return True
            time.sleep(1)
        return False

    # ======================================================================
    # Turning commands
    # ======================================================================

    def turn_left(self, heading_degrees: float = None, bank_angle: float = 30.0) -> bool:
        if not self.is_flying():
            return False
        if heading_degrees is not None:
            return self._send_command_long(
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                p1=float(heading_degrees), p2=0.0, p3=-1.0, p4=0.0,
            )
        self._set_mode("CRUISE")
        return True

    def turn_right(self, heading_degrees: float = None, bank_angle: float = 30.0) -> bool:
        if not self.is_flying():
            return False
        if heading_degrees is not None:
            return self._send_command_long(
                mavutil.mavlink.MAV_CMD_CONDITION_YAW,
                p1=float(heading_degrees), p2=0.0, p3=1.0, p4=0.0,
            )
        self._set_mode("CRUISE")
        return True

    def set_heading(self, heading_degrees: float) -> bool:
        if not self.is_flying():
            return False
        heading_degrees %= 360.0
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(heading_degrees), p2=0.0, p3=1.0, p4=0.0,
        )
    def set_manual(self) -> bool:
        
        print(f"{self._LOG_PREFIX} Requesting mode switch to MANUAL...")
        return self._set_mode("MANUAL")

    # ======================================================================
    # Waypoint management (Local generation storage)
    # ======================================================================

    def save_waypoint(self) -> bool:
        loc = self.get_location()
        if loc is None:
            return False

        wp = {
            "index": len(self._waypoints),
            "lat":   loc["lat"],
            "lon":   loc["lon"],
            "alt":   loc["alt_relative"],
        }
        self._waypoints.append(wp)
        return True

    def goto_last_waypoint(self) -> bool:
        if not self._waypoints:
            return False
        wp = self._waypoints[-1]
        return self.goto_waypoint(wp["lat"], wp["lon"], wp["alt"])

    def export_mission(self) -> bool:
        if not self._waypoints:
            return False

        filename = "plane_mission.waypoints"
        with open(filename, "w") as f:
            f.write("QGC WPL 110\n")
            for i, wp in enumerate(self._waypoints):
                f.write(
                    f"{i}\t0\t3\t16\t0\t0\t0\t0\t"
                    f"{wp['lat']:.7f}\t{wp['lon']:.7f}\t{wp['alt']:.2f}\t1\n"
                )
        return True

    # ======================================================================
    # Emergency
    # ======================================================================

    def trigger_emergency_safe_state(self) -> str:
        if self._hover_timer is not None:
            self._hover_timer.cancel()
        with self._state_lock:
            self._state = self._STATE_EMERGENCY

        self._set_mode("RTL")
        return "RTL_ACTIVATED"

    def close_connection(self):
        self._monitor_active = False
        super().close_connection()