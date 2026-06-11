import math
import time
import threading
from typing import Optional
from pymavlink import mavutil
from pymavlink.mavutil import mavfile

# ======================================================================
# MAVLink drone driver — pymavlink only
# Tested against ArduCopter SITL on udp:127.0.0.1:14550.
# ======================================================================

MIN_GPS_FIX = 3
HEARTBEAT_TIMEOUT = 10

class MAVLinkDroneDriver:
    """
    Thin pymavlink wrapper that exposes high-level drone actions used by
    the voice-control system.
    """

    def __init__(self, connection_string="udp:127.0.0.1:14550"):
        print(f"Connecting to vehicle on: {connection_string}")
        self.master: mavfile = mavutil.mavlink_connection(connection_string)  # type: ignore

        print("Waiting for heartbeat...")
        msg = self.master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
        if msg is None:
            raise ConnectionError(
                f"No heartbeat received within {HEARTBEAT_TIMEOUT} s. "
                "Is the SITL or flight controller running?"
            )

        print(
            f"Heartbeat received — system {self.master.target_system}, "
            f"component {self.master.target_component}"
        )
        self.message_cache = {}
        self.cache_lock = threading.Lock()
        self.listener_running = True
        self.listener_thread = threading.Thread(
            target=self._message_listener, 
            daemon=True)
        self.listener_thread.start()

    def _message_listener(self):
        """
        Background thread to continuously read messages and update cache.
        """
        while self.listener_running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue
                
                with self.cache_lock:
                     self.message_cache[msg.get_type()] = msg
            except OSError as e:
                break
            except Exception as e:
                print(f"Cache Error: {e}")
    
    def _get_cached_message(self, msg_type: str):
        """Get the latest message of a given type from the cache."""
        with self.cache_lock:
            return self.message_cache.get(msg_type)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _send_command_long(self, command, p1=0.0, p2=0.0, p3=0.0,
                           p4=0.0, p5=0.0, p6=0.0, p7=0.0):
        """Send a MAV_CMD via COMMAND_LONG and wait for ACK."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            command,
            0,
            p1, p2, p3, p4, p5, p6, p7,
        )
        return self._wait_command_ack(command)

    def _wait_command_ack(self, command, timeout=10):
        """
        Block until COMMAND_ACK for the given command ID arrives.
         Returns True if ACK received with result ACCEPTED, False on
         rejection or timeout.
        """
        with self.cache_lock:
            self.message_cache.pop("COMMAND_ACK", None)

        deadline = time.time() + timeout

        while time.time() < deadline:
            #msg = self.master.recv_match(type="COMMAND_ACK", blocking=True, timeout=1)
            msg = self._get_cached_message("COMMAND_ACK")
            if msg and msg.command == command:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    return True
                print(f"[DRIVER] Command {command} rejected — result code {msg.result}")
                return False
            time.sleep(0.05)

        print(f"[DRIVER] Timeout waiting for ACK of command {command}")
        return False

    def _set_mode(self, mode_name: str) -> bool:
        """
        Change the flight mode by name (e.g. 'GUIDED', 'LAND', 'RTL').
        """
        mode_mapping = self.master.mode_mapping()
        if not mode_mapping:
            raise RuntimeError(
                "Failed to retrieve mode mapping from vehicle.")
        if mode_name not in mode_mapping:
            raise ValueError(
                f"Mode '{mode_name}' is not available on this vehicle.")

        mode_id = mode_mapping[mode_name]

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

        deadline = time.time() + 5
        while time.time() < deadline:
            msg = self._get_cached_message("HEARTBEAT")
            if msg and msg.custom_mode == mode_id:
                return True
            time.sleep(0.1)

        print(f"[DRIVER] Mode change to {mode_name} not confirmed in time.")
        return False

    def _get_altitude(self):
        """Return current relative altitude in metres, or None if unavailable."""
        msg = self._get_cached_message("GLOBAL_POSITION_INT")
        if msg:
            return msg.relative_alt / 1000.0
        return None

    def _is_armed(self) -> bool:
        """Return True if the vehicle reports as armed."""
        msg = self._get_cached_message("HEARTBEAT")
        if msg:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        return False

    def _wait_until_armed(self, timeout=10) -> bool:
        """
        Poll HEARTBEAT until armed flag is set.
        ArduPilot needs a moment to physically transition to armed state.
        Firing takeoff without this wait causes result code 4 (FAILED).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_armed():
                return True
            time.sleep(0.3)
        return False

    def _wait_for_armable(self, timeout=30) -> bool:
        """
        Block until pre-arm checks pass.

        Also checks EKF status via EKF_STATUS_REPORT in addition
        to GPS fix. SITL frequently rejects arming due to EKF not ready
        even when GPS fix_type >= 3.
        """
        print("[DRIVER] Waiting for pre-arm checks (GPS + EKF)...")
        deadline = time.time() + timeout
        gps_ok = False
        ekf_ok = False

        while time.time() < deadline:
            # Check GPS
            gps = self._get_cached_message("GPS_RAW_INT")
            if gps and gps.fix_type >= MIN_GPS_FIX:
                gps_ok = True

            # Check EKF flags (flags bit 0 = attitude, bit 1 = vel horiz, bit 2 = pos horiz)
            ekf = self._get_cached_message("EKF_STATUS_REPORT")
            if ekf:
                # Require attitude + velocity + position estimates to be healthy
                EKF_MINIMUM_FLAGS = 0x07
                if (ekf.flags & EKF_MINIMUM_FLAGS) == EKF_MINIMUM_FLAGS:
                    ekf_ok = True

            if gps_ok and ekf_ok:
                print("[DRIVER] GPS fix + EKF ready — vehicle is armable.")
                return True

            status = f"GPS={'OK' if gps_ok else 'waiting'}, EKF={'OK' if ekf_ok else 'waiting'}"
            print(f"[DRIVER] Pre-arm: {status}")
            time.sleep(1)

        print("[DRIVER] Timed out waiting for pre-arm checks.")
        return False
    
    def _get_global_position(self):
        """
        Return the latest GLOBAL_POSITION_INT message, or None.
        Used internally by navigation and status methods.
        """
        return self._get_cached_message("GLOBAL_POSITION_INT")
    # ======================================================================
    # Public API
    # ======================================================================

    def arm_vehicle(self) -> bool:
        """
        Switch to GUIDED mode, wait for pre-arm checks, then arm motors.
        Added _wait_until_armed() after ACK so callers (especially
        execute_takeoff) don't fire commands before the drone is truly armed.
        """
        if self._is_armed():
            print("[DRIVER] Vehicle is already armed.")
            return False

        if not self._set_mode("GUIDED"):
            print("[DRIVER] Could not enter GUIDED mode — aborting arm.")
            return False

        if not self._wait_for_armable():
            print("[DRIVER] Cannot arm — pre-arm checks failed.")
            return False

        print("[DRIVER] Sending arm command...")
        result = self._send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            p1=1,
        )

        if result:
            # Wait for the armed state to actually propagate.
            if self._wait_until_armed(timeout=8):
                print("[DRIVER] Armed.")
                return True

        print("[DRIVER] Arming failed.")
        return False

    def disarm_vehicle(self) -> bool:
        """Disarm the motors."""
        if not self._is_armed():
            print("[DRIVER] Vehicle is already disarmed.")
            return False

        print("[DRIVER] Disarming...")
        result = self._send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            p1=0,
        )

        if result:
            deadline = time.time() + 5
            while time.time() < deadline:
                if not self._is_armed():
                    print("[DRIVER] Disarmed.")
                    return True
                time.sleep(0.5)

        print("[DRIVER] Disarm failed.")
        return False

    def execute_takeoff(self, target_altitude=5.0) -> bool:
        """
        Command a takeoff to target_altitude metres (relative).
        Blocks until the vehicle reaches 95% of the target altitude.
        """
        if not self._is_armed():
            print("[DRIVER] Vehicle not armed — arming first...")
            if not self.arm_vehicle():
                print("[DRIVER] Takeoff aborted — could not arm.")
                return False

        print(f"[DRIVER] Taking off to {target_altitude:.1f} m...")
        success = self._send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            p7=target_altitude,
        )

        if not success:
            print("[DRIVER] Takeoff command rejected.")
            return False

        # Timeout + stuck-altitude detection.
        TAKEOFF_TIMEOUT = 30       # seconds total
        STUCK_THRESHOLD = 0.05     # metres — if altitude barely changes
        STUCK_WINDOW = 8           # seconds to watch for stuck condition

        deadline = time.time() + TAKEOFF_TIMEOUT
        stuck_start = time.time()
        last_alt = 0.0

        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                print("[DRIVER] Waiting for altitude data...")
                time.sleep(1)
                continue

            print(f"[DRIVER] Altitude: {alt:.2f} m")

            if alt >= target_altitude * 0.95:
                print("[DRIVER] Target altitude reached.")
                return True

            # Detect stuck: if altitude hasn't moved in STUCK_WINDOW seconds
            if abs(alt - last_alt) > STUCK_THRESHOLD:
                last_alt = alt
                stuck_start = time.time()
            elif time.time() - stuck_start > STUCK_WINDOW:
                print(f"[DRIVER] Takeoff stuck at {alt:.2f} m — aborting.")
                return False

            time.sleep(1)

        print(f"[DRIVER] Takeoff timed out after {TAKEOFF_TIMEOUT} s.")
        return False
    
    def land(self) -> bool:
        """
        Command the vehicle to land at its current position.

        Switches to LAND mode and blocks until altitude drops below
        0.3 m (ground contact), or times out after 60 s.

        Voice triggers: 
        "land", "land now", "set down", "touch down"
        """
        print("[DRIVER] Landing...")
        if not self._set_mode("LAND"):
            print("[DRIVER] Failed to switch to LAND mode.")
            return False

        deadline = time.time() + 60
        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                time.sleep(1)
                continue
            print(f"[DRIVER] Descending — altitude: {alt:.2f} m")
            if alt < 0.3:
                print("[DRIVER] Landed.")
                return True
            time.sleep(1)

        print("[DRIVER] Landing timed out.")
        return False

    def return_to_launch(self) -> bool:
        """
        Switch to RTL (Return To Launch) mode.

        The vehicle will climb to RTL altitude, fly back to the arming
        location, and land automatically.

        Voice triggers: 
        "come home", "return to launch", "go home", "RTL"
        """
        print("[DRIVER] Returning to launch...")
        success = self._set_mode("RTL")
        if not success:
            print("[DRIVER] Failed to switch to RTL mode.")
        return success
    
# ======================================================================
# Public API — In-flight body-frame movement
# ======================================================================

    def send_body_translation(self, vx: float, vy: float, vz: float):
        """
        Send a SET_POSITION_TARGET_LOCAL_NED velocity command in the
        body/NED frame (m/s).

        Args:
            vx: Forward velocity (m/s, positive = forward).
            vy: Right velocity  (m/s, positive = right).
            vz: Down velocity   (m/s, positive = down — MAVLink convention).
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
        # Caller is responsible for the send loop and timing.
    
    def _move_direction(self, vx: float, vy: float, vz: float,
                        speed: float, duration: float):
        """
        Internal: send a velocity command for `duration` seconds at `speed`
        m/s, then command a hover stop.  Republishes every 0.2 s so
        ArduPilot doesn't time out the velocity command.

        Args:
            vx, vy, vz: Direction unit vector (body frame).
            speed:      Desired speed in m/s.
            duration:   How long to move (seconds).
        """
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
        if not self.is_flying():
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        print(f"[DRIVER] Moving forward at {speed} m/s for {duration} s...")
        self._move_direction(1.0, 0.0, 0.0, speed, duration)
        return True

    def move_backward(self, speed: float = 2.0, duration: float = 2.0) -> bool:
        """
        Fly backward at `speed` m/s for `duration` seconds, then hover.

        Voice triggers: "go back", "fly backward", "move backward X seconds"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        print(f"[DRIVER] Moving backward at {speed} m/s for {duration} s...")
        self._move_direction(-1.0, 0.0, 0.0, speed, duration)
        return True

    def move_left(self, speed: float = 2.0, duration: float = 2.0) -> bool:
        """
        Strafe left at `speed` m/s for `duration` seconds, then hover.

        Voice triggers: "go left", "move left", "fly left X seconds"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        print(f"[DRIVER] Moving left at {speed} m/s for {duration} s...")
        self._move_direction(0.0, -1.0, 0.0, speed, duration)
        return True

    def move_right(self, speed: float = 2.0, duration: float = 2.0) -> bool:
        """
        Strafe right at `speed` m/s for `duration` seconds, then hover.

        Voice triggers: "go right", "move right", "fly right X seconds"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        print(f"[DRIVER] Moving right at {speed} m/s for {duration} s...")
        self._move_direction(0.0, 1.0, 0.0, speed, duration)
        return True
    
    def ascend(self, metres: float = 5.0, speed: float = 1.5) -> bool:
        """
        Climb by `metres` relative to current altitude at `speed` m/s.

        Reads current altitude, computes a target, then sends upward
        velocity until target is reached or a timeout fires.

        Voice triggers: "go up", "ascend", "climb X metres",
                        "go up X metres", "increase altitude"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot ascend — drone is not airborne.")
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            print("[DRIVER] Cannot read current altitude.")
            return False

        target_alt = current_alt + metres
        print(f"[DRIVER] Ascending {metres} m to {target_alt:.1f} m...")

        timeout = metres / speed + 10        # generous deadline
        deadline = time.time() + timeout

        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                time.sleep(0.3)
                continue
            if alt >= target_alt - 0.3:
                self.hover()
                print(f"[DRIVER] Ascent complete — altitude {alt:.2f} m.")
                return True
            # vz is negative = upward in MAVLink NED convention
            self.send_body_translation(0.0, 0.0, -speed)
            time.sleep(0.2)

        self.hover()
        print("[DRIVER] Ascent timed out.")
        return False

    def descend(self, metres: float = 5.0, speed: float = 1.0) -> bool:
        """
        Descend by `metres` relative to current altitude at `speed` m/s.

        Stops at 1 m minimum to avoid landing unintentionally.

        Voice triggers: "go down", "descend", "drop X metres",
                        "go down X metres", "decrease altitude", "lower"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot descend — drone is not airborne.")
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            print("[DRIVER] Cannot read current altitude.")
            return False

        MIN_SAFE_ALT = 1.0
        target_alt = max(current_alt - metres, MIN_SAFE_ALT)
        actual_descent = current_alt - target_alt
        if actual_descent <= 0:
            print(f"[DRIVER] Already at or below {MIN_SAFE_ALT} m — not descending.")
            return False

        print(f"[DRIVER] Descending {actual_descent:.1f} m to {target_alt:.1f} m...")

        timeout = actual_descent / speed + 10
        deadline = time.time() + timeout

        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                time.sleep(0.3)
                continue
            if alt <= target_alt + 0.3:
                self.hover()
                print(f"[DRIVER] Descent complete — altitude {alt:.2f} m.")
                return True
            # vz positive = downward in MAVLink NED convention
            self.send_body_translation(0.0, 0.0, speed)
            time.sleep(0.2)

        self.hover()
        print("[DRIVER] Descent timed out.")
        return False

    def hover(self, duration: float = 0.0) -> bool:
        """
        Immediately stop all translational motion and hold position.

        Sends a zero-velocity command. Must be in GUIDED mode.

        Voice triggers: "stop", "hover", "stay", "hold", "freeze",
                        "stop moving"
        """
        print("[DRIVER] Hovering — zeroing velocity.")
        self.send_body_translation(0.0, 0.0, 0.0)

        if duration > 0:
            deadline = time.time() + duration
            while time.time() < deadline:
                self.send_body_translation(0.0, 0.0, 0.0)
                time.sleep(0.2)
        return True
    
    # ======================================================================
    # Public API — Yaw / Heading
    # ======================================================================

    def rotate_left(self, degrees: float = 90.0) -> bool:
        """
        Yaw counter-clockwise by `degrees`.

        Uses MAV_CMD_CONDITION_YAW with relative flag set.
        Blocks until the command is ACK'd (does not wait for yaw
        completion — yaw happens asynchronously in ArduPilot).

        Voice triggers: "turn left", "rotate left", "yaw left X degrees",
                        "spin left"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot rotate — drone is not airborne.")
            return False
        print(f"[DRIVER] Rotating left {degrees}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(degrees),   # yaw angle
            p2=20.0,             # yaw speed deg/s
            p3=-1.0,             # -1 = CCW (left)
            p4=1.0,              # 1 = relative
        )

    def rotate_right(self, degrees: float = 90.0) -> bool:
        """
        Yaw clockwise by `degrees`.

        Voice triggers: "turn right", "rotate right", "yaw right X degrees",
                        "spin right"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot rotate — drone is not airborne.")
            return False
        print(f"[DRIVER] Rotating right {degrees}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(degrees),
            p2=20.0,
            p3=1.0,              # 1 = CW (right)
            p4=1.0,              # relative
        )

    def set_heading(self, heading_degrees: float) -> bool:
        """
        Point the drone to an absolute compass bearing (0–360°).

        0° = North, 90° = East, 180° = South, 270° = West.

        Voice triggers: "face north", "face south", "face east", "face west",
                        "turn to X degrees", "heading X", "face 270"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot set heading — drone is not airborne.")
            return False
        heading_degrees = heading_degrees % 360.0
        print(f"[DRIVER] Setting heading to {heading_degrees:.0f}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(heading_degrees),
            p2=20.0,             # yaw speed deg/s
            p3=1.0,              # CW (ArduPilot chooses shortest path when absolute)
            p4=0.0,              # 0 = absolute
        )
    
    # ======================================================================
    # Public API — GPS navigation
    # ======================================================================

    def goto_waypoint(self, lat: float, lon: float,
                      alt: Optional[float] = None) -> bool:
        """
        Fly to a GPS coordinate in GUIDED mode.

        If `alt` is None the drone maintains its current altitude.

        Uses MAV_CMD_NAV_WAYPOINT.  Does not block until arrival — the
        caller or NLP layer should poll is_at_location() or get_location()
        if it needs to wait.

        Args:
            lat: Target latitude  (decimal degrees, e.g. 16.8661)
            lon: Target longitude (decimal degrees, e.g. 96.1951)
            alt: Target altitude  (metres, relative to home). Defaults to
                 current altitude.

        Voice triggers: "go to coordinates X Y", "fly to latitude X
                        longitude Y", "navigate to waypoint"
        """
        if not self.is_flying():
            print("[DRIVER] Cannot navigate — drone is not airborne.")
            return False

        if alt is None:
            alt = self._get_altitude() or 10.0
            print(f"[DRIVER] No altitude specified — maintaining {alt:.1f} m.")

        print(f"[DRIVER] Navigating to ({lat:.6f}, {lon:.6f}) at {alt:.1f} m...")

        # Ensure GUIDED mode so the vehicle actually follows the waypoint.
        if not self._set_mode("GUIDED"):
            print("[DRIVER] Cannot switch to GUIDED — navigation aborted.")
            return False

        # SET_POSITION_TARGET_GLOBAL_INT (absolute lat/lon, relative alt)
        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            # type_mask: use position only, ignore velocity/acc/yaw
            0b0000_111111111000,
            int(lat * 1e7),   # lat_int (degrees × 1e7)
            int(lon * 1e7),   # lon_int
            alt,              # alt (metres)
            0, 0, 0,          # vx, vy, vz
            0, 0, 0,          # afx, afy, afz
            0, 0,             # yaw, yaw_rate
        )
        return True

    # ======================================================================
    # Public API — Speed control
    # ======================================================================

    def set_airspeed(self, speed_ms: float) -> bool:
        """
        Set the target airspeed (m/s).

        Uses MAV_CMD_DO_CHANGE_SPEED with speed_type=0 (airspeed).

        Voice triggers: "set airspeed to X", "airspeed X metres per second"
        """
        print(f"[DRIVER] Setting airspeed to {speed_ms:.1f} m/s...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            p1=0.0,             # 0 = airspeed
            p2=float(speed_ms),
            p3=-1.0,            # -1 = no throttle change
        )

    def set_groundspeed(self, speed_ms: float) -> bool:
        """
        Set the target ground speed (m/s).

        Uses MAV_CMD_DO_CHANGE_SPEED with speed_type=1 (ground speed).

        Voice triggers: "set speed to X", "fly at X metres per second",
                        "go faster", "slow down" (NLP maps to numeric value)
        """
        print(f"[DRIVER] Setting groundspeed to {speed_ms:.1f} m/s...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            p1=1.0,             # 1 = ground speed
            p2=float(speed_ms),
            p3=-1.0,
        )

    # ======================================================================
    # Public API — Flight-mode shortcuts
    # ======================================================================

    def change_flight_mode(self, mode_name: str) -> bool:
        """Change to an arbitrary ArduPilot flight mode by name."""
        print(f"[DRIVER] Changing flight mode to: {mode_name}")
        success = self._set_mode(mode_name)
        if not success:
            print(f"[DRIVER] Failed to change mode to {mode_name}.")
        return success

    def set_loiter(self) -> bool:
        """
        Switch to LOITER mode — GPS-assisted position hold with yaw control.

        In LOITER the pilot (or NLP) can still command small movements;
        the autopilot fights wind drift automatically.

        Voice triggers: "loiter", "loiter mode", "circle", "hover in place"
        """
        print("[DRIVER] Switching to LOITER mode...")
        return self._set_mode("LOITER")

    def set_position_hold(self) -> bool:
        """
        Switch to POSHOLD mode — full position, velocity, and altitude hold.

        Stricter than LOITER: no stick input needed, autopilot maintains
        exact GPS coordinate.

        Voice triggers: "hold position", "position hold", "stay here",
                        "maintain position"
        """
        print("[DRIVER] Switching to POSHOLD mode...")
        return self._set_mode("POSHOLD")

    # ======================================================================
    # Public API — Status queries (voice feedback)
    # ======================================================================

    def get_altitude(self) -> float | None:
        """
        Return current relative altitude in metres, or None.

        Voice triggers: "what is my altitude", "how high am I",
                        "altitude check", "current altitude"
        """
        alt = self._get_altitude()
        if alt is not None:
            print(f"[DRIVER] Current altitude: {alt:.2f} m")
        return alt

    def get_heading(self) -> float | None:
        """
        Return current compass heading in degrees (0–360), or None.

        Voice triggers: "what direction am I facing", "what is my heading",
                        "compass heading", "which way am I pointing"
        """
        msg = self._get_global_position()
        if msg:
            heading = msg.hdg / 100.0   # cdeg → deg
            print(f"[DRIVER] Current heading: {heading:.1f}°")
            return heading
        return None

    def get_groundspeed(self) -> float | None:
        """
        Return current ground speed in m/s, or None.

        Voice triggers: "how fast am I going", "current speed",
                        "what is my speed", "ground speed"
        """
        msg = self._get_global_position()
        if msg:
            vx = msg.vx / 100.0    # cm/s → m/s
            vy = msg.vy / 100.0
            speed = math.sqrt(vx ** 2 + vy ** 2)
            print(f"[DRIVER] Current ground speed: {speed:.2f} m/s")
            return speed
        return None

    def get_battery(self) -> dict | None:
        """
        Return battery status as a dict, or None if unavailable.

        Returns:
            {
                "voltage_v":    float,   # volts
                "current_a":    float,   # amps  (-1 if unknown)
                "remaining_pct": int,    # 0 - 100 (-1 if unknown)
            }

        Voice triggers: "check battery", "battery level",
                        "how much battery do I have", "battery status"
        """
        msg = self._get_cached_message("BATTERY_STATUS")
        if msg:
            voltage = msg.voltages[0] / 1000.0 if msg.voltages[0] != 65535 else None
            current = msg.current_battery / 100.0 if msg.current_battery != -1 else -1
            remaining = msg.battery_remaining  # percent, -1 if unknown
            result = {
                "voltage_v": voltage,
                "current_a": current,
                "remaining_pct": remaining,
            }
            print(f"[DRIVER] Battery: {voltage:.2f} V, {remaining}% remaining")
            return result

        # Fallback: try SYS_STATUS which also carries battery info
        msg = self._get_cached_message("SYS_STATUS")
        if msg:
            voltage = msg.voltage_battery / 1000.0
            current = msg.current_battery / 100.0 if msg.current_battery != -1 else -1
            remaining = msg.battery_remaining
            result = {
                "voltage_v": voltage,
                "current_a": current,
                "remaining_pct": remaining,
            }
            print(f"[DRIVER] Battery: {voltage:.2f} V, {remaining}% remaining")
            return result

        print("[DRIVER] Battery status unavailable.")
        return None

    def get_gps_status(self) -> dict | None:
        """
        Return GPS fix status, or None.

        Returns:
            {
                "fix_type":         int,   # 0=no fix … 6=RTK fixed
                "satellites_visible": int,
                "hdop":             float, # horizontal dilution of precision
            }

        Voice triggers: "check GPS", "GPS status", "satellite count",
                        "how is my GPS signal"
        """
        msg = self._get_cached_message("GPS_RAW_INT")
        if msg:
            result = {
                "fix_type": msg.fix_type,
                "satellites_visible": msg.satellites_visible,
                "hdop": msg.eph / 100.0,    # cm → m (HDOP proxy)
            }
            fix_names = {0: "no fix", 1: "no fix", 2: "2D", 3: "3D",
                         4: "DGPS", 5: "RTK float", 6: "RTK fixed"}
            fix_str = fix_names.get(msg.fix_type, str(msg.fix_type))
            print(
                f"[DRIVER] GPS: {fix_str}, "
                f"{msg.satellites_visible} satellites, HDOP {result['hdop']:.1f}"
            )
            return result

        print("[DRIVER] GPS status unavailable.")
        return None

    def get_current_mode(self) -> str | None:
        """
        Return the vehicle's current flight mode name, or None.

        Voice triggers: "what mode am I in", "current flight mode",
                        "which mode", "what mode is the drone in"
        """
        #msg = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        msg = self._get_cached_message("HEARTBEAT")
        if msg:
            mode_name = mavutil.mode_string_v10(msg)
            print(f"[DRIVER] Current mode: {mode_name}")
            return mode_name
        print("[DRIVER] Could not read current mode.")
        return None

    def get_location(self) -> dict | None:
        """
        Return current GPS position and altitude, or None.

        Returns:
            {
                "lat":          float,   # decimal degrees
                "lon":          float,   # decimal degrees
                "alt_relative": float,   # metres above home
                "alt_asl":      float,   # metres above sea level
            }

        Voice triggers: "where am I", "current position",
                        "what are my coordinates", "GPS coordinates"
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
                f"[DRIVER] Location: ({result['lat']:.6f}, {result['lon']:.6f}), "
                f"alt {result['alt_relative']:.1f} m AGL"
            )
            return result

        print("[DRIVER] Location unavailable.")
        return None

    # ======================================================================
    # Public API — State checks
    # ======================================================================

    def is_flying(self) -> bool:
        """Return True if the vehicle is airborne (altitude > 0.3 m)."""
        alt = self._get_altitude()
        if alt is None:
            return False
        return alt > 0.3

    # ======================================================================
    # Public API — Connection
    # ======================================================================

    def close_connection(self):
        """Close the MAVLink connection."""
        print("[DRIVER] Closing vehicle connection...")
        
        self.listener_running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2)
        if self.master:
            self.master.close()
