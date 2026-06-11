import math
import time
import threading
from typing import Optional
from pymavlink import mavutil
from pymavlink.mavutil import mavfile


MIN_GPS_FIX        = 3
HEARTBEAT_TIMEOUT  = 10


class MAVLinkDroneDriver:
    

    # ── Altitude-gate tuning ──────────────────────────────────────────
    ALTITUDE_TOLERANCE = 0.5    # metres — "close enough" to target
    HOVER_TIMEOUT      = 10.0   # seconds before auto-LOITER after reaching alt

    # ── Battery safety thresholds ─────────────────────────────────────
    BATTERY_MIN_PCT     = 20    # % — block arming below this level
    BATTERY_WARN_PCT    = 30    # % — warn but still allow arming
    BATTERY_MIN_VOLTAGE = 10.5  # V  — absolute floor (3S LiPo = 3.5 V/cell)

    # ── Internal state labels ─────────────────────────────────────────
    _STATE_IDLE      = "IDLE"
    _STATE_LOCKED    = "LOCKED"
    _STATE_READY     = "READY"
    _STATE_FLYING    = "FLYING"
    _STATE_HOVER     = "HOVER"
    _STATE_EMERGENCY = "EMERGENCY"

    # ======================================================================
    # Construction / connection
    # ======================================================================

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        print(f"[DRIVER] Connecting to vehicle on: {connection_string}")
        self.master: mavfile = mavutil.mavlink_connection(connection_string)  # type: ignore

        print("[DRIVER] Waiting for heartbeat...")
        msg = self.master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT)
        if msg is None:
            raise ConnectionError(
                f"No heartbeat received within {HEARTBEAT_TIMEOUT} s. "
                "Is the SITL or flight controller running?"
            )
        print(
            f"[DRIVER] Heartbeat received — system {self.master.target_system}, "
            f"component {self.master.target_component}"
        )

        # ── Message cache ─────────────────────────────────────────────
        self._message_cache: dict = {}
        self._cache_lock = threading.Lock()
        self._listener_running = True
        self._listener_thread = threading.Thread(
            target=self._message_listener, daemon=True
        )
        self._listener_thread.start()

        # ── Safety state machine ──────────────────────────────────────
        self._state           = self._STATE_IDLE
        self._state_lock      = threading.Lock()
        self._target_altitude: Optional[float] = None
        self._altitude_monitor: Optional[threading.Thread] = None
        self._hover_timer: Optional[threading.Timer] = None
        self._hover_heartbeat_active = True
        self._hover_thread = threading.Thread(target=self._production_hover_watchdog, daemon=True)
        self._hover_thread.start()

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
                print(f"[DRIVER] Cache error: {exc}")

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
                print(f"[DRIVER] Command {command} rejected — result code {msg.result}")
                return False
            time.sleep(0.05)

        print(f"[DRIVER] Timeout waiting for ACK of command {command}")
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

        print(f"[DRIVER] Mode change to {mode_name} not confirmed in time.")
        return False

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
        """Poll HEARTBEAT until armed flag is set."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_armed():
                return True
            time.sleep(0.3)
        return False

    def _wait_for_armable(self, timeout: float = 30) -> bool:
        """
        Block until pre-arm checks pass.

        Checks:
          - GPS_RAW_INT  fix_type >= 3 (3-D fix)
          - EKF_STATUS_REPORT flags bits 0-2 all set
                bit 0 = attitude estimate OK
                bit 1 = horizontal velocity estimate OK
                bit 2 = horizontal position estimate OK

        Prints a detailed per-check status every second so the exact
        blocking condition is always visible in the log.
        """
        print("[DRIVER] Waiting for pre-arm checks (GPS + EKF)...")
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
                        f"[DRIVER] GPS not ready — "
                        f"fix_type={fix_names.get(gps.fix_type, gps.fix_type)}, "
                        f"sats={gps.satellites_visible}"
                    )
            else:
                print("[DRIVER] GPS — no GPS_RAW_INT message received yet.")

            # ── EKF check ─────────────────────────────────────────────
            ekf = self._get_cached("EKF_STATUS_REPORT")
            if ekf:
                attitude_ok  = bool(ekf.flags & 0x01)
                velocity_ok  = bool(ekf.flags & 0x02)
                position_ok  = bool(ekf.flags & 0x04)
                if attitude_ok and velocity_ok and position_ok:
                    ekf_ok = True
                else:
                    print(
                        f"[DRIVER] EKF not ready — "
                        f"attitude={'OK' if attitude_ok else 'WAIT'}, "
                        f"velocity={'OK' if velocity_ok else 'WAIT'}, "
                        f"position={'OK' if position_ok else 'WAIT'} "
                        f"(raw flags=0x{ekf.flags:02X})"
                    )
            else:
                print("[DRIVER] EKF — no EKF_STATUS_REPORT received yet "
                      "(normal for first ~5 s on SITL).")

            if gps_ok and ekf_ok:
                print("[DRIVER] ✅ GPS + EKF ready — vehicle is armable.")
                return True

            time.sleep(1)

        print(
            f"[DRIVER]  Pre-arm timeout after {timeout:.0f} s — "
            f"GPS={'OK' if gps_ok else 'FAILED'}, "
            f"EKF={'OK' if ekf_ok else 'FAILED'}"
        )
        return False

    # ======================================================================
    # Safety state machine — internal
    # ======================================================================

    def _is_command_allowed(self, command_name: str) -> bool:
        """
        Return True if a movement/mode command is allowed in the current
        state. Prints a human-readable rejection when blocked.
        """
        with self._state_lock:
            if self._state == self._STATE_LOCKED:
                alt = self._get_altitude()
                remaining = (
                    f"{self._target_altitude - alt:.1f} m remaining"
                    if alt is not None else "altitude unknown"
                )
                print(
                    f"[DRIVER]  '{command_name}' BLOCKED — vehicle has not "
                    f"reached target altitude ({self._target_altitude} m). "
                    f"{remaining}. Please wait."
                )
                return False
        return True

    def _start_altitude_monitor(self, target_altitude: float):
        """
        Spin a daemon thread that watches altitude and transitions
        LOCKED → READY once the vehicle arrives, then starts the
        hover timer.
        """
        self._target_altitude = target_altitude
        with self._state_lock:
            self._state = self._STATE_LOCKED

        def _monitor():
            print(
                f"[DRIVER] 🔒 Altitude gate ACTIVE — commands locked until "
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
                                f"[DRIVER] ✅ Target altitude {target_altitude} m reached "
                                f"(current: {alt:.1f} m). Commands now accepted."
                            )
                    self._start_hover_timer()
                    return

                time.sleep(0.5)

        self._altitude_monitor = threading.Thread(target=_monitor, daemon=True)
        self._altitude_monitor.start()

    def _production_hover_watchdog(self):
        """
        Production-grade background watchdog thread. 
        Completely silent battery tracking and active hover hold.
        """
        has_warned_low = False
        has_warned_anomaly = False

        while self._hover_heartbeat_active:
            with self._state_lock:
                current_state = self._state
            
            batt = self.get_battery()
            if batt is not None:
                voltage = batt.get("voltage_v")
                remaining = batt.get("remaining_pct", -1)

                if voltage is None:
                    voltage = 12.6

                # 1. TRUE CRITICAL EXTRACTION (Only prints if the drone is actually aborting)
                if (voltage < self.BATTERY_MIN_VOLTAGE) or (remaining != -1 and remaining < self.BATTERY_MIN_PCT and voltage < 11.1):
                    print(f"\n[CRITICAL] 🔋 REAL BATTERY EXHAUSTION DETECTED: {voltage:.2f}V, {remaining}%.")
                    if self.is_flying():
                        print("[CRITICAL] EMERGENCY SAFETY ACTION: Aborting flight!")
                        self.trigger_emergency_safe_state() 
                        break

                # 2. SILENT ANOMALY TRACKING (Prints exactly ONCE instead of loops)
                elif remaining == 0 and voltage >= self.BATTERY_MIN_VOLTAGE:
                    if not has_warned_anomaly:
                        print(f"\n[DRIVER]Telemetry Anomaly: Capacity reports 0% but Voltage is SAFE at {voltage:.2f}V.")
                        print("[DRIVER]Flight continuing safely. Adjust capacity ratings in Mission Planner later.")
                        has_warned_anomaly = True

                # 3. STANDARD LOW BATTERY WARNING (Prints exactly ONCE when battery drops)
                elif remaining != -1 and remaining < self.BATTERY_WARN_PCT:
                    if not has_warned_low:
                        print(f"\n[DRIVER]Low Battery Alert: {remaining}% remaining ({voltage:.2f}V).")
                        has_warned_low = True
                
                else:
                    # Reset tracking flags if a healthy battery state is restored
                    has_warned_low = False
                    has_warned_anomaly = False

            # ── HOVER HEARTBEAT ──────────────────────────────────────────
            if current_state == "HOVER":
                try:
                    # Active zero-velocity streaming inside GUIDED mode
                    self.send_body_translation(vx=0.0, vy=0.0, vz=0.0)
                except Exception as e:
                    print(f"[DRIVER ERROR] Failed streaming hover heartbeat: {e}")
            
            time.sleep(0.5)
    
    
    
    HOVER_TIMEOUT=10.0
    def _start_hover_timer(self):
        if self._hover_timer is not None:
            self._hover_timer.cancel()
        
        # We will use individual threading timers to create a safe timeline
        self._warning_timer = None
        
        print(f"[DRIVER]Hover timer initialized ({self.HOVER_TIMEOUT}s).")

        # Stage 1: Verbal Warning halfway through
        def _speak_warning():
            with self._state_lock:
                if self._state == self._STATE_READY:
                    # Accessing your voice pipeline natively or via terminal print
                    print("[DRIVER]  Warning: No command detected. Engaging auto-hover fallback in 2 seconds...")

        # Stage 2: Final Fallback Execution
        def _engage_hover_hold():
            with self._state_lock:
                if self._state == self._STATE_READY:
                    self._state = self._STATE_HOVER
            print("[DRIVER]  Safeguard active: Streaming continuous 0-velocity hold in GUIDED mode.")

        # Schedule the verbal warning to happen 3 seconds in
        if self.HOVER_TIMEOUT > 3.0:
            self._warning_timer = threading.Timer(3.0, _speak_warning)
            self._warning_timer.daemon = True
            self._warning_timer.start()

        # Schedule the actual active hover engagement
        self._hover_timer = threading.Timer(self.HOVER_TIMEOUT, _engage_hold)
        self._hover_timer.daemon = True
        self._hover_timer.start()

    def reset_hover_timeout(self):
        """Call this whenever the user starts typing or talking to keep the window open!"""
        with self._state_lock:
            if self._state == self._STATE_READY:
                print("[DRIVER]  Action detected! Resetting hover countdown window.")
                self._start_hover_timer()

    def _accept_command(self):
        """
        Called at the top of every movement method that passes the gate.
        Cancels the hover timer and transitions to FLYING.
        """
        if self._hover_timer is not None:
            self._hover_timer.cancel()
            self._hover_timer = None
        with self._state_lock:
            if self._state in (self._STATE_READY, self._STATE_HOVER):
                self._state = self._STATE_FLYING
                print("[DRIVER]  Command accepted — state → FLYING.")

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
    # Public — arming
    # ======================================================================

    def _check_battery_safe(self) -> tuple[bool, str]:
        """
        Read battery status and decide if it is safe to arm.

        Returns
        -------
        (safe: bool, reason: str)
            safe   = True  → OK to proceed with arming
            safe   = False → block arming, reason explains why

        Logic
        -----
        - No battery data at all        → WARN and allow (SITL / no sensor)
        - Voltage below BATTERY_MIN_VOLTAGE → BLOCK (hard floor)
        - Percent below BATTERY_MIN_PCT → BLOCK
        - Percent below BATTERY_WARN_PCT → ALLOW with warning printed
        """
        batt = self.get_battery()

        if batt is None:
            print("[DRIVER]   Battery data unavailable — proceeding without check.")
            return True, "NO_DATA"

        voltage   = batt.get("voltage_v")
        remaining = batt.get("remaining_pct", -1)

        # ── Hard voltage floor ────────────────────────────────────────
        if voltage is not None and voltage < self.BATTERY_MIN_VOLTAGE:
            reason = (
                f"BATTERY_CRITICAL_VOLTAGE: {voltage:.2f} V "
                f"(minimum {self.BATTERY_MIN_VOLTAGE} V)"
            )
            print(f"[DRIVER]  Arming BLOCKED — {reason}")
            return False, reason

        # ── Percentage checks ─────────────────────────────────────────
        if remaining != -1:
            if remaining < self.BATTERY_MIN_PCT:
                reason = (
                    f"BATTERY_TOO_LOW: {remaining}% "
                    f"(minimum {self.BATTERY_MIN_PCT}%)"
                )
                print(f"[DRIVER]  Arming BLOCKED — {reason}")
                return False, reason

            if remaining < self.BATTERY_WARN_PCT:
                print(
                    f"[DRIVER]   Battery warning: {remaining}% remaining "
                    f"(below {self.BATTERY_WARN_PCT}% caution threshold). "
                    f"Proceeding with arm — plan a short flight."
                )
                return True, f"BATTERY_WARNING_{remaining}pct"

        v_str = f"{voltage:.2f} V" if voltage is not None else "unknown V"
        p_str = f"{remaining}%" if remaining != -1 else "unknown %"
        print(f"[DRIVER]  Battery OK — {v_str}, {p_str}")
        return True, f"BATTERY_OK_{p_str}"

    def arm_vehicle(self) -> bool:
        """
        Arm the motors using the correct ArduPilot sequence with enhanced 
        telemetry safety validations.
        """
        if self._is_armed():
            print("[DRIVER] Vehicle is already armed.")
            return True # Retuning True avoids blocking subsequent automated takeoff runs

        # Step 0 — Explicit Battery Safety Validation (Missing from your original step sequence)
        safe, reason = self._check_battery_safe()
        if not safe:
            print(f"[DRIVER] Cannot arm — Battery safety block: {reason}")
            return False

        # Step 1 — wait for healthy sensors BEFORE setting the mode
        if not self._wait_for_armable(timeout=30):
            print("[DRIVER] Cannot arm — pre-arm checks failed.")
            return False

        # Step 2 — set GUIDED only after the FC is stable
        print("[DRIVER] Setting GUIDED mode...")
        if not self._set_mode("GUIDED"):
            print("[DRIVER] Could not enter GUIDED mode — aborting arm.")
            return False

        # Step 3 — give the FC a moment to settle into GUIDED
        time.sleep(1.0)

        # Step 4 — send arm command (p2=21196 bypasses soft pre-arm warnings)
        print("[DRIVER] Sending arm command...")
        
        # Flush key cache data to ensure fresh verification loops
        with self._cache_lock:
            self._message_cache.pop("COMMAND_ACK", None)
            
        result = self._send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            p1=1,
            p2=21196,   # force-arm magic number — skips soft checks on SITL
        )

        # Step 5 — confirm armed flag in HEARTBEAT with real-time fallback polling
        if result and self._wait_until_armed(timeout=10):
            print("[DRIVER] Armed successfully.")
            return True

        print("[DRIVER] Arming failed — Check Mission Planner HUD or SITL terminal for PreArm errors.")
        return False

    def _wait_until_armed(self, timeout: float = 10) -> bool:
        """Poll HEARTBEAT or clear incoming buffer stream directly until armed flag is found."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._is_armed():
                return True
            
            # Fallback: Read directly from incoming stream buffer if cache loop experiences lag
            msg = self.master.recv_match(type='HEARTBEAT', blocking=False)
            if msg:
                with self._cache_lock:
                    self._message_cache['HEARTBEAT'] = msg
                if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    return True
            time.sleep(0.2)
        return False
    
    def disarm_vehicle(self) -> bool:
        """Disarm the motors and reset the safety state to IDLE."""
        if not self._is_armed():
            print("[DRIVER] Vehicle is already disarmed.")
            return False

        # Tear down timers cleanly
        if self._hover_timer is not None:
            self._hover_timer.cancel()
        with self._state_lock:
            self._state = self._STATE_IDLE

        print("[DRIVER] Disarming...")
        result = self._send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, p1=0
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

    # ======================================================================
    # Public — takeoff  (activates altitude gate)
    # ======================================================================

    def execute_takeoff(self, target_altitude: float = 5.0) -> bool:
        """
        Arm (if needed), command takeoff to target_altitude metres, and
        engage the altitude gate.

        Blocks until the vehicle reaches 95 % of target altitude OR a
        timeout / stuck-altitude condition fires.  All movement commands
        are rejected until the gate clears.
        """
        if not self._is_armed():
            print("[DRIVER]  CRITICAL: Takeoff rejected! Motors must be armed first.")
            return False

        print(f"[DRIVER] Taking off to {target_altitude:.1f} m — commands locked until that altitude is confirmed.")
        success = self._send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, p7=target_altitude
    )
        if not success:
            print("[DRIVER] Takeoff command rejected.")
            return False

        # Start background altitude gate (non-blocking)
        self._start_altitude_monitor(target_altitude)

        # Also block the caller until we arrive (same logic, foreground)
        TAKEOFF_TIMEOUT = 30
        STUCK_THRESHOLD = 0.05
        STUCK_WINDOW    = 8

        deadline    = time.time() + TAKEOFF_TIMEOUT
        stuck_start = time.time()
        last_alt    = 0.0

        while time.time() < deadline:
            alt = self._get_altitude()
            if alt is None:
                print("[DRIVER] Waiting for altitude data...")
                time.sleep(1)
                continue

            print(f"[DRIVER] Altitude: {alt:.2f} m")

            if alt >= (target_altitude - self.ALTITUDE_TOLERANCE):
                print("[DRIVER] Target altitude reached.")
                return True

            if abs(alt - last_alt) > STUCK_THRESHOLD:
                last_alt    = alt
                stuck_start = time.time()
            elif time.time() - stuck_start > STUCK_WINDOW:
                print(f"[DRIVER] Takeoff stuck at {alt:.2f} m — aborting.")
                return False

            time.sleep(1)

        print(f"[DRIVER] Takeoff timed out after {TAKEOFF_TIMEOUT} s.")
        return False

    # ======================================================================
    # Public — landing / RTL
    # ======================================================================

    def land(self) -> bool:
        """
        Land at the current position.
        Blocks until altitude < 0.3 m or a 60 s timeout.

        Voice triggers: "land", "land now", "set down", "touch down"
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
        Switch to RTL — climbs to RTL altitude, flies home, lands.

        Voice triggers: "come home", "return to launch", "go home", "RTL"
        """
        print("[DRIVER] Returning to launch...")
        success = self._set_mode("RTL")
        if not success:
            print("[DRIVER] Failed to switch to RTL mode.")
        return success

    # ======================================================================
    # Public — in-flight velocity
    # ======================================================================

    def send_body_translation(self, vx: float, vy: float, vz: float):
        """
        Send a SET_POSITION_TARGET_LOCAL_NED velocity command (body/NED frame, m/s).
        Republish every 0.2 s — ArduPilot times out velocity commands otherwise.

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
        """
        Internal: republish a velocity command for `duration` seconds, then hover.
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
        if not self._is_command_allowed("move_forward"):
            return False
        if not self.is_flying():
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"[DRIVER] Moving forward at {speed} m/s for {duration} s...")
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
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"[DRIVER] Moving backward at {speed} m/s for {duration} s...")
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
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"[DRIVER] Moving left at {speed} m/s for {duration} s...")
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
            print("[DRIVER] Cannot move — drone is not airborne.")
            return False
        self._accept_command()
        print(f"[DRIVER] Moving right at {speed} m/s for {duration} s...")
        self._move_direction(0.0, 1.0, 0.0, speed, duration)
        return True

    def ascend(self, metres: float = 5.0, speed: float = 1.5) -> bool:
        """
        Climb `metres` relative to current altitude at `speed` m/s.

        Voice triggers: "go up", "ascend", "climb X metres",
                        "go up X metres", "increase altitude"
        """
        if not self._is_command_allowed("ascend"):
            return False
        if not self.is_flying():
            print("[DRIVER] Cannot ascend — drone is not airborne.")
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            print("[DRIVER] Cannot read current altitude.")
            return False

        self._accept_command()
        target_alt = current_alt + metres
        print(f"[DRIVER] Ascending {metres} m to {target_alt:.1f} m...")

        timeout  = metres / speed + 10
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
            self.send_body_translation(0.0, 0.0, -speed)   # negative vz = up
            time.sleep(0.2)

        self.hover()
        print("[DRIVER] Ascent timed out.")
        return False

    def descend(self, metres: float = 5.0, speed: float = 1.0) -> bool:
        """
        Descend `metres` relative to current altitude at `speed` m/s.
        Stops at 1 m minimum to avoid unintentional landing.

        Voice triggers: "go down", "descend", "drop X metres",
                        "go down X metres", "decrease altitude", "lower"
        """
        if not self._is_command_allowed("descend"):
            return False
        if not self.is_flying():
            print("[DRIVER] Cannot descend — drone is not airborne.")
            return False

        current_alt = self._get_altitude()
        if current_alt is None:
            print("[DRIVER] Cannot read current altitude.")
            return False

        MIN_SAFE_ALT   = 1.0
        target_alt     = max(current_alt - metres, MIN_SAFE_ALT)
        actual_descent = current_alt - target_alt

        if actual_descent <= 0:
            print(f"[DRIVER] Already at or below {MIN_SAFE_ALT} m — not descending.")
            return False

        self._accept_command()
        print(f"[DRIVER] Descending {actual_descent:.1f} m to {target_alt:.1f} m...")

        timeout  = actual_descent / speed + 10
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
            self.send_body_translation(0.0, 0.0, speed)   # positive vz = down
            time.sleep(0.2)

        self.hover()
        print("[DRIVER] Descent timed out.")
        return False

    def hover(self, duration: float = 0.0) -> bool:
        """
        Immediately stop all translational motion and hold position.

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
    # Public — yaw / heading
    # ======================================================================

    def rotate_left(self, degrees: float = 90.0) -> bool:
        """
        Yaw counter-clockwise by `degrees`.

        Voice triggers: "turn left", "rotate left", "yaw left X degrees",
                        "spin left"
        """
        if not self._is_command_allowed("rotate_left"):
            return False
        if not self.is_flying():
            print("[DRIVER] Cannot rotate — drone is not airborne.")
            return False
        self._accept_command()
        print(f"[DRIVER] Rotating left {degrees}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(degrees), p2=20.0, p3=-1.0, p4=1.0,
        )

    def rotate_right(self, degrees: float = 90.0) -> bool:
        """
        Yaw clockwise by `degrees`.

        Voice triggers: "turn right", "rotate right", "yaw right X degrees",
                        "spin right"
        """
        if not self._is_command_allowed("rotate_right"):
            return False
        if not self.is_flying():
            print("[DRIVER] Cannot rotate — drone is not airborne.")
            return False
        self._accept_command()
        print(f"[DRIVER] Rotating right {degrees}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(degrees), p2=20.0, p3=1.0, p4=1.0,
        )

    def set_heading(self, heading_degrees: float) -> bool:
        """
        Point to an absolute compass bearing (0–360°).
        0° = North, 90° = East, 180° = South, 270° = West.

        Voice triggers: "face north/south/east/west",
                        "turn to X degrees", "heading X"
        """
        if not self._is_command_allowed("set_heading"):
            return False
        if not self.is_flying():
            print("[DRIVER] Cannot set heading — drone is not airborne.")
            return False
        self._accept_command()
        heading_degrees %= 360.0
        print(f"[DRIVER] Setting heading to {heading_degrees:.0f}°...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            p1=float(heading_degrees), p2=20.0, p3=1.0, p4=0.0,
        )

    # ======================================================================
    # Public — GPS navigation
    # ======================================================================

    def goto_waypoint(self, lat: float, lon: float,
                      alt: Optional[float] = None) -> bool:
        """
        Fly to a GPS coordinate in GUIDED mode.
        If `alt` is None the drone maintains its current altitude.

        Voice triggers: "go to coordinates X Y", "fly to latitude X
                        longitude Y", "navigate to waypoint"
        """
        if not self._is_command_allowed("goto_waypoint"):
            return False
        if not self.is_flying():
            print("[DRIVER] Cannot navigate — drone is not airborne.")
            return False

        if alt is None:
            alt = self._get_altitude() or 10.0
            print(f"[DRIVER] No altitude specified — maintaining {alt:.1f} m.")

        self._accept_command()
        print(f"[DRIVER] Navigating to ({lat:.6f}, {lon:.6f}) at {alt:.1f} m...")

        if not self._set_mode("GUIDED"):
            print("[DRIVER] Cannot switch to GUIDED — navigation aborted.")
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

    # ======================================================================
    # Public — speed control
    # ======================================================================

    def set_airspeed(self, speed_ms: float) -> bool:
        """
        Set the target airspeed (m/s).

        Voice triggers: "set airspeed to X", "airspeed X metres per second"
        """
        print(f"[DRIVER] Setting airspeed to {speed_ms:.1f} m/s...")
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
        print(f"[DRIVER] Setting groundspeed to {speed_ms:.1f} m/s...")
        return self._send_command_long(
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            p1=1.0, p2=float(speed_ms), p3=-1.0,
        )

    # ======================================================================
    # Public — flight mode shortcuts
    # ======================================================================

    def change_flight_mode(self, mode_name: str) -> bool:
        """Change to an arbitrary ArduPilot flight mode by name."""
        if not self._is_command_allowed(f"change_flight_mode({mode_name})"):
            return False
        self._accept_command()
        print(f"[DRIVER] Changing flight mode to: {mode_name}")
        success = self._set_mode(mode_name)
        if not success:
            print(f"[DRIVER] Failed to change mode to {mode_name}.")
        return success

    def set_loiter(self) -> bool:
        """
        Switch to LOITER — GPS-assisted position hold with yaw control.

        Voice triggers: "loiter", "loiter mode", "circle", "hover in place"
        """
        if not self._is_command_allowed("set_loiter"):
            return False
        self._accept_command()
        print("[DRIVER] Switching to LOITER mode...")
        return self._set_mode("LOITER")

    def set_position_hold(self) -> bool:
        """
        Switch to POSHOLD — full position, velocity, and altitude hold.

        Voice triggers: "hold position", "position hold", "stay here",
                        "maintain position"
        """
        if not self._is_command_allowed("set_position_hold"):
            return False
        self._accept_command()
        print("[DRIVER] Switching to POSHOLD mode...")
        return self._set_mode("POSHOLD")

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
            print(f"[DRIVER] Current altitude: {alt:.2f} m")
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
            print(f"[DRIVER] Current heading: {heading:.1f}°")
            return heading
        return None

    def get_groundspeed(self) -> Optional[float]:
        """
        Return current ground speed in m/s, or None.

        Voice triggers: "how fast am I going", "current speed",
                        "what is my speed", "ground speed"
        """
        msg = self._get_global_position()
        if msg:
            vx    = msg.vx / 100.0
            vy    = msg.vy / 100.0
            speed = math.sqrt(vx ** 2 + vy ** 2)
            print(f"[DRIVER] Current ground speed: {speed:.2f} m/s")
            return speed
        return None

    def get_battery(self) -> Optional[dict]:
        """
        Return battery status as a dict, or None if unavailable.

        Returns:
            { "voltage_v": float, "current_a": float, "remaining_pct": int }

        Voice triggers: "check battery", "battery level",
                        "how much battery do I have", "battery status"
        """
        msg = self._get_cached("BATTERY_STATUS")
        if msg:
            voltage   = msg.voltages[0] / 1000.0 if msg.voltages[0] != 65535 else None
            current   = msg.current_battery / 100.0 if msg.current_battery != -1 else -1
            remaining = msg.battery_remaining
            print(f"[DRIVER] Battery: {voltage:.2f} V, {remaining}% remaining")
            return {"voltage_v": voltage, "current_a": current, "remaining_pct": remaining}

        msg = self._get_cached("SYS_STATUS")
        if msg:
            voltage   = msg.voltage_battery / 1000.0
            current   = msg.current_battery / 100.0 if msg.current_battery != -1 else -1
            remaining = msg.battery_remaining
            print(f"[DRIVER] Battery: {voltage:.2f} V, {remaining}% remaining")
            return {"voltage_v": voltage, "current_a": current, "remaining_pct": remaining}

        print("[DRIVER] Battery status unavailable.")
        return None

    def get_gps_status(self) -> Optional[dict]:
        """
        Return GPS fix status, or None.

        Returns:
            { "fix_type": int, "satellites_visible": int, "hdop": float }

        Voice triggers: "check GPS", "GPS status", "satellite count",
                        "how is my GPS signal"
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
                f"[DRIVER] GPS: {fix_names.get(msg.fix_type, str(msg.fix_type))}, "
                f"{msg.satellites_visible} satellites, HDOP {result['hdop']:.1f}"
            )
            return result

        print("[DRIVER] GPS status unavailable.")
        return None

    def get_current_mode(self) -> Optional[str]:
        """
        Return the vehicle's current flight mode name, or None.

        Voice triggers: "what mode am I in", "current flight mode",
                        "which mode", "what mode is the drone in"
        """
        msg = self._get_cached("HEARTBEAT")
        if msg:
            mode_name = mavutil.mode_string_v10(msg)
            print(f"[DRIVER] Current mode: {mode_name}")
            return mode_name
        print("[DRIVER] Could not read current mode.")
        return None

    def get_location(self) -> Optional[dict]:
        """
        Return current GPS position and altitude, or None.

        Returns:
            { "lat": float, "lon": float,
              "alt_relative": float, "alt_asl": float }

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
    # Public — emergency
    # ======================================================================

    def trigger_emergency_safe_state(self) -> str:
        """
        Cancel all pending timers, override the altitude gate, and
        switch to the safest available mode.

        Returns "RTL_ACTIVATED" or "LAND_ACTIVATED".
        """
        print("[CRITICAL] EMERGENCY SAFE STATE TRIGGERED!")

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
    # Public — connection lifecycle
    # ======================================================================

    def close_connection(self):
        """Shut down the listener thread and close the MAVLink connection."""
        print("[DRIVER] Closing vehicle connection...")
        if self._hover_timer is not None:
            self._hover_timer.cancel()
        self._listener_running = False
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2)
        if self.master:
            self.master.close()