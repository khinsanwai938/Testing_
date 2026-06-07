import time
import os
import math
from datetime import datetime
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

class MAVLinkDroneDriver:
    def __init__(self, connection_string='127.0.0.1:14551'):
        print(f"[DRONE DRIVER] Connecting to vehicle on: {connection_string}")
        self.vehicle = connect(connection_string, wait_ready=True)
        print("[DRONE DRIVER] Vehicle link established successfully.")
        self.saved_waypoints = []

    def arm_vehicle(self):
        if self.vehicle.armed:
            print("[DRONE DRIVER] Vehicle is already armed.")
            return False
            
        print("[DRONE DRIVER] Running pre-arm safety checks...")
        timeout = 5
        while not self.vehicle.is_armable and timeout > 0:
            print("[DRONE DRIVER] Waiting for vehicle to initialize (GPS/Gyros)...")
            time.sleep(1)
            timeout -= 1
            
        print("[DRONE DRIVER] Sending direct MAVLink ARM command signal...")
        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.5)
        
        self.vehicle.armed = True
        
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, # 1 to arm
            0, 0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        time.sleep(2)
        return True

    def disarm_vehicle(self):
        print("[DRONE DRIVER] Sending disarm command...")
        self.vehicle.armed = False
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, # 0 to disarm
            0, 0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        time.sleep(1)
        return not self.vehicle.armed

    def execute_takeoff(self, target_altitude=5.0):
        print("[DRONE DRIVER] Switching flight mode to GUIDED...")
        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.5)
        print(f"[DRONE DRIVER] Initiating takeoff sequence to target altitude: {target_altitude}m...")
        self.vehicle.simple_takeoff(target_altitude)
        return True

    def is_flying(self):
        try:
            return self.vehicle.armed and self.vehicle.location.global_relative_frame.alt > 0.5
        except AttributeError:
            return False

    def change_mode(self, mode_name):
        print(f"[DRONE DRIVER] Changing flight mode to {mode_name}...")
        self.vehicle.mode = VehicleMode(mode_name)

    def set_absolute_altitude(self, altitude):
        try:
            current_location = self.vehicle.location.global_relative_frame
            if current_location:
                print(f"[DRONE DRIVER] Adjusting absolute target altitude to {altitude}m...")
                target_location = LocationGlobalRelative(current_location.lat, current_location.lon, altitude)
                self.vehicle.simple_goto(target_location)
        except AttributeError:
            print("[DRONE DRIVER] Altitude command failed: Telemetry frames not fully ready.")

    def send_body_translation(self, x, y, z):
        try:
            current_loc = self.vehicle.location.global_relative_frame
            if not current_loc or current_loc.lat is None:
                print("[DRONE DRIVER] Translation error: Weak GPS positional telemetry.")
                return

            print(f"[DRONE DRIVER] Calculating physical GPS offsets -> Forward/Back(X): {x}m, Left/Right(Y): {y}m")
            R = 6378137
            dLat = x / R
            dLon = y / (R * math.cos(math.pi * current_loc.lat / 180.0))
            
            target_lat = current_loc.lat + (dLat * 180.0 / math.pi)
            target_lon = current_loc.lon + (dLon * 180.0 / math.pi)
            target_alt = current_loc.alt + z
            
            target_location = LocationGlobalRelative(target_lat, target_lon, target_alt)
            self.vehicle.simple_goto(target_location)
        except AttributeError:
            print("[DRONE DRIVER] Translation error: Telemetry frames not fully populated.")

    def save_waypoint(self):
        try:
            current_loc = self.vehicle.location.global_relative_frame
            if current_loc and current_loc.lat is not None:
                self.saved_waypoints.append(current_loc)
                waypoint_idx = len(self.saved_waypoints)
                print(f"[DRONE DRIVER] Waypoint {waypoint_idx} saved.")
                return waypoint_idx
        except AttributeError:
            print("[DRONE DRIVER] Save failed: Empty spatial telemetry data structure.")
        return 0

    def goto_waypoint(self, index):
        if 0 < index <= len(self.saved_waypoints):
            target_loc = self.saved_waypoints[index - 1]
            self.vehicle.simple_goto(target_loc)
            return True
        return False

    def export_waypoints_to_file(self):
        if not self.saved_waypoints:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_mission_{timestamp}.waypoints"
        try:
            with open(filename, "w") as f:
                f.write("QGC WPL 110\n")
                home = self.vehicle.home_location or self.vehicle.location.global_relative_frame
                f.write(f"0\t1\t0\t16\t0.000000\t0.000000\t0.000000\t0.000000\t{home.lat:.6f}\t{home.lon:.6f}\t{home.alt:.6f}\t1\n")
                for idx, loc in enumerate(self.saved_waypoints, start=1):
                    f.write(f"{idx}\t0\t3\t16\t0.000000\t0.000000\t0.000000\t0.000000\t{loc.lat:.6f}\t{loc.lon:.6f}\t{loc.alt:.6f}\t1\n")
            return filename
        except Exception:
            return None

    def trigger_emergency_safe_state(self):
        print("[CRITICAL] EMERGENCY SAFE STATE TRIGGERED!")
        self.change_mode("LOITER")
        time.sleep(1)
        
        try:
            current_alt = self.vehicle.location.global_relative_frame.alt
            if current_alt > 3.0:
                self.change_mode("RTL")
                return "RTL_ACTIVATED"
        except AttributeError:
            pass
            
        self.change_mode("LAND")
        return "LAND_ACTIVATED"

    def close(self):
        self.vehicle.close()