import time
import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

class MAVLinkDroneDriver:
    def __init__(self, connection_string='127.0.0.1:14552'):
        print(f"[DRONE DRIVER] Connecting to vehicle on: {connection_string}")
        self.vehicle = connect(connection_string, wait_ready=True)
        print("[DRONE DRIVER] Vehicle link established successfully.")

    def arm_vehicle(self):
        """Arms the vehicle's propulsion motors safely."""
        if self.vehicle.armed:
            print("[DRONE DRIVER] Vehicle is already armed.")
            return False
            
        print("[DRONE DRIVER] Running pre-arm safety checks...")
        timeout = 5
        while not self.vehicle.is_armable and timeout > 0:
            print("[DRONE DRIVER] Waiting for vehicle to initialize (GPS/Gyros)...")
            time.sleep(1)
            timeout -= 1
            
        print("[DRONE DRIVER] Switching to GUIDED mode for arming...")
        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.5)
        
        print("[DRONE DRIVER] Sending direct MAVLink ARM command signal...")
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,                                         # target system, target component
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, # command
            0,                                            # confirmation
            1,                                            # param1: 1 to arm
            0, 0, 0, 0, 0, 0                              # params 2-7
        )
        self.vehicle.send_mavlink(msg)
        
        # Wait up to 3 seconds for confirmation from the autopilot board
        timeout = 3
        while not self.vehicle.armed and timeout > 0:
            time.sleep(0.5)
            timeout -= 0.5
            
        return self.vehicle.armed

    def disarm_vehicle(self):
        """Disarms the vehicle immediately."""
        print("[DRONE DRIVER] Sending disarm command...")
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,                                            # param1: 0 to disarm
            0, 0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        
        timeout = 3
        while self.vehicle.armed and timeout > 0:
            time.sleep(0.5)
            timeout -= 0.5
            
        return not self.vehicle.armed

    def execute_takeoff(self, target_altitude=5.0):
        """Asynchronously triggers autonomous takeoff."""
        print("[DRONE DRIVER] Switching flight mode to GUIDED...")
        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.5)
        
        print(f"[DRONE DRIVER] Initiating takeoff sequence to target altitude: {target_altitude}m...")
        self.vehicle.simple_takeoff(target_altitude)
        return True

    def is_flying(self):
        """Combines relative altitude with armed state metrics for safety verification."""
        return self.vehicle.armed and self.vehicle.location.global_relative_frame.alt > 0.5

    def change_mode(self, mode_name):
        print(f"[DRONE DRIVER] Changing flight mode to {mode_name}...")
        self.vehicle.mode = VehicleMode(mode_name)

    def set_absolute_altitude(self, altitude):
        """Changes target altitude while holding horizontal position."""
        print(f"[DRONE DRIVER] Adjusting absolute target altitude to {altitude}m...")
        current_location = self.vehicle.location.global_relative_frame
        target_location = LocationGlobalRelative(current_location.lat, current_location.lon, altitude)
        self.vehicle.simple_goto(target_location)

    def send_body_translation(self, x, y, z):
        """Moves the drone to a coordinate offset location using GPS calculation formulas."""
        print(f"[DRONE DRIVER] Calculating physical GPS offsets -> Forward/Back(X): {x}m, Left/Right(Y): {y}m")
        R = 6378137
        current_loc = self.vehicle.location.global_relative_frame
        
        dLat = x / R
        dLon = y / (R * math.cos(math.pi * current_loc.lat / 180.0))
        
        target_lat = current_loc.lat + (dLat * 180.0 / math.pi)
        target_lon = current_loc.lon + (dLon * 180.0 / math.pi)
        target_alt = current_loc.alt + z
        
        target_location = LocationGlobalRelative(target_lat, target_lon, target_alt)
        self.vehicle.simple_goto(target_location)

    def close(self):
        print("[DRONE DRIVER] Closing vehicle connection link...")
        self.vehicle.close()