import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil

class MAVLinkDroneDriver:
    def __init__(self, connection_string='127.0.0.1:14550'):
        print(f"[DRONE DRIVER] Initializing link on {connection_string}...")
        self.vehicle = connect(connection_string, wait_ready=True)
        print("[DRONE DRIVER] Telemetry stream link confirmed.")

    def arm_vehicle(self):
        if not self.vehicle.armed:
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(0.5)
            self.vehicle.armed = True
            return True
        return False

    def change_mode(self, mode_name):
        """Safely shifts flight controller state parameters."""
        if mode_name in ["RTL", "LOITER", "LAND", "GUIDED"]:
            self.vehicle.mode = VehicleMode(mode_name)
            return True
        return False

    def is_flying(self):
        """Safety checkpoint evaluating vertical posture."""
        return self.vehicle.armed and self.vehicle.location.global_relative_frame.alt >= 0.5

    def get_current_altitude(self):
        return self.vehicle.location.global_relative_frame.alt

    def send_body_translation(self, x, y, z):
        """Moves vehicle relative to its heading vector (Body Frame)."""
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_NED, 
            0b0000111111111000, 
            x, y, z, 
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def set_absolute_altitude(self, target_meters):
        current_loc = self.vehicle.location.global_relative_frame
        self.vehicle.simple_goto(current_loc, altitude=float(target_meters))

    def close(self):
        self.vehicle.close()
        print("[DRONE DRIVER] Telemetry ports safely released.")