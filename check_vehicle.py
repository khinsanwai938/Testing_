import time
from pymavlink import mavutil

def detect_and_route_vehicle(connection_string="udpin:localhost:14551"):
    print(f"Connecting to Mission Planner SITL on {connection_string}...")
    # Establish connection (Mission Planner SITL usually outputs on 14550 or 14551)
    master = mavutil.mavlink_connection(connection_string)
    
    # Wait for the first heartbeat message to identify the vehicle type
    print("Waiting for heartbeat...")
    heartbeat = master.wait_heartbeat(timeout=10)
    
    if not heartbeat:
        print("Error: No heartbeat received from SITL. Is Mission Planner running?")
        return

    # Extract MAV_TYPE enum integer
    vehicle_type = heartbeat.type
    print(f"Connected! Detected MAVLink System ID: {master.target_system}, MAV_TYPE ID: {vehicle_type}")

    # Check vehicle category
    if vehicle_type in [mavutil.mavlink.MAV_TYPE_FIXED_WING]:
        print("\n>>> SUCCESS: Detected Plane SITL. Executing plane pipeline...")
        run_plane_commands(master)
        
    elif vehicle_type in [mavutil.mavlink.MAV_TYPE_QUADROTOR, 
                        mavutil.mavlink.MAV_TYPE_HEXAROTOR, 
                        mavutil.mavlink.MAV_TYPE_OCTOROTOR, 
                        mavutil.mavlink.MAV_TYPE_TRICOPTER, 
                        mavutil.mavlink.MAV_TYPE_HELICOPTER]:
        print("\n>>> SUCCESS: Detected Drone/Multirotor SITL. Executing drone pipeline...")
        run_drone_commands(master)
        
    else:
        print(f"Detected an unsupported or hybrid vehicle type ({vehicle_type}). Executing universal safety fallback...")
        run_universal_commands(master)

# =====================================================================
# Specific Commands for Drone
# =====================================================================
def run_drone_commands(master):
    print("[DRONE CMD] Setting Mode to GUIDED...")
    master.set_mode_rtl() # Just an example or custom guided setup
    # Drones require arming followed immediately by a specific takeoff altitude command
    print("[DRONE CMD] Arming motors...")
    master.arducopter_arm() 
    print("[DRONE CMD] Sending Vertical Takeoff command to 10 meters...")
    # MAV_CMD_NAV_TAKEOFF (22): Pitch, Empty, Empty, Yaw, Lat, Lon, Alt
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0, 0, 0, 10
    )

# =====================================================================
# Specific Commands for Plane
# =====================================================================
def run_plane_commands(master):
    print("[PLANE CMD] Setting Mode to TAKEOFF or FBWA...")
    # Planes often require passing to a runway launch/takeoff sequence mode or FBWA
    master.set_mode('TAKEOFF') 
    print("[PLANE CMD] Arming propulsion throttle...")
    master.arducopter_arm() # The underlying arm command helper remains identical
    print("[PLANE CMD] Setting target airspeed cruise to 15 m/s...")
    # Planes manage forward velocity via airspeed
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0,
        0, 15, -1, 0, 0, 0, 0 # Speed type 0 (Airspeed), 15 m/s, Throttle unchanged
    )

# =====================================================================
# Universal Commands (Both)
# =====================================================================
def run_universal_commands(master):
    print("[BOTH] Requesting general GPS telemetry...")
    # Works seamlessly across all vehicles
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=3)
    if msg:
        print(f"[Telemetry] Relative Altitude: {msg.relative_alt / 1000.0}m")
    
    print("[BOTH] Triggering universal Return To Launch (RTL)...")
    master.set_mode_rtl()

if __name__ == '__main__':
    # Adjust port configuration based on your Mission Planner settings
    detect_and_route_vehicle("udpin:127.0.0.1:14551")