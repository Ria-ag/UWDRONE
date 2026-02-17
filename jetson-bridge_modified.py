cat << 'EOF' > jetson_bridge_mode.py
import time
import sys
from pymavlink import mavutil
from cv_node import cv_node

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

# UPDATE THIS IP to match your Ground Station Laptop!
GCS_IP = input("Enter GCS IP Address: ")
GCS_PORT = 14550
CMD_PORT = 14551

# --- CONSTANTS ---
SAFE_SPEED = 0.1  # <--- CHANGED FROM 0.5 TO 0.1 m/s
COM_RC_OVERRIDE = 1
COM_OBL_RC_ACT = 1

# --- SETUP ---
print(f"Connecting to Pixhawk on {SERIAL_PORT}...")
try:
    vehicle = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE, source_system=1)
except Exception as e:
    print(f"Error connecting to vehicle: {e}")
    sys.exit(1)

print(f"Opening UDP Bridge to {GCS_IP}:{GCS_PORT}...")
gcs = mavutil.mavlink_connection(f'udpout:{GCS_IP}:{GCS_PORT}', source_system=1)

print(f"Opening Command Listener on Port {CMD_PORT}...")
cmd_listener = mavutil.mavlink_connection(f'udpin:0.0.0.0:{CMD_PORT}', source_system=1)

vehicle.wait_heartbeat()
print(f"Vehicle Connected! Mode: {vehicle.flightmode}")

# --- STATE VARIABLES ---
target_vx = 0
target_vy = 0
target_vz = 0
last_velocity_time = 0
VELOCITY_INTERVAL = 0.1 

def send_velocity_command(vx, vy, vz):
    type_mask = int(0b110111000111)
    vehicle.mav.set_position_target_local_ned_send(
        0, vehicle.target_system, vehicle.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, 
        type_mask, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0
    )

cv_node = cv_node()

MODE_RC = 0
MODE_MANUAL = 1
MODE_AUTO = 2

current_mode = MODE_AUTO

print(f"Bridge Active. Speed Limit: {SAFE_SPEED} m/s")

try:
    while True:
        # 1. BRIDGE: Downlink
        msg = vehicle.recv_match(blocking=False)
        if msg:
            gcs.write(msg.get_msgbuf())

        # 2. BRIDGE: Uplink
        gcs_msg = gcs.recv_match(blocking=False)
        if gcs_msg:
            vehicle.write(gcs_msg.get_msgbuf())

        # 3. COMMAND LISTENER
        manual_override = False
        cmd_msg = cmd_listener.recv_match(blocking=False)
        if cmd_msg:
            if cmd_msg.get_type() == 'COMMAND_LONG' and cmd_msg.command == 31010:
                manual_override = True
                action_id = int(cmd_msg.param1)
                
                if action_id == 0:   # STOP
                    target_vx, target_vy, target_vz = 0, 0, 0
                    print("CMD: STOP")
                elif action_id == 1: # NORTH
                    if current_mode == MODE_MANUAL:
                        target_vx, target_vy, target_vz = SAFE_SPEED, 0, 0
                    print("CMD: NORTH")
                elif action_id == 2: # SOUTH
                    if current_mode == MODE_MANUAL:
                        target_vx, target_vy, target_vz = -SAFE_SPEED, 0, 0
                    print("CMD: SOUTH")
                elif action_id == 3: # EAST
                    if current_mode == MODE_MANUAL:
                        target_vx, target_vy, target_vz = 0, SAFE_SPEED, 0
                    print("CMD: EAST")
                elif action_id == 4: # WEST
                    if current_mode == MODE_MANUAL:
                        target_vx, target_vy, target_vz = 0, -SAFE_SPEED, 0
                    print("CMD: WEST")
                elif action_id == 100: # AUTO
                    current_mode = MODE_AUTO
                    print("Switched to AUTO")
                elif action_id == 101: # MANUAL
                    current_mode = MODE_MANUAL
                    print("Switched to MANUAL")
                elif action_id == 102: #RC  
                    current_mode = MODE_RC
                    print("Switched to RC")

        # 4. CV AUTONOMY
        if current_mode == MODE_AUTO:
            cv_cmd = cv_node.step()

            if cv_cmd:
                offset_x = cv_cmd["offset_x"]   # negative -> object left, positive -> object right
                depth = cv_cmd["depth_m"]
                print("[BRIDGE] CV command received:", cv_cmd)

                # Forward/backward: maintain ~1 meter distance
                if depth is not None:
                    if depth > 1.2:
                        target_vx = SAFE_SPEED
                    elif depth < 0.8:
                        target_vx = -SAFE_SPEED
                    else:
                        target_vx = 0
                else:
                    target_vx = 0

                # Left/right to center object in frame
                target_vy = -SAFE_SPEED * offset_x  # positive offset_x -> object right -> move right

                target_vz = 0  # keep altitude constant for now

        # 5. EXECUTE AUTONOMY
        current_time = time.time()
        if (current_time - last_velocity_time > VELOCITY_INTERVAL):
            
            if current_mode in [MODE_AUTO, MODE_MANUAL]:
                send_velocity_command(target_vx, target_vy, target_vz)
            else:
                # Heartbeat (0,0,0)
                send_velocity_command(0, 0, 0)
            
            last_velocity_time = current_time
            
        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopping...")
    vehicle.close()
    gcs.close()
    cmd_listener.close()
EOF