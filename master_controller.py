import time
import sys
from pymavlink import mavutil
from cv_yolo import cv_yolo

# CONFIG
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
SAFE_SPEED = 0.2
VELOCITY_INTERVAL = 0.05

GCS_IP = input("Enter GCS IP Address: ")
GCS_PORT = 14550
CMD_PORT = 14551

MODE_RC = 0
MODE_MANUAL = 1
MODE_AUTO = 2
current_mode = MODE_RC

# CONNECT PIXHAWK
print("Connecting to Pixhawk...")
vehicle = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE, source_system=1)
vehicle.wait_heartbeat()
print("Connected.")

# UDP BRIDGE
gcs = mavutil.mavlink_connection(f'udpout:{GCS_IP}:{GCS_PORT}', source_system=1)
cmd_listener = mavutil.mavlink_connection(f'udpin:0.0.0.0:{CMD_PORT}', source_system=1)

cv = cv_yolo()

# STATE
target_vx = 0
target_vy = 0
target_vz = 0
last_velocity_time = 0

# FUNCTIONS
def set_offboard():
    vehicle.set_mode_px4("OFFBOARD")
    time.sleep(0.5)

def send_velocity(vx, vy, vz):
    vehicle.mav.set_position_target_local_ned_send(
        0,
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # changed from LOCAL_NED
        int(0b110111000111),
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )

def zero_velocity():
    send_velocity(0,0,0)

# START
print("System Ready. Starting in RC mode.")

try:
    while True:

        # MAVLINK BRIDGE DOWNLINK
        msg = vehicle.recv_match(blocking=False)
        if msg:
            gcs.write(msg.get_msgbuf())

        # MAVLINK BRIDGE UPLINK
        gcs_msg = gcs.recv_match(blocking=False)
        if gcs_msg:
            vehicle.write(gcs_msg.get_msgbuf())

        # COMMAND LISTENER
        cmd_msg = cmd_listener.recv_match(blocking=False)
        if cmd_msg and cmd_msg.get_type() == 'COMMAND_LONG' and cmd_msg.command == 31010:
            action_id = int(cmd_msg.param1)

            # --- MODE SWITCHING ---
            if action_id == 100:
                current_mode = MODE_AUTO
                set_offboard()
                print("MODE -> AUTO")

            elif action_id == 101:
                current_mode = MODE_MANUAL
                set_offboard()
                print("MODE -> MANUAL")

            elif action_id == 102:
                current_mode = MODE_RC
                print("MODE -> RC")

            # --- MANUAL MOVEMENT ---
            elif current_mode == MODE_MANUAL:

                if action_id == 0:
                    target_vx, target_vy, target_vz = 0,0,0
                    print("STOP")

                elif action_id == 1:
                    target_vx = SAFE_SPEED; target_vy = 0; target_vz = 0
                elif action_id == 2:
                    target_vx = -SAFE_SPEED; target_vy = 0; target_vz = 0
                elif action_id == 3:
                    target_vx = 0; target_vy = SAFE_SPEED; target_vz = 0
                elif action_id == 4:
                    target_vx = 0; target_vy = -SAFE_SPEED; target_vz = 0

        # AUTO MODE (CV DRIVEN)
        if current_mode == MODE_AUTO:
            cv_cmd = cv.step()
            if cv_cmd:
                offset_x = cv_cmd["offset_x"]
                depth = cv_cmd["depth_m"]

                # Distance control (~1m target)
                if depth:
                    if depth > 1.2:
                        target_vx = SAFE_SPEED
                    elif depth < 0.8:
                        target_vx = -SAFE_SPEED
                    else:
                        target_vx = 0
                else:
                    target_vx = 0

                target_vy = -SAFE_SPEED * offset_x
                target_vz = 0

        # EXECUTION LOOP
        now = time.time()
        if now - last_velocity_time > VELOCITY_INTERVAL:

            if current_mode in [MODE_MANUAL, MODE_AUTO]:
                send_velocity(target_vx, target_vy, target_vz)

            elif current_mode == MODE_RC:
                zero_velocity()

            last_velocity_time = now

        time.sleep(0.002)

except KeyboardInterrupt:
    print("\nShutting down...")
    zero_velocity()
    cv.release()
    vehicle.close()
    gcs.close()
    cmd_listener.close()
