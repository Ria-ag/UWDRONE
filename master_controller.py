import time
import sys
from pymavlink import mavutil
# multi_detect_cv returns the {'valid': ...} dict this file expects
# (cv_yolo.py does not, which caused KeyError: 'valid')
from multi_detect_cv import cv_yolo
import numpy as np

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

CV_MAX_AGE = 0.5  # s; older CV results count as "target lost"
# --- CV TARGETING ---
# "aruco": only steer toward ArUco markers (YOLO still runs/streams but can't move the drone)
# "any":   prefer markers, fall back to the largest YOLO object
# "yolo":  largest YOLO object only (old behavior; chases false positives like hands)
CV_TARGET_MODE = "aruco"
CV_TARGET_ARUCO_ID = None   # e.g. 2 to follow only marker 2; None = largest marker
YOLO_CONF = 0.6
YOLO_CLASSES = None         # e.g. [0] for person only; None = all 80 COCO classes
cv = cv_yolo(target_mode=CV_TARGET_MODE, target_aruco_id=CV_TARGET_ARUCO_ID,
            conf_thresh=YOLO_CONF, target_classes=YOLO_CLASSES).start()  # background thread: YOLO can't stall the 20 Hz setpoints

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
            cv_cmd = cv.latest(max_age=CV_MAX_AGE)
            if cv_cmd and cv_cmd["valid"] and cv_cmd.get("depth_m") is not None:
                offset_x = cv_cmd["offset_x"]
                depth = cv_cmd["depth_m"]
                # --- FORWARD CONTROL ---
                TARGET_DIST = 1.0
                depth_error = depth - TARGET_DIST  # NOTE: YOLO depth_m is a size proxy, not meters
                # proportional forward velocity
                target_vx = float(np.clip(depth_error * 0.25, -0.3, 0.3))
                # --- LATERAL CONTROL ---
                target_vy = float(np.clip(-offset_x * 0.35, -0.25, 0.25))
                # altitude fixed for now
                target_vz = 0
                print(
                    f"[AUTO] "
                    f"depth={depth:.2f} "
                    f"offset={offset_x:.2f} "
                    f"vx={target_vx:.2f} "
                    f"vy={target_vy:.2f}"
                )
            else:
                # target lost -> hover
                target_vx = 0
                target_vy = 0
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
