import time
import threading
import sys
from pymavlink import mavutil
from cv_yolo import cv_yolo

SERIAL_PORT = '/dev/ttyACM0'
BAUD = 115200
SAFE_SPEED = 0.2

MODE_RC = 0
MODE_MANUAL = 1
MODE_AUTO = 2

current_mode = MODE_RC

target_vx = 0
target_vy = 0
target_vz = 0

print("Connecting Pixhawk...")
vehicle = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD)
vehicle.wait_heartbeat()
print("Pixhawk connected")

# QGC bridge (optional but recommended)
gcs = mavutil.mavlink_connection('udpout:127.0.0.1:14550')

cv = cv_yolo()

def send_velocity(vx, vy, vz):
    vehicle.mav.set_position_target_local_ned_send(
        0,
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        int(0b110111000111),
        0,0,0,
        vx,vy,vz,
        0,0,0,
        0,0
    )

def set_offboard():
    vehicle.set_mode_px4("OFFBOARD")
    time.sleep(0.5)

def keyboard_thread():
    global target_vx, target_vy, target_vz, current_mode

    print("\nKeyboard Controls")
    print("WASD movement")
    print("SPACE stop")
    print("T auto")
    print("M manual")
    print("R rc")

    while True:
        key = input("> ").lower()

        if key == 't':
            current_mode = MODE_AUTO
            set_offboard()
            print("AUTO")

        elif key == 'm':
            current_mode = MODE_MANUAL
            set_offboard()
            print("MANUAL")

        elif key == 'r':
            current_mode = MODE_RC
            print("RC")

        elif current_mode == MODE_MANUAL:

            if key == 'w':
                target_vx = SAFE_SPEED
                target_vy = 0

            elif key == 's':
                target_vx = -SAFE_SPEED
                target_vy = 0

            elif key == 'd':
                target_vy = SAFE_SPEED
                target_vx = 0

            elif key == 'a':
                target_vy = -SAFE_SPEED
                target_vx = 0

            elif key == ' ':
                target_vx = 0
                target_vy = 0


threading.Thread(target=keyboard_thread, daemon=True).start()

last_send = time.time()

while True:

    # telemetry bridge to QGC
    msg = vehicle.recv_match(blocking=False)
    if msg:
        gcs.write(msg.get_msgbuf())

    # AUTO MODE
    if current_mode == MODE_AUTO:
        cv_cmd = cv.step()

        if cv_cmd:
            offset = cv_cmd["offset_x"]
            depth = cv_cmd["depth_m"]

            if depth:
                if depth > 1.2:
                    target_vx = SAFE_SPEED
                elif depth < 0.8:
                    target_vx = -SAFE_SPEED
                else:
                    target_vx = 0

            target_vy = -offset * SAFE_SPEED

    # send at 20Hz
    if time.time() - last_send > 0.05:

        if current_mode in [MODE_AUTO, MODE_MANUAL]:
            send_velocity(target_vx, target_vy, target_vz)
        else:
            send_velocity(0,0,0)

        last_send = time.time()

    time.sleep(0.002)