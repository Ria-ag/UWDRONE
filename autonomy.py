import time
import sys
from pymavlink import mavutil
from multi_detect_cv import cv_yolo

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

GCS_IP = input("GCS IP: ")
GCS_PORT = 14550
CMD_PORT = 14551

SAFE_SPEED = 0.25
VELOCITY_HZ = 20

MODE_RC = 0
MODE_MANUAL = 1
MODE_AUTO = 2

current_mode = MODE_RC

print("[SYS] Connecting PX4...")
vehicle = mavutil.mavlink_connection(SERIAL_PORT, baud=BAUD_RATE)
vehicle.wait_heartbeat()
print("[SYS] Connected")

gcs = mavutil.mavlink_connection(f"udpout:{GCS_IP}:{GCS_PORT}")
cmd_listener = mavutil.mavlink_connection(f"udpin:0.0.0.0:{CMD_PORT}")

cv = cv_yolo()

vx, vy, vz = 0, 0, 0
last_send = 0

def set_offboard(enable: bool):
    """
    PX4 OFFBOARD must be continuously fed.
    We only "logically" enable it by sending velocity stream.
    """
    if enable:
        print("[MODE] OFFBOARD ENABLED")
    else:
        print("[MODE] OFFBOARD DISABLED (RC fallback)")


def send_velocity(vx, vy, vz):
    vehicle.mav.set_position_target_local_ned_send(
        0,
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        int(0b110111000111),
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )


def zero():
    send_velocity(0, 0, 0)

print("[SYS] Running...")

try:
    while True:
        msg = vehicle.recv_match(blocking=False)
        if msg:
            gcs.write(msg.get_msgbuf())

        gcs_msg = gcs.recv_match(blocking=False)
        if gcs_msg:
            vehicle.write(gcs_msg.get_msgbuf())

        cmd_msg = cmd_listener.recv_match(blocking=False)

        if cmd_msg and cmd_msg.get_type() == "COMMAND_LONG":
            if int(cmd_msg.command) == 31010:
                action = int(cmd_msg.param1)

                # MODE SWITCHING
                if action == 100:
                    current_mode = MODE_AUTO
                    set_offboard(True)

                elif action == 101:
                    current_mode = MODE_MANUAL
                    set_offboard(True)

                elif action == 102:
                    current_mode = MODE_RC
                    set_offboard(False)

                # MANUAL CONTROL
                elif current_mode == MODE_MANUAL:

                    if action == 0:
                        vx, vy, vz = 0, 0, 0
                    elif action == 1:
                        vx, vy, vz = SAFE_SPEED, 0, 0
                    elif action == 2:
                        vx, vy, vz = -SAFE_SPEED, 0, 0
                    elif action == 3:
                        vx, vy, vz = 0, SAFE_SPEED, 0
                    elif action == 4:
                        vx, vy, vz = 0, -SAFE_SPEED, 0

        if current_mode == MODE_AUTO:
            cv_out = cv.step()

            if cv_out:
                offset = cv_out["offset_x"]
                depth = cv_out["depth_m"]

                # forward motion baseline
                vx = SAFE_SPEED

                # steer toward target / avoid obstacle bias
                vy = -SAFE_SPEED * offset

                # crude depth avoidance
                if depth:
                    if depth < 0.8:
                        vx = -SAFE_SPEED   # back off
                    elif depth > 1.5:
                        vx = SAFE_SPEED    # move forward
                else:
                    vx = 0

                vz = 0

        if current_mode == MODE_RC:
            vx, vy, vz = 0, 0, 0

        now = time.time()

        if now - last_send > 1.0 / VELOCITY_HZ:

            if current_mode in [MODE_MANUAL, MODE_AUTO]:
                send_velocity(vx, vy, vz)
            else:
                zero()

            last_send = now

        time.sleep(0.002)


except KeyboardInterrupt:
    print("\n[SYS] stopping")

finally:
    zero()
    cv.release()
    vehicle.close()
    gcs.close()
    cmd_listener.close()