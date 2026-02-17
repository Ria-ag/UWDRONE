cat << 'EOF' > jetson_bridge_mode.py
import time
import sys
from pymavlink import mavutil
import cv2
import numpy as np
import pytesseract

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

# Ask for GCS IP dynamically
GCS_IP = input("Enter GCS IP Address: ")
GCS_PORT = 14550
CMD_PORT = 14551

# --- CONSTANTS ---
SAFE_SPEED = 0.1  # m/s

# --- GStreamer helper ---
def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )

# --- CV Node ---
class CVNode:
    def __init__(
        self,
        camera_left=0,
        camera_right=1,
        width=1280,
        height=720,
        fps=30,
        publish_rate=5.0,
        detection_type="color",
        enable_depth=True,
        baseline=0.12,
        focal_length=525.0
    ):
        self.capL = cv2.VideoCapture(
            gstreamer_pipeline(sensor_id=camera_left, width=width, height=height),
            cv2.CAP_GSTREAMER
        )

        self.capR = None
        if enable_depth and camera_right is not None:
            self.capR = cv2.VideoCapture(
                gstreamer_pipeline(sensor_id=camera_right, width=width, height=height),
                cv2.CAP_GSTREAMER
            )

        if not self.capL.isOpened():
            raise RuntimeError("Failed to open left camera")

        self.detection_type = detection_type
        self.publish_rate = publish_rate
        self.enable_depth = enable_depth
        self.baseline = baseline
        self.focal_length = focal_length

        if enable_depth:
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=128,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            self.stereo = None

        self.last_publish = time.time()
        self.frame_count = 0
        self.start_time = time.time()

    def _red_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,120,70), (10,255,255))
        mask2 = cv2.inRange(hsv, (170,120,70), (180,255,255))
        return mask1 | mask2

    def step(self):
        now = time.time()
        if now - self.last_publish < 1.0 / self.publish_rate:
            return None
        self.last_publish = now
        t0 = time.time()

        retL, frameL = self.capL.read()
        if not retL:
            return None

        frameR = None
        if self.enable_depth and self.capR:
            retR, frameR = self.capR.read()
            if not retR:
                return None

        mask = self._red_mask(frameL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 500:
            return None

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2
        img_h, img_w, _ = frameL.shape
        offset_x = (cx - img_w / 2) / (img_w / 2)

        # Map offset to velocity
        vx = 0
        vy = -offset_x * SAFE_SPEED  # steer left/right based on offset
        vz = 0

        confidence = min(1.0, area / 5000.0)

        latency_ms = (time.time() - t0) * 1000
        self.frame_count += 1
        fps = self.frame_count / max(time.time() - self.start_time, 1e-6)

        return {
            "vx": vx * confidence,
            "vy": vy * confidence,
            "vz": vz,
            "latency_ms": latency_ms,
            "fps": fps
        }

    def release(self):
        self.capL.release()
        if self.capR:
            self.capR.release()

# --- SETUP MAVLINK ---
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

# --- INIT CV ---
cv_node = CVNode()
AUTONOMY_ENABLED = True

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
                elif action_id == 1: # NORTH
                    target_vx, target_vy, target_vz = SAFE_SPEED, 0, 0
                elif action_id == 2: # SOUTH
                    target_vx, target_vy, target_vz = -SAFE_SPEED, 0, 0
                elif action_id == 3: # EAST
                    target_vx, target_vy, target_vz = 0, SAFE_SPEED, 0
                elif action_id == 4: # WEST
                    target_vx, target_vy, target_vz = 0, -SAFE_SPEED, 0

        # 4. CV AUTONOMY
        if AUTONOMY_ENABLED and not manual_override:
            cv_cmd = cv_node.step()
            if cv_cmd:
                target_vx = cv_cmd["vx"]
                target_vy = cv_cmd["vy"]
                target_vz = cv_cmd["vz"]

        # 5. EXECUTE VELOCITY
        current_time = time.time()
        if (current_time - last_velocity_time > VELOCITY_INTERVAL):
            mode = vehicle.flightmode
            if mode in ['GUIDED', 'OFFBOARD']:
                send_velocity_command(target_vx, target_vy, target_vz)
            else:
                send_velocity_command(0, 0, 0)
            last_velocity_time = current_time

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\nStopping...")
    vehicle.close()
    gcs.close()
    cmd_listener.close()
    cv_node.release()
EOF
