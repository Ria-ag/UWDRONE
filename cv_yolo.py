import cv2
import time
import numpy as np
from ultralytics import YOLO

# def gstreamer_pipeline(
#     sensor_id=0,
#     capture_width=1280,
#     capture_height=720,
#     framerate=30,
#     flip_method=2,
# ):
#     return (
#         "nvarguscamerasrc sensor-id=%d ! "
#         "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
#         "format=(string)NV12, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, format=(string)BGRx ! "
#         "videoconvert ! video/x-raw, format=(string)BGR ! "
#         "appsink drop=true max-buffers=1 sync=false"
#         % (sensor_id, capture_width, capture_height, framerate, flip_method)
#     )

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, framerate=30):
    return (
        "nvarguscamerasrc sensor-id=%d ee-mode=1 ee-strength=1.0 tnr-mode=1 tnr-strength=1.0 ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=True"
        % (sensor_id, capture_width, capture_height, framerate)
    )

class cv_yolo:
    def __init__(
        self,
        camera_left=0,
        camera_right=1,
        width=1280,
        height=720,
        fps=30,
        publish_rate=10.0,
        target_class=0,
        enable_depth=True,
        baseline=0.12,
        focal_length=700.0
    ):
        print("[CV] Initializing stereo cameras...")

        self.capL = cv2.VideoCapture(
            gstreamer_pipeline(sensor_id=camera_left,
                               capture_width=width,
                               capture_height=height,
                               framerate=fps),
            cv2.CAP_GSTREAMER
        )

        self.capR = cv2.VideoCapture(
            gstreamer_pipeline(sensor_id=camera_right,
                               capture_width=width,
                               capture_height=height,
                               framerate=fps),
            cv2.CAP_GSTREAMER
        ) if enable_depth else None

        if not self.capL.isOpened():
            raise RuntimeError("Failed to open left camera")
        if enable_depth and (self.capR is None or not self.capR.isOpened()):
            raise RuntimeError("Failed to open right camera")

        print("[CV] Cameras opened")

        self.model = YOLO("yolov8n.pt")  # use n or s for real-time
        self.target_class = target_class

        self.enable_depth = enable_depth
        self.baseline = baseline
        self.focal_length = focal_length

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.publish_rate = publish_rate
        self.last_publish = time.time()
        self.frame_count = 0
        self.start_time = time.time()

    def _estimate_depth(self, bbox, disparity):
        x1, y1, x2, y2 = bbox
        roi = disparity[int(y1):int(y2), int(x1):int(x2)]
        d = np.median(roi)
        if d <= 0:
            return None
        return (self.focal_length * self.baseline) / d

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
        if self.enable_depth:
            retR, frameR = self.capR.read()
            if not retR:
                return None

        img_h, img_w, _ = frameL.shape

        # YOLO inference
        results = self.model(frameL, verbose=False)

        best_box = None
        best_conf = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == self.target_class and conf > best_conf:
                    best_box = box.xyxy[0].cpu().numpy()
                    best_conf = conf

        if best_box is None:
            return None

        x1, y1, x2, y2 = best_box
        cx = (x1 + x2) / 2

        offset_x = (cx - img_w / 2) / (img_w / 2)

        depth = None
        if self.enable_depth and frameR is not None:
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            disparity = self.stereo.compute(grayL, grayR).astype("float32") / 16.0
            depth = self._estimate_depth(best_box, disparity)

        latency_ms = (time.time() - t0) * 1000
        self.frame_count += 1
        fps = self.frame_count / max(time.time() - self.start_time, 1e-6)

        print(
            f"[CV] fps={fps:.1f} "
            f"conf={best_conf:.2f} "
            f"offset_x={offset_x:.2f} "
            f"depth={depth} "
            f"latency={latency_ms:.1f}ms"
        )

        return {
            "offset_x": offset_x,
            "depth_m": depth,
            "confidence": best_conf,
            "latency_ms": latency_ms,
            "fps": fps
        }

    def release(self):
        self.capL.release()
        if self.capR:
            self.capR.release()
