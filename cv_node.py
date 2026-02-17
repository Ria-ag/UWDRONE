cat << 'EOF' > cv_node.py
import cv2
import time
import numpy as np
import pytesseract

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class cv_node:
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
        print("[CV] Initializing cameras...")

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
        if not self.capR.isOpened():
            raise RuntimeError("Failed to open right camera")
        
        if self.capL:
            print("[CV] Left camera opened")
        if self.capR:
            print("[CV] Right camera opened")

        self.detection_type = detection_type
        self.publish_rate = publish_rate
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

        self.last_publish = time.time()
        self.frame_count = 0
        self.start_time = time.time()

    def _purple_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_purple = (125, 80, 50)
        upper_purple = (160, 255, 255)
        return cv2.inRange(hsv, lower_purple, upper_purple)

    def _classify_shape(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 3:
            return "triangle"
        elif len(approx) == 4:
            return "quadrilateral"
        elif len(approx) > 6:
            return "circle"
        return "unknown"

    def _recognize_text(self, frame, bbox):
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(gray, config=config)
        return text.strip()

    def _estimate_depth(self, bbox, disparity):
        x, y, w, h = bbox
        roi = disparity[y:y+h, x:x+w]
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
        if self.enable_depth and self.capR:
            retR, frameR = self.capR.read()
            if not retR:
                return None

        mask = self._purple_mask(frameL)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 500:
            return None

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w / 2
        cy = y + h / 2

        img_h, img_w, _ = frameL.shape
        offset_x = (cx - img_w / 2) / (img_w / 2)

        shape = None
        text = None
        depth = None

        if self.detection_type in ["shape", "shape+text"]:
            shape = self._classify_shape(c)

        if self.detection_type == "shape+text":
            text = self._recognize_text(frameL, (x, y, w, h))

        if self.enable_depth and frameR is not None:
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            disparity = self.stereo.compute(grayL, grayR).astype("float32") / 16.0
            depth = self._estimate_depth((x, y, w, h), disparity)

        confidence = min(1.0, area / 5000.0)

        latency_ms = (time.time() - t0) * 1000
        self.frame_count += 1
        fps = self.frame_count / max(time.time() - self.start_time, 1e-6)

        print(
            f"[CV] fps={fps:.1f} "
            f"offset_x={offset_x:.2f} "
            f"depth={depth} "
            f"latency={latency_ms:.1f}ms"
        )

        return {
            "offset_x": offset_x,
            "shape": shape,
            "text": text,
            "depth_m": depth,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "fps": fps
        }

    def release(self):
        self.capL.release()
        if self.capR:
            self.capR.release()
EOF