import cv2
import time
import numpy as np
from ultralytics import YOLO
import pytesseract

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, framerate=30):
    return (
        "nvarguscamerasrc sensor-id=%d ee-mode=1 ee-strength=1.0 "
        "tnr-mode=1 tnr-strength=1.0 ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
        % (sensor_id, capture_width, capture_height, framerate)
    )


class cv_yolo:
    def __init__(self, width=1280, height=720, fps=30,
                 stream_ip="192.168.89.250", stream_port=5000,
                 enable_stream=True, enable_ocr=True):

        print("[CV] Initializing camera...")

        self.capL = cv2.VideoCapture(
            gstreamer_pipeline(0, width, height, fps),
            cv2.CAP_GSTREAMER
        )

        if not self.capL.isOpened():
            raise RuntimeError("Camera failed")

        # YOLO
        self.model = YOLO("yolov8n.pt")
        self.class_names = self.model.names

        # ARUCO
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Streaming
        self.out = None
        if enable_stream:
            pipeline = (
                "appsrc ! "
                "video/x-raw, format=BGR, width=%d, height=%d, framerate=%d/1 ! "
                "videoconvert ! "
                "nvjpegenc quality=90 ! "
                "rtpjpegpay pt=26 ! "   # ✅ IMPORTANT
                "udpsink host=%s port=%d sync=false async=false"
                % (width, height, fps, stream_ip, stream_port)
            )

            self.out = cv2.VideoWriter(
                pipeline,
                cv2.CAP_GSTREAMER,
                0,
                fps,
                (width, height),
                True
            )

            if self.out.isOpened():
                print(f"[CV] Streaming to udp://{stream_ip}:{stream_port}")
            else:
                print("[CV] ⚠️ Stream failed")
                self.out = None

        # OCR
        self.enable_ocr = enable_ocr
        self.ocr_cache = []
        self.frame_count = 0
        self.start_time = time.time()

        print("[CV] Ready")

    def _run_ocr(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        return [l.strip() for l in text.split("\n") if l.strip()][:3]

    def step(self):
        ret, frame = self.capL.read()
        if not ret:
            return None

        t0 = time.time()

        h, w = frame.shape[:2]

        best_target = None
        best_area = 0

        # ===== YOLO =====
        results = self.model(frame, verbose=False)

        for r in results:
            for box in r.boxes:

                conf = float(box.conf[0])
                if conf < 0.45:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                cls = int(box.cls[0])
                label = self.class_names[cls]

                area = (x2 - x1) * (y2 - y1)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # normalized horizontal offset
                offset_x = (cx - (w / 2)) / (w / 2)

                # crude pseudo-depth estimate
                # larger object = closer
                depth_est = 15000 / max(area, 1)

                # pick largest target
                if area > best_area:
                    best_area = area

                    best_target = {
                        "label": label,
                        "confidence": conf,
                        "offset_x": offset_x,
                        "depth_m": depth_est,
                        "cx": cx,
                        "cy": cy,
                        "area": area
                    }

                # VISUALIZATION
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                            (0,255,0), 2)

                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,0),
                            2)

        # ===== STREAM =====
        if self.out:
            self.out.write(frame)

        # ===== STATS =====
        self.frame_count += 1
        fps = self.frame_count / (time.time() - self.start_time)
        latency = (time.time() - t0) * 1000

        print(f"[CV] fps={fps:.1f} latency={latency:.1f}ms")

        # ===== OUTPUT =====
        if best_target:
            return {
                "valid": True,
                "offset_x": best_target["offset_x"],
                "depth_m": best_target["depth_m"],
                "label": best_target["label"],
                "confidence": best_target["confidence"]
            }

        return {
            "valid": False
        }

    def release(self):
        self.capL.release()
        if self.out:
            self.out.release()