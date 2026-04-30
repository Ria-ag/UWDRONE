import cv2
import time
import numpy as np
from ultralytics import YOLO
from pyzbar.pyzbar import decode as decode_barcodes
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
    def __init__(
        self,
        camera_left=0,
        camera_right=1,
        width=1280,
        height=720,
        fps=30,
        publish_rate=10.0,
        target_class=None,          # None = no tracking target filter
        enable_depth=False,         # start simpler unless stereo is stable
        baseline=0.12,
        focal_length=700.0,
        stream_ip="192.168.89.250",
        stream_port=5000,
        enable_stream=True,
        enable_ocr=True,
        enable_barcodes=True,
        ocr_every_n_frames=10,
        conf_threshold=0.4
    ):
        print("[CV] Initializing cameras...")

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

        # YOLO
        self.model = YOLO("yolov8n.pt")
        self.class_names = self.model.names
        self.target_class = target_class
        self.conf_threshold = conf_threshold

        # Depth
        self.enable_depth = enable_depth
        self.baseline = baseline
        self.focal_length = focal_length

        if self.enable_depth:
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=128,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

        # Streaming
        self.enable_stream = enable_stream
        self.out = None
        if enable_stream:
            print("[CV] Initializing video stream...")
            # Streaming
            self.enable_stream = enable_stream
            self.out = None
            if enable_stream:
                print("[CV] Initializing video stream...")

                stream_pipeline = (
                    "appsrc ! "
                    "video/x-raw, format=BGR, width=%d, height=%d, framerate=%d/1 ! "
                    "videoconvert ! "
                    "nvjpegenc quality=90 ! "
                    "rtpjpegpay ! "
                    "udpsink host=%s port=%d sync=false async=false"
                    % (width, height, fps, stream_ip, stream_port)
                )

                self.out = cv2.VideoWriter(
                    stream_pipeline,
                    cv2.CAP_GSTREAMER,
                    0,
                    fps,
                    (width, height),
                    True
                )

                if not self.out.isOpened():
                    print("[CV] Warning: Failed to open video stream")
                    self.out = None
                else:
                    print(f"[CV] Streaming to udp://{stream_ip}:{stream_port}")

        # OCR / Barcode
        self.enable_ocr = enable_ocr
        self.enable_barcodes = enable_barcodes
        self.ocr_every_n_frames = ocr_every_n_frames
        self.ocr_text_cache = []
        self.barcode_cache = []

        # Timing
        self.publish_rate = publish_rate
        self.last_publish = time.time()
        self.frame_count = 0
        self.start_time = time.time()

        print("[CV] Initialization complete")

    def _estimate_depth(self, bbox, disparity):
        x1, y1, x2, y2 = bbox
        roi = disparity[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return None
        d = np.median(roi)
        if d <= 0:
            return None
        return (self.focal_length * self.baseline) / d

    def _run_ocr(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(thresh, config="--psm 6")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines[:5]  # keep top few lines

    def _run_barcodes(self, frame):
        decoded = decode_barcodes(frame)
        results = []

        for obj in decoded:
            x, y, w, h = obj.rect
            data = obj.data.decode("utf-8", errors="ignore")
            btype = obj.type
            results.append({
                "type": btype,
                "data": data,
                "rect": (x, y, w, h)
            })

        return results

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

        # ---------- YOLO ----------
        results = self.model(frameL, verbose=False)

        detections = []
        tracked_detection = None
        best_conf = 0
        offset_x = 0
        depth = None

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                label = self.class_names.get(cls, str(cls))

                detections.append({
                    "class_id": cls,
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

                # Draw detection
                cv2.rectangle(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frameL,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # Track ONE target class for control if desired
                if self.target_class is not None and cls == self.target_class and conf > best_conf:
                    tracked_detection = [x1, y1, x2, y2]
                    best_conf = conf

        # ---------- Target tracking output ----------
        if tracked_detection is not None:
            x1, y1, x2, y2 = tracked_detection
            cx = (x1 + x2) / 2
            offset_x = (cx - img_w / 2) / (img_w / 2)

            # Highlight tracked target
            cv2.rectangle(frameL, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(
                frameL,
                "TRACK TARGET",
                (x1, min(y2 + 25, img_h - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

            if self.enable_depth and frameR is not None:
                grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
                disparity = self.stereo.compute(grayL, grayR).astype("float32") / 16.0
                depth = self._estimate_depth(tracked_detection, disparity)

        # ---------- OCR ----------
        if self.enable_ocr and (self.frame_count % self.ocr_every_n_frames == 0):
            try:
                self.ocr_text_cache = self._run_ocr(frameL)
            except Exception as e:
                print(f"[CV] OCR error: {e}")
                self.ocr_text_cache = []

        # ---------- Barcodes / QR ----------
        if self.enable_barcodes:
            try:
                self.barcode_cache = self._run_barcodes(frameL)
            except Exception as e:
                print(f"[CV] Barcode error: {e}")
                self.barcode_cache = []

        # Draw barcode boxes
        for bc in self.barcode_cache:
            x, y, w, h = bc["rect"]
            cv2.rectangle(frameL, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(
                frameL,
                f'{bc["type"]}: {bc["data"][:20]}',
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

        # Draw OCR text overlay
        y_text = 30
        for line in self.ocr_text_cache[:3]:
            cv2.putText(
                frameL,
                f"OCR: {line[:50]}",
                (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2
            )
            y_text += 28

        # ---------- Stream ----------
        if self.out is not None:
            self.out.write(frameL)

        latency_ms = (time.time() - t0) * 1000
        self.frame_count += 1
        fps = self.frame_count / max(time.time() - self.start_time, 1e-6)

        print(
            f"[CV] fps={fps:.1f} "
            f"detections={len(detections)} "
            f"track_conf={best_conf:.2f} "
            f"offset_x={offset_x:.2f} "
            f"depth={depth} "
            f"barcodes={len(self.barcode_cache)} "
            f"ocr_lines={len(self.ocr_text_cache)} "
            f"latency={latency_ms:.1f}ms"
        )

        return {
            "offset_x": offset_x,
            "depth_m": depth,
            "confidence": best_conf,
            "latency_ms": latency_ms,
            "fps": fps,
            "detections": detections,
            "ocr_text": self.ocr_text_cache,
            "barcodes": self.barcode_cache
        }

    def release(self):
        print("[CV] Releasing resources...")
        self.capL.release()
        if self.capR:
            self.capR.release()
        if self.out:
            self.out.release()