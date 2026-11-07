import argparse
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from aruco_utils import ArucoDetector, draw_markers

try:
    import pytesseract
except ImportError:  # OCR is optional; don't take down the whole CV node
    pytesseract = None


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


def stream_pipeline(width, height, fps, ip, port, encoder):
    return (
        "appsrc ! "
        "video/x-raw, format=BGR, width=%d, height=%d, framerate=%d/1 ! "
        "videoconvert ! %s quality=90 ! "
        "rtpjpegpay pt=26 ! "
        "udpsink host=%s port=%d sync=false async=false"
        % (width, height, fps, encoder, ip, port)
    )


def gstreamer_available():
    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer" in line:
            return "YES" in line
    return False


def open_camera(source, width, height, fps, sensor_id=0):
    """source=None -> Jetson CSI camera via GStreamer.
    source=int -> USB webcam index (laptop testing). source=str -> video/image file."""
    if source is None:
        cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id, width, height, fps), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            hint = "" if gstreamer_available() else (
                " This OpenCV build has NO GStreamer support (often caused by "
                "`pip install opencv-python` overriding JetPack's OpenCV). "
                "Run: pip uninstall opencv-python opencv-contrib-python")
            raise RuntimeError(f"Camera failed to open (CSI sensor {sensor_id}).{hint}")
        return cap
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Camera/source failed to open: {source!r}")
    return cap


class cv_yolo:
    """YOLO + ArUco + OCR detector.

    step() always returns a dict:
        valid       True if a target was found this frame
        offset_x    -1 (left edge) .. +1 (right edge) of the chosen target
        offset_y    -1 (top) .. +1 (bottom)
        depth_m     distance estimate. For YOLO targets this is a SIZE PROXY
                    (15000 / box area in px), not meters. For ArUco targets it is
                    real meters when marker_size_m and focal_px are set.
        source      "yolo" or "aruco"
        label, confidence, area
        (YOLO conf_thresh defaults to 0.6: at 0.45 a hand was tagged "person" at 0.59.)
        aruco       list of every marker seen: {id, cx, cy, offset_x, offset_y, side_px, distance_m}
        detections  list of every YOLO box over threshold
        ocr         cached OCR lines
        fps, latency_ms
    """

    def __init__(self, width=1280, height=720, fps=30,
                 stream_ip="192.168.89.250", stream_port=5000,
                 enable_stream=True, enable_ocr=True,
                 source=None, sensor_id=0,
                 model_path="yolov8n.pt", conf_thresh=0.6, target_classes=None,
                 target_mode="yolo", target_aruco_id=None,
                 aruco_dict="DICT_4X4_50", aruco_ids=None, aruco_min_side_px=30,
                 marker_size_m=None, focal_px=None,
                 ocr_every=10, verbose=True):

        print("[CV] Initializing camera...")
        self.capL = open_camera(source, width, height, fps, sensor_id)
        self.verbose = verbose

        # YOLO
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.conf_thresh = conf_thresh
        self.target_classes = set(target_classes) if target_classes is not None else None

        # ARUCO
        self.aruco = ArucoDetector(aruco_dict, allowed_ids=aruco_ids, min_side_px=aruco_min_side_px)
        self.marker_size_m = marker_size_m
        self.focal_px = focal_px

        if target_mode not in ("yolo", "aruco", "any"):
            raise ValueError("target_mode must be 'yolo', 'aruco' or 'any'")
        self.target_mode = target_mode
        self.target_aruco_id = target_aruco_id

        # Streaming (hardware JPEG first, software JPEG as fallback)
        self.out = None
        if enable_stream:
            for enc in ("nvjpegenc", "jpegenc"):
                out = cv2.VideoWriter(stream_pipeline(width, height, fps, stream_ip, stream_port, enc),
                                      cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
                if out.isOpened():
                    self.out = out
                    print(f"[CV] Streaming to udp://{stream_ip}:{stream_port} ({enc})")
                    break
            if self.out is None:
                print("[CV] Stream failed, continuing without streaming")
        self.stream_size = (width, height)

        # OCR
        self.enable_ocr = enable_ocr and pytesseract is not None
        if enable_ocr and pytesseract is None:
            print("[CV] pytesseract not installed, OCR disabled")
        self.ocr_every = max(1, int(ocr_every))
        self.ocr_cache = []

        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame = None  # annotated frame from the most recent step()

        # background thread state (see start()/latest())
        self._lock = threading.Lock()
        self._latest = None
        self._latest_t = 0.0
        self._thread = None
        self._running = False

        print("[CV] Ready")

    # ---------------------------------------------------------------- helpers
    def _run_ocr(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        return [l.strip() for l in text.split("\n") if l.strip()][:3]

    def _run_yolo(self, frame, w, h):
        dets = []
        for r in self.model(frame, verbose=False):
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_thresh:
                    continue
                cls = int(box.cls[0])
                if self.target_classes is not None and cls not in self.target_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                dets.append({
                    "label": self.class_names[cls], "confidence": conf,
                    "box": (x1, y1, x2, y2), "cx": cx, "cy": cy, "area": area,
                    "offset_x": (cx - w / 2) / (w / 2),
                    "offset_y": (cy - h / 2) / (h / 2),
                    "depth_m": 15000 / max(area, 1),  # crude size proxy: larger = closer
                })
        return dets

    def _run_aruco(self, frame, w, h):
        markers = self.aruco.detect(frame)
        for m in markers:
            m["offset_x"] = (m["cx"] - w / 2) / (w / 2)
            m["offset_y"] = (m["cy"] - h / 2) / (h / 2)
            m["area"] = m["side_px"] ** 2
            if self.marker_size_m and self.focal_px:
                m["distance_m"] = self.focal_px * self.marker_size_m / m["side_px"]
            else:
                m["distance_m"] = None
        return markers

    def _pick_target(self, dets, markers):
        def from_marker():
            pool = markers
            if self.target_aruco_id is not None:
                pool = [m for m in markers if m["id"] == self.target_aruco_id]
            if not pool:
                return None
            m = max(pool, key=lambda m: m["area"])
            depth = m["distance_m"] if m["distance_m"] is not None else 15000 / max(m["area"], 1)
            return {"source": "aruco", "label": f"aruco_{m['id']}", "confidence": 1.0,
                    "offset_x": m["offset_x"], "offset_y": m["offset_y"],
                    "depth_m": depth, "area": m["area"]}

        def from_yolo():
            if not dets:
                return None
            d = max(dets, key=lambda d: d["area"])  # largest box = closest
            return {"source": "yolo", "label": d["label"], "confidence": d["confidence"],
                    "offset_x": d["offset_x"], "offset_y": d["offset_y"],
                    "depth_m": d["depth_m"], "area": d["area"]}

        if self.target_mode == "yolo":
            return from_yolo()
        if self.target_mode == "aruco":
            return from_marker()
        return from_marker() or from_yolo()

    # ---------------------------------------------------------------- main
    def step(self):
        ret, frame = self.capL.read()
        if not ret:
            return {"valid": False, "error": "no_frame"}

        t0 = time.time()
        h, w = frame.shape[:2]

        # Detect on the clean frame; draw on a copy so boxes/text can't
        # corrupt ArUco markers or OCR.
        dets = self._run_yolo(frame, w, h)
        markers = self._run_aruco(frame, w, h)

        if self.enable_ocr and self.frame_count % self.ocr_every == 0:
            try:
                self.ocr_cache = self._run_ocr(frame)
            except Exception as e:
                if self.verbose:
                    print("[CV] OCR error:", e)
                self.ocr_cache = []

        target = self._pick_target(dets, markers)

        # ===== VISUALIZATION =====
        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{d['label']} {d['confidence']:.2f}", (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        draw_markers(vis, markers)
        y = 30
        for line in self.ocr_cache:
            cv2.putText(vis, f"OCR: {line[:40]}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            y += 30
        self.last_frame = vis

        # ===== STREAM =====
        if self.out:
            if (w, h) != self.stream_size:
                vis = cv2.resize(vis, self.stream_size)
            self.out.write(vis)

        # ===== STATS =====
        self.frame_count += 1
        fps = self.frame_count / max(time.time() - self.start_time, 1e-6)
        latency = (time.time() - t0) * 1000
        if self.verbose:
            print(f"[CV] fps={fps:.1f} latency={latency:.1f}ms "
                  f"yolo={len(dets)} aruco={[m['id'] for m in markers]}")

        aruco_out = [{k: m[k] for k in ("id", "cx", "cy", "offset_x", "offset_y", "side_px", "distance_m")}
                     for m in markers]
        det_out = [{k: d[k] for k in ("label", "confidence", "offset_x", "offset_y", "area")} for d in dets]

        result = {"valid": target is not None, "aruco": aruco_out, "detections": det_out,
                  "ocr": list(self.ocr_cache), "fps": fps, "latency_ms": latency}
        if target:
            result.update(target)
        return result

    # ------------------------------------------------------ background mode
    def start(self):
        """Run step() continuously in a background thread so a slow YOLO frame
        never stalls the MAVLink loop (PX4 drops OFFBOARD if setpoints stop)."""
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self):
        while self._running:
            try:
                out = self.step()
            except Exception as e:
                print("[CV] step error:", e)
                out = {"valid": False, "error": str(e)}
                time.sleep(0.05)
            with self._lock:
                self._latest, self._latest_t = out, time.time()

    def latest(self, max_age=0.5):
        """Most recent result, or None if it is older than max_age seconds."""
        with self._lock:
            if self._latest is None or time.time() - self._latest_t > max_age:
                return None
            return self._latest

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.capL.release()
        if self.out:
            self.out.release()


MultiDetectCV = cv_yolo  # clearer alias; cv_yolo kept so existing imports work


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Standalone YOLO + ArUco test")
    ap.add_argument("--source", default=None,
                    help="omit for Jetson CSI cam; webcam index (0) or a video/image path")
    ap.add_argument("--mode", default="any", choices=["yolo", "aruco", "any"])
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = forever)")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    ap.add_argument("--save", default=None, help="save the last annotated frame here")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--conf", type=float, default=0.6, help="YOLO confidence threshold")
    ap.add_argument("--classes", default=None,
                    help="comma-separated YOLO class ids to keep, e.g. 0 for person only")
    args = ap.parse_args()
    src = int(args.source) if args.source is not None and args.source.isdigit() else args.source

    classes = [int(c) for c in args.classes.split(",")] if args.classes else None
    cv = cv_yolo(source=src, target_mode=args.mode, enable_stream=args.stream,
                 enable_ocr=not args.no_ocr, conf_thresh=args.conf, target_classes=classes)
    n = 0
    try:
        while True:
            out = cv.step()
            if out.get("error") == "no_frame":
                print("[CV] no frame, stopping")
                break
            summary = {k: out[k] for k in ("valid", "source", "label", "offset_x", "depth_m") if k in out}
            print("   ->", summary, "| aruco ids:", [m["id"] for m in out["aruco"]])
            if args.show and cv.last_frame is not None:
                cv2.imshow("multi_detect_cv", cv.last_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            n += 1
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.save and cv.last_frame is not None:
            cv2.imwrite(args.save, cv.last_frame)
            print(f"[CV] saved {args.save}")
        cv.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # headless OpenCV build
