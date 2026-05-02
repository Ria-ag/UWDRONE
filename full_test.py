import cv2
import time
import numpy as np
from ultralytics import YOLO
import pytesseract

# -------- CAMERA PIPELINE --------
def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ee-mode=1 ee-strength=1.0 "
        f"tnr-mode=1 tnr-strength=1.0 ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

# -------- STREAM PIPELINE --------
def stream_pipeline(width, height, fps, ip, port):
    return (
        f"appsrc ! "
        f"video/x-raw, format=BGR, width={width}, height={height}, framerate={fps}/1 ! "
        f"videoconvert ! "
        f"nvjpegenc quality=90 ! "
        f"rtpjpegpay pt=26 ! "
        f"udpsink host={ip} port={port} sync=false async=false"
    )

print("[TEST] 🚀 Starting FULL vision test...")

# -------- CAMERA --------
cap = cv2.VideoCapture(
    gstreamer_pipeline(),
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

# -------- STREAM --------
STREAM_IP = "192.168.89.250"
STREAM_PORT = 5000

out = cv2.VideoWriter(
    stream_pipeline(1280, 720, 30, STREAM_IP, STREAM_PORT),
    cv2.CAP_GSTREAMER,
    0,
    30,
    (1280, 720),
    True
)

if not out.isOpened():
    print("[TEST] ⚠️ Stream failed, continuing without streaming")
    out = None
else:
    print(f"[TEST] 📡 Streaming to udp://{STREAM_IP}:{STREAM_PORT}")

# -------- YOLO --------
model = YOLO("yolov8n.pt")

# -------- ARUCO --------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dict, cv2.aruco.DetectorParameters()
)

# -------- LOOP --------
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[TEST] Frame grab failed")
        break

    t0 = time.time()

    # ===== YOLO =====
    results = model(frame, verbose=False)
    detections = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            detections += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # ===== ARUCO =====
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    if ids is not None:
        for i, corner in enumerate(corners):
            pts = corner[0].astype(int)
            id_val = int(ids[i][0])

            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            cv2.putText(frame, f"Aruco {id_val}",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)

    # ===== OCR =====
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        y_text = 30
        for line in lines[:3]:
            cv2.putText(frame, f"OCR: {line[:40]}",
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255), 2)
            y_text += 30
    except Exception as e:
        print("[OCR error]", e)

    # ===== STREAM =====
    if out:
        out.write(frame)

    # ===== STATS =====
    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    latency = (time.time() - t0) * 1000

    print(f"[TEST] fps={fps:.1f} det={detections} "
          f"aruco={0 if ids is None else len(ids)} "
          f"latency={latency:.1f}ms")

# -------- CLEANUP --------
cap.release()
if out:
    out.release()