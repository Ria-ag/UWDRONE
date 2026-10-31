import cv2
import socket
import struct
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from rapidocr import EngineType, OCRVersion, RapidOCR


@dataclass
class CameraFrame:
    left: np.ndarray
    right: np.ndarray | None = None


class NetworkCamera:
    def __init__(self, host="0.0.0.0", port=5000):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)

        print("Waiting for camera connection...")
        self.conn, addr = self.server.accept()
        print(f"Connected: {addr}")

    def recv_exact(self, size):
        data = bytearray()
        while len(data) < size:
            chunk = self.conn.recv(size - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return data

    def read(self):
        header = self.recv_exact(4)
        if header is None:
            return False, None

        size = struct.unpack("!I", header)[0]
        data = self.recv_exact(size)
        if data is None:
            return False, None

        frame = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return False, None

        return True, CameraFrame(left=frame)

    def release(self):
        self.conn.close()
        self.server.close()


class LocalCamera:
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        return True, CameraFrame(left=frame)

    def release(self):
        self.cap.release()


class StereoCamera:
    def __init__(self, left_device=0, right_device=1):
        self.left = cv2.VideoCapture(left_device)
        self.right = cv2.VideoCapture(right_device)

    def read(self):
        left_ok, left = self.left.read()
        right_ok, right = self.right.read()

        if not left_ok or not right_ok:
            return False, None

        return True, CameraFrame(left=left, right=right)

    def release(self):
        self.left.release()
        self.right.release()


MODEL_DIR = Path("/home/uwdrones/ocrmodels")

# Choose camera source
camera = NetworkCamera()
# camera = LocalCamera(0)
# camera = StereoCamera(0, 1)

ocr = RapidOCR(
    params={
        "Det.engine_type": EngineType.TENSORRT,
        "Det.model_path": str(MODEL_DIR / "det.onnx"),
        "Det.ocr_version": OCRVersion.PPOCRV6,

        "Rec.engine_type": EngineType.TENSORRT,
        "Rec.model_path": str(MODEL_DIR / "rec.onnx"),
        "Rec.rec_keys_path": str(MODEL_DIR / "ppocrv6_dict.txt"),
        "Rec.ocr_version": OCRVersion.PPOCRV6,

        "EngineConfig.tensorrt.use_fp16": True,
        "EngineConfig.tensorrt.cache_dir": str(MODEL_DIR / "trt_cache"),
    }
)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        result = ocr(frame.left, use_cls=False)

        for text, score in zip(result.txts or [], result.scores or []):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"{timestamp} | {text.upper():<10} | {score:.0%}", flush=True)
finally:
    camera.release()
