# Runs on the Jetson by itself. Reads frames from the IMX219 CSI camera,
# runs EasyOCR, and prints the text it finds.
# If the camera isn't connected yet, it keeps retrying until it is.
# Stop with Ctrl+C.
import time
import cv2, easyocr

SENSOR_ID = 0   # 0 = left camera, 1 = right camera

# 1280x720 at 60 fps is one of the IMX219's built-in modes
PIPELINE = (
    f"nvarguscamerasrc sensor-id={SENSOR_ID} ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1"   # always hand us the newest frame
)

reader = easyocr.Reader(
    ['en'],
    gpu=True,
    model_storage_directory='/home/uwdrones/.EasyOCR/model',
    user_network_directory='/home/uwdrones/.EasyOCR/user_network',
    download_enabled=False,   # models were copied over manually
)

while True:
    cam = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not cam.isOpened():
        print("Camera not found - retrying in 3s")
        time.sleep(3)
        continue
    print("Camera connected")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Lost the camera - reconnecting")
            break

        start = time.time()
        results = reader.readtext(frame)
        ms = (time.time() - start) * 1000

        if results:
            for box, text, conf in results:
                print(f"{text} ({conf:.2f})")
        else:
            print("(no text)")
        print(f"--- OCR took {ms:.0f} ms")

    cam.release()
    time.sleep(3)
