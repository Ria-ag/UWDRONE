import cv2

gst_pipeline = (
    "nvarguscamerasrc ee-mode=1 ee-strength=1.0 tnr-mode=1 tnr-strength=1.0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=True"
)

# Initialize the capture
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open the GStreamer pipeline.")
    exit()

print("OpenCV Pipeline Started. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Empty frame received.")
            break

        # --- PROCESS WITH OPENCV ---
        h, w = frame.shape[:2]

        # add text to image
        cv2.putText(frame, f"Jetson Orin - 720p Live", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite("opencv_test.jpg", frame)
        print("Success! Saved opencv_test.jpg")
        break

finally:
    cap.release()