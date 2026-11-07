import cv2
from multi_detect_cv import cv_yolo

cv = cv_yolo(enable_stream=False)
show = True  # set False (or it auto-disables) when running headless over SSH

try:
    while True:
        data = cv.step()
        if data.get("valid"):
            print({k: data[k] for k in ("source", "label", "offset_x", "depth_m")},
                  "aruco:", [m["id"] for m in data["aruco"]])

        # show the annotated frame from step() (reading capL again would
        # steal a frame from the detector)
        if show and cv.last_frame is not None:
            try:
                cv2.imshow("Camera", cv.last_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except cv2.error:
                print("[test] no display available, printing results only")
                show = False
finally:
    cv.release()
    if show:
        cv2.destroyAllWindows()
