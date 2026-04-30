from multi_detect import cv_yolo
import cv2

cv = cv_yolo(enable_depth=False)  # keep it simple first

while True:
    data = cv.step()
    if data:
        print(data)

    # show camera feed
    ret, frame = cv.capL.read()
    if ret:
        cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv.release()
cv2.destroyAllWindows()