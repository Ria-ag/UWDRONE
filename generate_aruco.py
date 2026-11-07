import cv2
from aruco_utils import generate_marker

# The white border ("quiet zone") is required for detection: a marker that
# touches a dark edge (screen bezel, dark paper, cropped printout) won't be found.
BORDER = 50

for i in range(5):
    m = generate_marker(i, 300)
    m = cv2.copyMakeBorder(m, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(f"aruco_{i}.png", m)
