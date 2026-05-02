import cv2
aruco = cv2.aruco
dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for i in range(5):
    img = aruco.generateImageMarker(dict, i, 300)
    cv2.imwrite(f"aruco_{i}.png", img)