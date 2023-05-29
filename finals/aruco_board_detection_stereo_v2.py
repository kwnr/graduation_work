import cv2
import numpy as np
import pickle
from aruco_board_detection_fn import Detector

M1 = np.array(
    [
        [465.24217997, 0.0, 341.98704948],
        [0.0, 458.21944784, 215.39055162],
        [0.0, 0.0, 1.0],
    ]
)
M2 = np.array(
    [
        [465.24217997, 0.0, 341.98704948],
        [0.0, 458.21944784, 215.39055162],
        [0.0, 0.0, 1.0],
    ]
)
dist1 = np.array(
    [[-0.01575441, -0.16604411, -0.00668898, 0.01829878, 0.1655708]]
)
dist2 = np.array(
    [[-0.01575441, -0.16604411, -0.00668898, 0.01829878, 0.1655708]]
)

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)

w = 640
h = 480
fps = 10


detL = Detector(capL)
detR = Detector(capR)

detL.camera_matrix=M1
detR.camera_matrix=M1
detL.dist_coeffs=dist1
detR.dist_coeffs=dist2

detL.set_cap_frame_size(w, h)
detR.set_cap_frame_size(w, h)

detL.set_cap_frame_rate(fps)
detR.set_cap_frame_rate(fps)


while capL.isOpened():
    rvec1, tvec1, img1 = detL.run(draw=True)
    rvec2, tvec2, img2 = detR.run(draw=True)

    img=np.hstack((img1,img2))

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
