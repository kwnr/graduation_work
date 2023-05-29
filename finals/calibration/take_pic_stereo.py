import cv2
import numpy as np

capL = cv2.VideoCapture(0)
while True:
    if capL.isOpened():
        print("camera1 opened")
        break

capR = cv2.VideoCapture(2)
while True:
    if capR.isOpened():
        print("camera2 opened")
        break


capL.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
capR.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

h = 480
w = 640
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, w)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, w)
capL.set(cv2.CAP_PROP_FPS, 10)
capR.set(cv2.CAP_PROP_FPS, 10)
count = 0
img_left_path = "img_left/"
img_right_path = "img_right/"
while True:
    if capL.isOpened() and capR.isOpened():
        _, imgL = capL.read()
        _, imgR = capR.read()
        imgL = cv2.rotate(imgL, cv2.ROTATE_180)
        imgR = cv2.rotate(imgR, cv2.ROTATE_180)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(
            grayL,
            (8, 5),
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        retR, cornersR = cv2.findChessboardCorners(
            grayR,
            (8, 5),
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        cv2.drawChessboardCorners(imgL, (8, 5), cornersL, retL)
        cv2.drawChessboardCorners(imgR, (8, 5), cornersR, retR)

        img = np.hstack((imgL, imgR))

        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            cv2.imwrite(f"{img_left_path}img_calib_l_{count}.png", imgL)
            cv2.imwrite(f"{img_right_path}img_calib_r_{count}.png", imgR)
            print(count, "image writed")
            count += 1
cv2.destroyAllWindows()
capL.release()
capR.release()
