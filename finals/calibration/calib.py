import numpy as np
import cv2

import glob

objpoints = []
imgpointsL = []
imgpointsR = []
objp = np.zeros((8 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


fnames_l = glob.glob("img_left/*.png")
fnames_r = glob.glob("img_right/*.png")

fnames_l.sort()
fnames_r.sort()

for i in range(len(fnames_l)):
    imgL = cv2.imread(fnames_l[i])
    imgR = cv2.imread(fnames_r[i])
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
    if retL and retR:
        print("!!!!")
        rtL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        rtR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(imgL, (8, 5), None, retL)
        cv2.drawChessboardCorners(imgR, (8, 5), None, retR)
        img = np.hstack((imgL, imgR))
        cv2.imshow(fnames_l[i], img)
        cv2.waitKey(500)
        objpoints.append(objp)
        imgpointsL.append(rtL)
        imgpointsL.append(rtR)


ret, M1, dist1, M2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, grayL[::-1]
)
print(M1, dist1)
print(M2, dist2)
print(R)
print(T)
print(E)
print(F)
