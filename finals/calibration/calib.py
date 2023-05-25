import numpy as np
import cv2
from aruco_board_detection_fn import Detector

objpoints=[]
imgpoints=[]

aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector=cv2.aruco.ArucoDetector(aruco_dict)
board=cv2.aruco.GridBoard([2,2],100,20,aruco_dict,ids=np.array([1,11,21,31]))

for fname in fnames:
    img=cv2.imread(fname)
    corners,ids,_=detector.detectMarkers(img)
    objp,imgp=board.matchImagePoints(corners,ids)
    if objp is not None:
        objpoints.append(objp)
        imgpoints.append(imgp)
    objp=None
    imgp=None
    
ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objpoints,imgpoints,img.shape[::-1],None,None)
print(mtx,dist)