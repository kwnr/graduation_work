# https://docs.opencv.org/4.7.0/df/d4a/tutorial_charuco_detection.html 참조

import cv2
from cv2 import aruco
import numpy as np
import pickle

f=open("/Users/hyeokbeom/Desktop/graduation_work/camera_matrix.pkl",'rb')
cameraMatrix,distCoeffs=pickle.load(f)
f.close()

cap=cv2.VideoCapture(1)

aruco_dict=aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ids=np.array([1,11,21,31])

board=aruco.CharucoBoard([3,3],0.4,0.2,aruco_dict,ids=ids)

board_img=cv2.imread('/Users/hyeokbeom/Desktop/graduation_work/charuco.png')
arucoDetector=aruco.ArucoDetector(aruco_dict)
board_corners,board_ids,_=aruco.ArucoDetector.detectMarkers(arucoDetector,board_img)
_,board_corners,_=aruco.interpolateCornersCharuco(board_corners,board_ids,board_img,board,cameraMatrix,distCoeffs)
objp=np.hstack((board_corners[:,0],np.zeros((board_corners.shape[0],1))))


while cap.isOpened():
    _,img=cap.read()
    img_copy=img.copy()
    
    
    markerCorners,markerIds, rejectedImgPoints=aruco.ArucoDetector.detectMarkers(arucoDetector,img)
    
    if markerIds is not None:
        aruco.drawDetectedMarkers(img_copy,markerCorners,markerIds)
        retval, charucoCorners, charucoIds=aruco.interpolateCornersCharuco(markerCorners,markerIds,img,board,cameraMatrix,distCoeffs)
        if charucoCorners is not None:
            color=(255,0,0)
            aruco.drawDetectedCornersCharuco(img_copy,charucoCorners,charucoIds,color)
            rvec=np.zeros((1,3))
            tvec=np.zeros((1,3))
            retval,rvec,tvec,_=cv2.solvePnPRansac(np.float32(objp),np.float32(charucoCorners),cameraMatrix,distCoeffs)
            if retval:
                cv2.drawFrameAxes(img_copy,cameraMatrix,distCoeffs,rvec,tvec,0.5)
                
                
    cv2.imshow("charuco",img_copy)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()