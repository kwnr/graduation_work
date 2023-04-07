# https://docs.opencv.org/4.7.0/df/d4a/tutorial_charuco_detection.html 참조

import cv2
from cv2 import aruco
import numpy as np
import pickle

f=open("/Users/hyeokbeom/Desktop/graduation_work/camera_matrix.pkl",'rb')
cameraMatrix,distCoeffs=pickle.load(f)
f.close()

cap=cv2.VideoCapture(1)

aruco_dict=aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# marker=cv2.imread('/Users/hyeokbeom/Desktop/graduation_work/aruco_margin.png')
# marker_corners,marker_id,_=aruco.detectMarkers(marker,aruco_dict)
# marker_corners=np.hstack((marker_corners[0][0],np.zeros((4,1))))

marker_corners=np.array([
    [100,100,0],
    [0,100,0],
    [0,0,0],
    [100,0,0]    
],dtype=np.float32)

criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 30 ,0.001)

while cap.isOpened():
    _,img=cap.read()
    img_copy=img.copy()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    corners,ids,_=aruco.detectMarkers(img,aruco_dict)
    aruco.drawDetectedMarkers(img_copy,corners,ids)
    
    if len(corners)>0:
        for i in range(len(corners)):
            corner=corners[i]
            corner2=cv2.cornerSubPix(gray,corner,(10,10),(-1,-1),criteria)
            retval,rvec,tvec,inliers=cv2.solvePnPRansac(marker_corners,corner2,cameraMatrix,distCoeffs)
            rvec,tvec=cv2.solvePnPRefineLM(marker_corners,corner2,cameraMatrix,distCoeffs,rvec,tvec)
            
            if retval!=0:
                pers_mtrx=cv2.getPerspectiveTransform(corner2,np.array([marker_corners[:,:2]]))
                warp=cv2.warpPerspective(img,pers_mtrx,(100,100))
                cv2.imshow('warp',warp)
                
                cv2.drawFrameAxes(img_copy,cameraMatrix,distCoeffs,rvec,tvec,100)
    
    
                
                
    cv2.imshow("charuco",img_copy)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()