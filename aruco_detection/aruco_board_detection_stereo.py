import cv2
import numpy as np
import pickle
import time

cap1=cv2.VideoCapture(0,cv2.CAP_V4L2)

while True:
    if cap1.isOpened():
        print("camera1 opened")
        break
cap2=cv2.VideoCapture(2,cv2.CAP_V4L2)
while True:
    if cap2.isOpened():
        print("camera2 opened")
        break

w=640
h=480

cap1.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap2.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))



aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
with open('../camera_matrix.pkl','rb') as f:
    camera_matrix,dist_coeffs=pickle.load(f)
    

detector=cv2.aruco.ArucoDetector(aruco_dict)

board_img=cv2.imread('aruco_board.png')
board=cv2.aruco.GridBoard([2,2],100,20,aruco_dict,ids=np.array([1,11,21,31]))

while cap1.isOpened():
    _,img1=cap1.read()
    _,img2=cap2.read()
    if img1 is not None and img2 is not None:
        img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        img=np.hstack((img1_gray,img2_gray))
        
        img_copy=img.copy()
        
        corners,ids,_=detector.detectMarkers(img)
        #corners,ids,_=cv2.aruco.ArucoDetector.detectMarkers(detector,img)
        if len(corners)>0:
            
            objp,imgp=board.matchImagePoints(corners,ids)
            if objp is not None:
                retval,rvec,tvec=cv2.solvePnP(objp,imgp,camera_matrix,dist_coeffs,flags=cv2.SOLVEPNP_IPPE)
                rvec,tvec=cv2.solvePnPRefineLM(objp,imgp,camera_matrix,dist_coeffs,rvec,tvec)
                pitch,yaw,roll=np.rad2deg(rvec)[:,0]
                cv2.aruco.drawDetectedMarkers(img_copy,corners,ids)
                cv2.drawFrameAxes(img_copy,camera_matrix,dist_coeffs,rvec,tvec,100)
                cv2.putText(img_copy,f'roll: {roll:.2f}, pitch: {pitch:.2f}, yaw: {yaw:.2f}',(10,700),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
                print(tvec)
        cv2.imshow('img',img_copy)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap1.release()
cap2.release()
cv2.destroyAllWindows()
