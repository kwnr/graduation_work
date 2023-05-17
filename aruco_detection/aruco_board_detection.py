import cv2
import numpy as np
import pickle

cap=cv2.VideoCapture(0)
aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
with open('../camera_matrix.pkl','rb') as f:
    camera_matrix,dist_coeffs=pickle.load(f)
    

detector=cv2.aruco.ArucoDetector(aruco_dict)

board_img=cv2.imread('../aruco_board.png')
board=cv2.aruco.GridBoard([2,2],100,20,aruco_dict,ids=np.array([1,11,21,31]))

while cap.isOpened():
    _,img=cap.read()
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
    
cap.release()
cv2.destroyAllWindows()
