import cv2
import numpy as np
from aruco_board_detection_fn import BoardDetection
import time
import matplotlib.pyplot as plt

kf=cv2.KalmanFilter(9,3,0,cv2.CV_32F)
dt=1/30
dt2=1/2*dt**2
kf.transitionMatrix=np.array([
            [1,0,0,dt,0,0,dt2,0,0],
            [0,1,0,0,dt,0,0,dt2,0],
            [0,0,1,0,0,dt,0,0,dt2],
            [0,0,0,1,0,0,dt,0,0],
            [0,0,0,0,1,0,0,dt,0],
            [0,0,0,0,0,1,0,0,dt],
            [0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,1]
        ], dtype=np.float32)
kf.measurementMatrix=np.array([[1,0,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0]], dtype=np.float32)
kf.processNoiseCov=np.eye(kf.processNoiseCov.shape[0],dtype=np.float32)*1e-5
kf.measurementNoiseCov=np.eye(kf.measurementNoiseCov.shape[0],dtype=np.float32)*1e-4
kf.errorCovPost=np.eye(kf.errorCovPost.shape[0],dtype=np.float32)

bd=BoardDetection()

cap=cv2.VideoCapture(1)

ts=[]
measured=[]
filtered=[]
starttime=time.time()
while cap.isOpened():
    _,img=cap.read()
    
    rvec,tvec=bd.pipeline(img)
    if rvec is not None:
        ts.append(time.time()-starttime)
        tvec=np.float32(tvec)
        pred=kf.predict()
        corr=kf.correct(tvec)
        
        measured.append(tvec)
        filtered.append(corr)
        
    cv2.imshow('ing',img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

measured=np.array(measured)
filtered=np.array(filtered)

if len(measured)!=0:
        
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(ts,measured[:,0])
    plt.plot(ts,filtered[:,0])
    plt.legend(['x','x_kf'])

    plt.subplot(3,1,2)
    plt.plot(ts,measured[:,1])
    plt.plot(ts,filtered[:,1])
    plt.legend(['y','y_kf'])

    plt.subplot(3,1,3)
    plt.plot(ts,measured[:,2])
    plt.plot(ts,filtered[:,2])
    plt.legend(['z','z_kf'])

    plt.show()