import cv2
import numpy as np
from aruco_board_detection_fn import BoardDetection
import time
import matplotlib.pyplot as plt

kf=cv2.KalmanFilter(9,3,0)
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
        ])
kf.measurementMatrix=np.array([[1,0,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0]],dtype=np.float32)
kf.processNoiseCov=np.eye(kf.processNoiseCov.shape[0])*1e-5
kf.measurementNoiseCov=np.eye(kf.measurementNoiseCov.shape[0])*1e-4
kf.errorCovPost=np.eye(kf.errorCovPost.shape[0])

bd=BoardDetection()

cap=cv2.VideoCapture(1)

ts=[]
measured=[]
filtered=[]
starttime=time.time()
while cap.isOpened():
    _,img=cap.imread()
    
    ts.append(starttime-time.time)
    rvec,tvec=bd.pipeline(img)
    pred=kf.predict()
    corr=kf.correct(rvec)
    
    measured.append(rvec)
    filtered.append(corr)
    
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(ts,measured[:,0])
plt.plot(ts,filtered[:,0])
plt.legend(['x','x_kf'])

plt.subplot(3,1,2)
plt.plot(ts,measured[:,1])
plt.plot(ts,filtered[:,1])
plt.legend(['y','y_kf'])

plt.subplot(3,1,1)
plt.plot(ts,measured[:,2])
plt.plot(ts,filtered[:,2])
plt.legend(['z','z_kf'])
