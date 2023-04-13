import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from aruco_detection.aruco_board_detection_fn import pipeline

cap=cv2.VideoCapture(1)
start_time=time.time()
rolls=[]
pitches=[]
yaws=[]
ts=[]
while cap.isOpened():
    _,img=cap.read()
    t=time.time()-start_time
    
    roll,pitch,yaw=pipeline(img)
    rolls.append(roll)
    pitches.append(pitch)
    yaws.append(yaw)
    ts.append(t)
    if roll is not None:
        plt.plot(ts,rolls,color='r')
        plt.plot(ts,pitches,color='b')
        plt.plot(ts,yaws)
        plt.legend(['roll','pitch','qyaw'])
        plt.pause(0.05)
        
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()