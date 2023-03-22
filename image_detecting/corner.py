import cv2
import numpy as np

cam=cv2.VideoCapture(1)

while cam.isOpened():
    status, img=cam.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)
    for i in corners:
        cv2.circle(img, (int(i[0][0]),int(i[0][1]),),3,(0,0,255),2)
    
    cv2.imshow('img',img)
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break    
