import cv2
import numpy as np
import pickle

with open('data.pkl','rb') as f:
    data=pickle.load(f)
    
cap1=cv2.VideoCapture(0)

while True:
    _,img=cap1.read()
    img=cv2.rotate(img,cv2.ROTATE_180)
    img_und=cv2.undistort(img,data['M1'],data['dist1'])
    hs=np.hstack((img,img_und))
    cv2.imshow('img',hs)

    if cv2.waitKey(1)&0xff==ord('q'):
        break
    
