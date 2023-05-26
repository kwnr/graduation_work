import cv2
import numpy as np
import pickle
from aruco_board_detection_fn import Detector

capL=cv2.VideoCapture(0)
capR=cv2.VideoCapture(2)

w=640
h=480
fps=10

detL=Detector(capL)
detR=Detector(capR)

detL.set_cap_frame_size(w,h)
detR.set_cap_frame_size(w,h)

detL.set_cap_frame_rate(fps)
detR.set_cap_frame_rate(fps)


while capL.isOpened():
    rvec1,tvec1,img1=detL.run(draw=True)
    rvec2,tvec2,img2=detR.run(draw=True)
    img=np.hstack((img1,img2))

    imgp1=detL.imgp
    imgp2=detR.imgp
    
        
    
    
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap1.release()
cap2.release()
cv2.destroyAllWindows()
