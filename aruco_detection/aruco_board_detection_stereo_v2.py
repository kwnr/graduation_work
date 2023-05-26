import cv2
import numpy as np
import pickle
from aruco_board_detection_fn import Detector

cap1=cv2.VideoCapture(0)
cap2=cv2.VideoCapture(2)

w=640
h=480
fps=10

det1=Detector(cap1)
det2=Detector(cap2)

det1.set_cap_frame_size(w,h)
det2.set_cap_frame_size(w,h)

det1.set_cap_frame_rate(fps)
det2.set_cap_frame_rate(fps)


while cap1.isOpened():
    rvec1,tvec1,img1=det1.run(draw=True)
    rvec2,tvec2,img2=det2.run(draw=True)
    img=np.hstack((img1,img2))

    imgp1=det1.imgp
    imgp2=det2.imgp
    
        
    
    print(f'{tvec1.T}\n{tvec2.T}\n')
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap1.release()
cap2.release()
cv2.destroyAllWindows()
