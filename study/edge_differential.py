import cv2
import numpy as np

cap=cv2.VideoCapture(1)

while cap.isOpened():
    _,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gx_kernel=np.array([[-1,1]])
    gy_kernel=np.array([[-1],[1]])
    
    edge_gx=cv2.filter2D(img,-1,gx_kernel)
    edge_gy=cv2.filter2D(img,-1,gy_kernel)
    
    merged=np.vstack((img,edge_gx,edge_gy))
    cv2.imshow('img',merged)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()