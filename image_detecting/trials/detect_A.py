import cv2
import numpy as np

cap=cv2.VideoCapture(1)

while cap.isOpened():
    _,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img_bin=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
    
    contours,hierarchy=cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    inner_indice=np.nonzero(hierarchy[0][:,3].copy()+1)
    outer_indice=np.nonzero(hierarchy[0][:,2].copy()+1)
    

    for outer_index in outer_indice[0]:
        outer_contour=contours[outer_index]
        outer_eps=cv2.arcLength(outer_contour,True)
        outer_approx=cv2.approxPolyDP(outer_contour,outer_eps*0.01,True)
        if len(outer_approx)==8:
            inner_index=hierarchy[0,outer_index,2]
            inner_contour=contours[inner_index]
            inner_eps=cv2.arcLength(inner_contour,True)
            inner_approx=cv2.approxPolyDP(inner_contour,inner_eps*0.01,True)
            if len(inner_approx)==3:
                cv2.drawContours(img,[outer_approx],-1,(0,0,225),5)
                cv2.drawContours(img,[inner_approx],-1,(255,0,0),5)
                
    merge=np.hstack((img_gray,img_bin))
    cv2.imshow('merge',merge)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()