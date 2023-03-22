'''
카메라 영상에서 12각형에 해당하는 컨투어를 검출하고,
해당 컨투어가 템플릿과 얼마나 다른지 계산, 영상에 표시
'''
import numpy as np
import cv2
import utility

cap=cv2.VideoCapture(1)
utility.set_cap_resolution(cap,234)
target=cv2.imread('image_detecting/template.png',cv2.IMREAD_GRAYSCALE)

while cap.isOpened():
    _,img=cap.read()
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    _,img_bin=cv2.threshold(img_gray,-1,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    _,target_bin=cv2.threshold(target,122,255,cv2.THRESH_BINARY_INV)
    
    img_contours,_=cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    target_contour,_=cv2.findContours(target_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    for i in range(len(img_contours)):
        contour=img_contours[i]
        eps=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,eps*0.02,True)
        if len(approx)==12:
            match=cv2.matchShapes(contour,target_contour[0],cv2.CONTOURS_MATCH_I3,0)
            if match<0.1:
                cv2.putText(img,f'{match}',tuple(contour[0][0]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    