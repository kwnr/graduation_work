import cv2
import numpy as np

img=cv2.imread('/Users/hyeokbeom/Desktop/graduation_work/image_detecting/template_A.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img_bin=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

contours,hierarchy=cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

for contour in contours:
    eps=cv2.arcLength(contour,True)
    approx=cv2.approxPolyDP(contour,eps*0.02,True)
    print(len(approx))
    cv2.drawContours(img,[approx],-1,(0,0,255),5)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

        