import cv2
import numpy as np
import time

def setLabel(img, pts, label):
    # 사각형 좌표 받아오기
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    
    
cap=cv2.VideoCapture(1)
frame_rate=10
prev=0

w=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

canv=np.zeros([w,h])

while cap.isOpened():
    time_elapsed = time.time() - prev   #10FPS로 제한
    res, image = cap.read()             #

    if time_elapsed > 1./frame_rate:
        prev = time.time()

    status, img=cap.read()
    
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gauss=cv2.GaussianBlur(img_gray,(5,5),0)
    img_canny=cv2.Canny(img_gauss,75,255)
    a=np.hstack((img_canny,img_gauss))
    hough_lines=cv2.HoughLinesP(img_canny,1,np.pi/90,5)
    img_hough_lines=np.zeros_like(img_gray,dtype=np.uint8)
    
    if hough_lines is not None:
        for i in range(0, len(hough_lines)):
            l = hough_lines[i][0]
            cv2.line(img_hough_lines, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
        
        contours,hierarchy = cv2.findContours(img_hough_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 400: # 노이즈 제거, 너무 작으면 무시
                continue
            if np.count_nonzero(np.sum([contours[i]==0,contours[i]==h-1 , contours[i]==w-1]))!=0:
                continue
            eps=cv2.arcLength(contours[i],True)
            
            approx=cv2.approxPolyDP(contours[i],epsilon=eps*0.01,closed=True)
            vtc=len(approx)
            if vtc==7:
                hull=cv2.convexHull(approx,returnPoints=False)
                if hull is not None:
                    defects=cv2.convexityDefects(approx,hull)
                    
                    if defects is not None:
                        if len(defects)==2:
                            cv2.drawContours(img,[approx],0,(0,0,255),3)
                
        
    a=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
    b=np.hstack((cv2.cvtColor(img_hough_lines,cv2.COLOR_GRAY2BGR),img))
    merged=np.vstack((a,b))
    cv2.imshow('img',merged)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()
 