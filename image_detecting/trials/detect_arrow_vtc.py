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
    
    
cap=cv2.VideoCapture(0)
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
    img_gauss=cv2.GaussianBlur(img_gray,(5,5),0)    #blur capture
    img_canny=cv2.Canny(img_gauss,30,255)          #
    a=np.hstack((img_canny,img_gauss))
    hough_lines=cv2.HoughLinesP(img_canny,1,np.pi/180,10)
    img_hough_lines=np.zeros_like(img_gray,dtype=np.uint8)
    
    
    if hough_lines is not None:
        for i in range(0, len(hough_lines)):
            l = hough_lines[i][0]
            cv2.line(img_hough_lines, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
        img_hough_blur=cv2.GaussianBlur(img_hough_lines,(11,11),0)
        
        _,img_bin=cv2.threshold(img_hough_blur,127,255,cv2.THRESH_BINARY)  #불필요
        cv2.imshow('a',img_bin)
        contours,hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img,contours,-1,(255,0,0),3)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < 1000: # 노이즈 제거, 너무 작으면 무시
                continue
            if np.count_nonzero(np.sum([contours[i]==0,contours[i]==h-1,contours[i]==w-1]))!=0:
                continue
            #if hierarchy[0][i][2]!=-1:
            #    continue
            #if hierarchy[0][i][3]==-1:
            #    continue
            eps=cv2.arcLength(contours[i],True)
            
            approx=cv2.approxPolyDP(contours[i],epsilon=eps*0.01,closed=True)
            cv2.drawContours(img,[approx],0,(255,0,0),3)
            vtc=len(approx)
            cv2.putText(img,f'vtc:{vtc}',approx[0][0],cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),3)
            if vtc==7:
                cv2.drawContours(img,[approx],0,(0,255,0),3)

                hull=cv2.convexHull(approx,returnPoints=False)
                if hull is not None:
                    try:
                        defects=cv2.convexityDefects(approx,hull)
                    except:
                        print(f"****")
                        continue
                    
                    if defects is not None:
                        if len(defects)==2:
                            cv2.drawContours(img,[approx],0,(0,0,255),3)
                            x,y,w,h=cv2.boundingRect(approx)
                            cv2.rectangle(img,[x,y],[x+w,y+h],(0,0,255),5)
                
        
    a=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
    b=np.hstack((cv2.cvtColor(img_hough_lines,cv2.COLOR_GRAY2BGR),img))
    merged=np.vstack((a,b))
    cv2.putText(merged,f'{time_elapsed:.2f}',[10,50],cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)       
     
    cv2.imshow('img',merged)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()
 