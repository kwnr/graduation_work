import cv2
import numpy as np
import time
import shapely

def setLabel(img, pts, label):
    # 사각형 좌표 받아오기
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    
    
cap=cv2.VideoCapture(0)
frame_rate=1
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
    img_gauss=cv2.GaussianBlur(img_gray,(9,9),0)    #blur capture
    img_canny=cv2.Canny(img_gauss,100,200)          #
    a=np.hstack((img_canny,img_gauss))
    
    
    #_,img_bin=cv2.threshold(img_canny,127,255,cv2.THRESH_BINARY)
    _,img_bin=cv2.threshold(img_gray,-1,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,contours,-1,(255,0,0),1)
    for i in range(len(contours)):
        #cv2.putText(img,f'{i}',contours[i][0][0],cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        if cv2.contourArea(contours[i]) < 400: # 노이즈 제거, 너무 작으면 무시
            continue
        if not shapely.Polygon(contours[i][:,0,:]).is_valid:
            continue
            
        """if np.count_nonzero(contours[i][:,0,0]==0)!=0:
            continue
        if np.count_nonzero(contours[i][:,0,1]==0)!=0:
            continue
        if np.count_nonzero(contours[i][:,0,0]==w-1)!=0:
            continue
        if np.count_nonzero(contours[i][:,0,1]==h-1)!=0:
            continue
        if np.count_nonzero(contours[i][:,0,0]==w)!=0:
            continue
        if np.count_nonzero(contours[i][:,0,1]==h)!=0:
            continue"""

        eps=cv2.arcLength(contours[i],True)          
        approx=cv2.approxPolyDP(contours[i],epsilon=eps*0.02,closed=True)
        #cv2.drawContours(img,[approx],0,(255,0,0),3)
        vtc=len(approx)
        #cv2.putText(img,f'vtc:{vtc}',approx[0][0],cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),3)
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
                        points=np.array([defects[0][0][:2],defects[1][0][:2]])
                        if not ((points[:,0]-points[:,1])%7==5).all():
                            continue
                        cv2.drawContours(img,[approx],0,(0,0,255),3)
                        x,y,w,h=cv2.boundingRect(approx)
                        cv2.rectangle(img,[x,y],[x+w,y+h],(0,255,255),5)
                        cv2.putText(img,f'1:{points[0][0]},{points[0][1]}, 2:{points[1][0]},{points[1][1]}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),3)
                
        
    a=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
    b=np.hstack((cv2.cvtColor(img_bin,cv2.COLOR_GRAY2BGR),img))
    merged=np.vstack((a,b))
    cv2.putText(merged,f'{time_elapsed:.2f}',[10,50],cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)       
     
    cv2.imshow('img',merged)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()
 