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
    
while cap.isOpened():
    time_elapsed = time.time() - prev   #10FPS로 제한
    res, image = cap.read()             #

    if time_elapsed > 1./frame_rate:
        prev = time.time()

    status, img=cap.read()
    
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 400: # 노이즈 제거, 너무 작으면 무시
            continue
        eps=cv2.arcLength(contours[i],True)
        approx=cv2.approxPolyDP(contours[i],epsilon=eps*0.03,closed=True)
        vtc=len(approx)
        if vtc==12:
            M=cv2.moments(contours[i])  #꼭짓점이 12개인 폐곡선 컨투어의 이미지 모멘트 계산
            cx=int(M['m10']/M['m00'])   #폐곡선의 중심 계산
            cy=int(M['m01']/M['m00'])
            cv2.circle(img,(cx,cy),1,(0,0,255),2) 
            cv2.drawContours(img, [contours[i]], 0, (0,0,255), 2)
            cv2.putText(img, f'AREA: {cv2.contourArea(contours[i])}, COORD: ({cx}, {cy})', tuple(contours[i][0][0]),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
            rect=cv2.minAreaRect(contours[i])   #십자를 감싸는 최소 넓이 사각형, https://docs.opencv.org/4.3.0/dd/d49/tutorial_py_contour_features.html 참조
            box=cv2.boxPoints(rect)
            box=np.intp(box)
            cv2.drawContours(img,[box],0,(255,0,0),2)
            
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("a.png",img)
        break    

cap.release()
cv2.destroyAllWindows()
