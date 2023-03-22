import cv2

img=cv2.imread("a.png",cv2.IMREAD_COLOR)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,img_bin=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
contours,_=cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) < 400: # 노이즈 제거, 너무 작으면 무시
        continue
    # 근사화
    ls=[0.005,0.01,0.02,0.03,0.05,0.1]
    for j in range(len(ls)):
        eps=cv2.arcLength(contours[i], True)*ls[j]
        approx = cv2.approxPolyDP(contours[i], eps, True)
        print(f'eps * : {ls[j]}')
        print(f'epsilon: {eps}')
        print(f'vtc: {len(approx)}')
        print(f'\n')
        cv2.drawContours(img,[approx],0,(0,0,255),2)
        cv2.putText(img,f'{ls[j]}',tuple(approx[0][0]),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),1)
        cv2.imshow(f'{ls[j]}',img)
        cv2.waitKey(0)
cv2.destroyAllWindows()