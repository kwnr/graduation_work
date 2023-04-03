import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from arrow_detecting.trials import detect_arrow
import numpy as np
import cv2
import pickle

original_ratio=1.5 #교차점에서 head 까지의 거리/right까지의 거리

def find_angle_vector(v1,v2):
    theta=np.arccos(np.dot(v1,v2)/(np.sqrt(np.sum(np.square(v1)))*np.sqrt(np.sum(np.square(v1)))))
    theta=np.rad2deg(theta)
    return theta

def estimate_roll(properties):
    
    head=properties["head"]
    center=properties["center"]
    hx, hy = head
    cx, cy = center
    
    if hy - cy == 0:
        if hx > cx:
            heading = np.pi
        else:
            heading = -np.pi
    else:
        heading = np.arctan(np.divide(hx - cx, hy - cy))
        if hy > cy:
            if hx < cx:
                heading = heading + np.pi
            else:
                heading = heading - np.pi
    return np.rad2deg(heading)  

def get_objp():
    """저장된 이미지를 바탕으로 3차원 좌표 생성,\n
    외각 사각형을 기준으로 4개의 3차원 좌표 생성

    Returns:
        ndarray: shape 4*3
    """
    img=cv2.imread("/Users/hyeokbeom/Desktop/graduation_work/arrow_box.png")
    objp=np.zeros((4,3),np.float32)                     # 네 꼭짓점의 3차원 좌표를 담기 위한 배열
    arrows=detect_arrow.detect_arrow(img)
    
    for arrow in arrows:
        distance = [                                    # left에서 parent의 꼭짓점까지의 거리 계산
                np.sqrt(np.sum(np.square(arrow["parent"][i][0] - arrow["left"])))
                for i in range(4)
        ]
        
        tl = np.argmin(distance) - 1                    # parent의 왼쪽 상단 위치
        box=np.float32(                                 # parent에서 왼쪽 상단의 인덱스가 0 으로 정렬된 box 생성
            [
                arrow["parent"][tl % 4, 0, :],
                arrow["parent"][(tl + 1) % 4, 0, :],
                arrow["parent"][(tl + 2) % 4, 0, :],
                arrow["parent"][(tl + 3) % 4, 0, :],
            ])
        objp[:,:2]=box                                  # x,y 좌표 대입
        
        return objp ###indent error


#http://www.gisdeveloper.co.kr/?p=6908 참조
def estimate_pose(img,approx_box,mtx,dst,objp):
    axis = np.float32([[800,0,0], [0,800,0], [0,0,-800]]).reshape(-1,3)
    
    corners2=cv2.cornerSubPix(img,np.float32(approx_box),(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)) #box의 코너 위치 정밀화
    
    # 기준자세 좌표 (objp)와 정밀화된 꼭짓점을 이용해 자세추정
    # rvecs: 회전 벡터, tvecs: 이동 벡터
    _,rvecs,tvecs,inliers=cv2.solvePnPRansac(objp,corners2,mtx,dst) 
    
    # 회전된 축 그리기
    imgpts,jac=cv2.projectPoints(axis,rvecs,tvecs,mtx,dst)
    canv=img.copy()
    canv=cv2.cvtColor(canv,cv2.COLOR_GRAY2BGR)
    canv=draw_axes(canv,corners2,imgpts)
    print(rvecs)
    return rvecs
    
def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, np.int32(corner), np.int32(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, np.int32(corner), np.int32(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, np.int32(corner), np.int32(imgpts[2].ravel()), (0,0,255), 5)
    return img

if __name__=='__main__':
    cap=cv2.VideoCapture(1)
    f=open('/Users/hyeokbeom/Desktop/graduation_work/camera_matrix.pkl','rb')
    mtx,dst=pickle.load(f)
    f.close()
    objp=get_objp()
    while cap.isOpened():
        _,img=cap.read()
        #img=cv2.flip(img,1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        arrows=detect_arrow.detect_arrow(img)

        
        for arrow in arrows:
            distance = [
                np.sqrt(np.sum(np.square(arrow["parent"][i][0] - arrow["left"])))
                for i in range(4)
            ]
            tl = np.argmin(distance) - 1
            box=np.float32(
                        [
                            arrow["parent"][tl % 4, 0, :],
                            arrow["parent"][(tl + 1) % 4, 0, :],
                            arrow["parent"][(tl + 2) % 4, 0, :],
                            arrow["parent"][(tl + 3) % 4, 0, :],
                        ])
            rvecs=estimate_pose(gray,box,mtx,dst,objp)
            x,y,z=np.rad2deg(rvecs[:,0])
            cv2.putText(img,f'{x:.2f},{y:.2f},{z:.2f}',np.int32(box[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.imshow('a',img)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        
        
cap.release()
cv2.destroyAllWindows()
        