import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from image_detecting.trials import detect_arrow
import numpy as np
import cv2
import pickle

original_ratio=1.5 #head/right

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
    img=cv2.imread("/Users/hyeokbeom/Desktop/graduation_work/arrow_box.png")
    objp=np.zeros((4,3),np.float32)
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
        objp[:,:2]=box
        return objp


#http://www.gisdeveloper.co.kr/?p=6908 참조
def estimate_pose(img,approx_box,mtx,dst,objp):
    axis = np.float32([[800,0,0], [0,800,0], [0,0,-800]]).reshape(-1,3)
    corners2=cv2.cornerSubPix(img,np.float32(approx_box),(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    _,rvecs,tvecs,inliers=cv2.solvePnPRansac(objp,corners2,mtx,dst)
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
        