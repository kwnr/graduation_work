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
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    objp=np.zeros((4,3),np.float32)
    _,img_bin=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    contours,hierarchy=cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        eps=cv2.arcLength(contour,True) 
        approx=cv2.approxPolyDP(contour,eps*0.02,True)
        if len(approx)==4:
            objp[:,:2]=approx[:,0,:]
            return objp


#http://www.gisdeveloper.co.kr/?p=6908 참조
def estimate_pose(img,approx_box,mtx,dst,objp):
    axis = np.float32([[100,0,0], [0,100,0], [0,0,-100]]).reshape(-1,3)
    corners2=cv2.cornerSubPix(img,np.float32(approx_box[:,0,:]),(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    _,rvecs,tvecs,inliers=cv2.solvePnPRansac(objp,corners2,mtx,dst)
    imgpts,jac=cv2.projectPoints(axis,rvecs,tvecs,mtx,dst)
    canv=img.copy()
    canv=cv2.cvtColor(canv,cv2.COLOR_GRAY2BGR)
    canv=draw_axes(canv,corners2,imgpts)
    cv2.imshow('img',canv)
    
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
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        arrows=detect_arrow.detect_arrow(img)
        for arrow in arrows:
            estimate_pose(gray,arrow['parent'],mtx,dst,objp)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        
        
cap.release()
cv2.destroyAllWindows()
        