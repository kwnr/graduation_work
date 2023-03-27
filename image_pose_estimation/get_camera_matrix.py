#http://www.gisdeveloper.co.kr/?p=6868

import cv2
import numpy as np
import glob
import pickle

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp=np.zeros((9*6,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints=[]
imgpoints=[]

images=glob.glob("/Users/hyeokbeom/Desktop/graduation_work/image_rotate_estimation/*.jpg")

for image in images:
    img=cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret,corners=cv2.findChessboardCorners(gray,(9,6),None,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK)
 
    
    if ret:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img,(9,6),corners2,ret)
ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
f=open('camera_matrix.pkl','wb')
pickle.dump([mtx,dist],f)
print(mtx)
        
        
cv2.destroyAllWindows()