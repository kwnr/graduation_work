import cv2
import numpy as np
import pickle

class Detector():
    def __init__(self,cap) -> None:
        self.aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        with open('../camera_matrix.pkl','rb') as f:
            self.camera_matrix,self.dist_coeffs=pickle.load(f)
        self.detector=cv2.aruco.ArucoDetector(self.aruco_dict)
        self.board=cv2.aruco.GridBoard([2,2],100,20,self.aruco_dict,ids=np.array([1,11,21,31]))
        self.cap=cap
        self.rvec=np.array([0,0,0])
        self.tvec=np.array([0,0,0])
        
    def set_cap_frame_size(self,w,h):
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,w)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,h)
        
    def set_cap_frame_rate(self,fps):
        self.cap.set(cv2.CAP_PROP_FPS,fps)
        
    def preprocessing(self,img):
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
        return img
    
    def detect_match_marker(self,img):
        self.corners,self.ids,_=self.detector.detectMarkers(img)
        objp,imgp=self.board.matchImagePoints(self.corners,self.ids)
        return objp,imgp

    def solve_pnp(self,objp,imgp):
        retval,rvec,tvec=cv2.solvePnP(objp,imgp,self.camera_matrix,self.dist_coeffs,flags=cv2.SOLVEPNP_IPPE)
        #retval,rvec,tvec,inliers=cv2.solvePnPRansac(objp,imgp,self.camera_matrix,self.dist_coeffs,flags=cv2.SOLVEPNP_IPPE)
        rvec,tvec=cv2.solvePnPRefineLM(objp,imgp,self.camera_matrix,self.dist_coeffs,rvec,tvec)
        return rvec,tvec

    def get_pose(rvec):
        pitch,yaw,roll=np.rad2deg(rvec)[:,0]
        return pitch,yaw,roll
    
    def run(self,draw=False):
        _,img=self.cap.read()
        img=self.preprocessing(img)
        self.objp,self.imgp=self.detect_match_marker(img)
        if self.objp is not None:
            self.rvec,self.tvec=self.solve_pnp(self.objp,self.imgp)
            if draw:
                cv2.aruco.drawDetectedMarkers(img,self.corners,self.ids)
                cv2.drawFrameAxes(img,self.camera_matrix,self.dist_coeffs,self.rvec,self.tvec,100)
                cv2.putText(img,f'{self.rvec}, {self.tvec}',(0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),1)
            
        return self.rvec,self.tvec,img
