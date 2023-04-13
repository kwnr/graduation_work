import cv2
import numpy as np
import pickle

class BoardDetection():
    def __init__(self) -> None:
        self.aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        with open('camera_matrix.pkl','rb') as f:
            self.camera_matrix,self.dist_coeffs=pickle.load(f)
        self.detector=cv2.aruco.ArucoDetector(self.aruco_dict)
        self.board=cv2.aruco.GridBoard([2,2],100,20,self.aruco_dict,ids=np.array([1,11,21,31]))
    
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

    def pipeline(self,img,draw=False):
        roll,pitch,yaw=[None,None,None]
        objp,imgp=self.detect_match_marker(img)
        if objp is not None:
            rvec,tvec=self.solve_pnp(objp,imgp)
            if draw:
                cv2.aruco.drawDetectedMarkers(img,self.corners,self.ids)
                cv2.drawFrameAxes(img,self.camera_matrix,self.dist_coeffs,rvec,tvec,10)
            
            return np.vstack((tvec,rvec))