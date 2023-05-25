import numpy as np
import cv2
from pyrf24.rf24 import *
from aruco_board_detection_fn import Detector
import time
import threading
from transmitter import symaTX
from read_sensor import MPU6050
from kalman_filter import KalmanFilter
from numpy import sin
from numpy import cos
from numpy import tan
import control

global phi,psi,theta

class gyro(threading.Thread):
    def __init__(self):
        super().__init__()
        self.sensor=MPU6050()
        self.kf=KalmanFilter(4,4)
        self.kf.H=np.eye(4)
        self.kf.Q=np.eye(4)*0.0001
        self.kf.R=np.eye(4)*10
        self.kf.x=np.array([0,0,0,1]).reshape(4,1)
        self.kf.P=np.eye(4)

        self.biasGx=-0.00878016
        self.biasGy= 0.00555219
        self.biasGz=-0.00088954

        self.phi,self.theta,self.psi=0,0,0
        self.dt=0.1
        self.g=9.81
        self.phis=[self.phi]
        self.thetas=[self.theta]
        self.psis=[self.psi]
        
    def run(self):
        while True:
            Ax,Ay,Az,Gx,Gy,Gz=self.sensor.read_value()
            self.kf.A=np.eye(4)+self.dt*1/2*np.array([
                [0,-Gx,-Gy,-Gz],
                [Gx,0,Gz,-Gy],
                [Gy,-Gz,0,Gx],
                [Gz,Gy,-Gx,0]
            ])
            
            self.theta=np.arctan(Ay/Az)
            self.phi=np.arctan(Ax/np.sqrt(Ay**2+Az**2))
            self.psi=0
            
            cosphi=np.cos(self.phi/2)
            costhe=np.cos(self.theta/2)
            cospsi=np.cos(self.psi/2)
            sinphi=np.sin(self.phi/2)
            sinthe=np.sin(self.theta/2)
            sinpsi=np.sin(self.psi/2)
            
            z=np.array([
                [cosphi*costhe*cospsi+sinphi*sinthe*sinpsi],
                [sinphi*costhe*cospsi-cosphi*sinthe*sinpsi],
                [cosphi*sinthe*cospsi+sinphi*costhe*sinpsi],
                [cosphi*costhe*sinpsi-sinphi*sinthe*cospsi]
            ])
            x=self.kf.run(z)
            self.phi=np.arctan2(2*(x[2]*x[3]+x[0]*x[1]),1-2*(x[1]**2+x[2]**2))
            self.theta=-np.arcsin(2*(x[1]*x[3]-x[0]*x[2]))
            self.psi=np.arctan2(2*(x[1]*x[2]+x[0]*x[3]),1-2*(x[2]**2+x[3]**2))
            time.sleep(self.dt)

class main(threading.Thread):
    def __init__(self):
        super().__init__()
        
        cap1=cv2.VideoCapture(0)
        cap2=cv2.VideoCapture(2)

        w=640
        h=480
        fps=10

        self.det1=Detector(cap1)
        self.det2=Detector(cap2)

        self.det1.set_cap_frame_size(w,h)
        self.det2.set_cap_frame_size(w,h)

        self.det1.set_cap_frame_rate(fps)
        self.det2.set_cap_frame_rate(fps)
        
        
        self.tx=symaTX()
        self.tx.radio.begin()
        self.tx.radio.setAutoAck(False)
        self.tx.radio.setAddressWidth(5)
        self.tx.radio.setRetries(15,15)
        self.tx.radio.setDataRate(RF24_250KBPS)
        self.tx.radio.setPALevel(RF24_PA_HIGH)
        self.tx.radio.setPayloadSize(10)
        self.mpu=gyro()
        time.sleep(0.015)
    
        addr=[161, 105, 1, 104, 204]
        addr.reverse()
        self.tx.addr=addr
        self.tx.init2()
        self.tx.start()
        self.mpu.start()
        
    def control_init(self):
        g=9.81
        m=2
        Iz=1/2*m*0.1**2
        Ix=1/12*m*(3*0.1**2+0.1**2)
        Iy=Ix
        self.A=np.array([
            [0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,g,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,-g,0],
            [0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0]
            ])

        self.B=np.array([
            [0,0,0,0],
            [1/m,0,0,0],
            [0,0,0,0],
            [0,1/Ix,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,1/Iy,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,1/Iz]
            ])
        Q=np.eye(12)
        R=np.eye(4)
        self.K,S,E=control.lqr(self.A,self.B,Q,R)
        self.state_des=np.zeros((12,1))
        self.state=np.zeros((12,1))
    
    def get_control(self):
        u=self.K@(self.state_des-self.state)
        return u
    
    def rot(self,a,b,g):
        return np.array([
    [cos(a)*cos(b),cos(a)*sin(b)*sin(g)-sin(a)*cos(g),cos(a)*sin(b)*cos(g)+sin(a)*sin(g)],
    [sin(a)*cos(b),sin(a)*sin(b)*sin(g)+cos(a)*cos(g),sin(a)*sin(b)*cos(g)-cos(a)*sin(g)],
    [-sin(b),cos(b)*sin(g),cos(b)*cos(g)]
    ])
    
        
        
    def run(self):
        while True:
            rvec1,tvec1,img1=self.det1.run(draw=True)
            rvec2,tvec2,img2=self.det2.run(draw=True)
            imgp1=self.det1.imgp
            imgp2=self.det2.imgp
            c=np.array([0,0],dtype=np.float32)
            cc=0
            if self.det1.imgp is not None:
                M1=cv2.moments(imgp1)
                c+=[M1['m10']/M1['m00'],M1['m01']/M1['m00']]
                cc+=1
            if self.det2.imgp is not None:
                M2=cv2.moments(imgp2)
                c+=[M2['m10']/M2['m00'],M2['m01']/M2['m00']]
                cc+=1
            if cc!=0:
                c=c/cc
                cv2.circle(img1,np.int32(c),3,(255,0,0),-1)
                cv2.circle(img2,np.int32(c),3,(255,0,0),-1)
            
            
            
            img=np.hstack((img1,img2))
            print(self.mpu.phi,self.mpu.psi,self.mpu.theta)
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                exit(0)

        
        
        

        
        
if __name__=="__main__":
    ctrl=main()
    ctrl.run()

