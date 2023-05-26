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
import pickle

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
            self.acc=[Ax,Ay,Az]
            self.ang_rate=[Gx,Gy,Gz]
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
            self.phi_kf=np.arctan2(2*(x[2]*x[3]+x[0]*x[1]),1-2*(x[1]**2+x[2]**2))
            self.theta_kf=-np.arcsin(2*(x[1]*x[3]-x[0]*x[2]))
            self.psi_kf=np.arctan2(2*(x[1]*x[2]+x[0]*x[3]),1-2*(x[2]**2+x[3]**2))
            time.sleep(self.dt)

class LowPassFilter(object):
    def __init__(self, cut_off_freqency, ts):
    	# cut_off_freqency: 차단 주파수
        # ts: 주기
        
        self.ts = ts
        self.cut_off_freqency = cut_off_freqency
        self.tau = self.get_tau()

        self.prev_data = 0.
        
    def get_tau(self):
        return 1 / (2 * np.pi * self.cut_off_freqency)

    def filter(self, data):
        val = (self.ts * data + self.tau * self.prev_data) / (self.tau + self.ts)
        self.prev_data = val
        return val

class main(threading.Thread):
    def __init__(self):
        super().__init__()
        
        self.capL=cv2.VideoCapture(0)
        self.capR=cv2.VideoCapture(2)

        w=640
        h=480
        fps=10

        self.detL=Detector(self.capL)
        self.detR=Detector(self.capR)

        self.detL.set_cap_frame_size(w,h)
        self.detR.set_cap_frame_size(w,h)

        self.detL.set_cap_frame_rate(fps)
        self.detR.set_cap_frame_rate(fps)
        
        
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
        
        self.control_init()
        self.des_dist=-400
        
        self.M1=np.array([[465.24217997,   0.         ,341.98704948],
                [  0.         ,458.21944784 ,215.39055162],
                [  0.           ,0.           ,1.        ]])
        self.M2=np.array([[465.24217997,   0.         ,341.98704948],
                [  0.         ,458.21944784 ,215.39055162],
                [  0.           ,0.           ,1.        ]])
        self.dist1=np.array([[-0.01575441, -0.16604411, -0.00668898,  0.01829878,  0.1655708 ]])
        self.dist2=np.array([[-0.01575441, -0.16604411, -0.00668898,  0.01829878,  0.1655708 ]])
        self.R=np.array([[ 1.00000000e+00, -1.19438593e-11,  9.23983516e-12],
                [ 1.19438593e-11,  1.00000000e+00, -7.68874323e-13],
                [-9.23983516e-12,  7.68874323e-13,  1.00000000e+00]])
        self.T=np.array([[-1.46177495e-10],
                        [ 1.49886332e-11],
                        [ 9.75874994e-11]])

        self.P1=np.array([
            [480.14789983   ,0.         ,354.73688889   ,0.        ],
            [  0.         ,480.14789983, 128.49066401,   0.        ],
            [  0.,           0.,           1.,           0.        ]])
        self.P2=np.array([
            [4.80147900e+02, 0.00000000e+00, 3.54736889e+02, 0.00000000e+00],
            [0.00000000e+00, 4.80147900e+02, 1.28490664e+02, 7.44211607e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])
        
        self.detL.camera_matrix=self.M1
        self.detR.camera_matrix=self.M2
        self.detL.dist_coeffs=self.dist1
        self.detR.dist_coeffs=self.dist2
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        
        
        
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
        self.tspan=[]
        self.throts=[]
    
    def get_control(self):
        u=self.K@(self.state_des-self.state)
        u[0][0]=max(min(u[0][0],255),0)
        u[1][0]=max(min(u[1][0],127),-127)
        u[2][0]=max(min(u[2][0],127),-127)
        u[3][0]=max(min(u[2][0],127),-127)

        ##roll left 255 right 127
        ##pitch forward 127 backwrad 255
        ##yaw left 255 right 127
        for i in range(3):
            if u[i+1]<=0:
                u[i+1]=abs(u[i+1])
            else:
                u[i+1]=u[i+1]+128
        u=np.int16(u)
        u%=256
        return u
    
    def rot(self,a,b,g):
        return np.array([
    [cos(a)*cos(b),cos(a)*sin(b)*sin(g)-sin(a)*cos(g),cos(a)*sin(b)*cos(g)+sin(a)*sin(g)],
    [sin(a)*cos(b),sin(a)*sin(b)*sin(g)+cos(a)*cos(g),sin(a)*sin(b)*cos(g)-cos(a)*sin(g)],
    [-sin(b),cos(b)*sin(g),cos(b)*cos(g)]
    ])
    
    def translate_signal(self,sig):
        t=[sig[0],0,0,0]
        for i in range(1,4):
            if sig[i]>128:
                 t[i]=sig[i]-128
            else:
                t[i]=-sig[i]
        return t
    
    
        
    def run(self):
        prev_time=time.monotonic()
        while True:
            rvecL,tvecL,imgL=self.detL.run(draw=True)
            rvecR,tvecR,imgR=self.detR.run(draw=True)
            imgpL=self.detL.imgp
            imgpR=self.detR.imgp
            c=np.array([0,0],dtype=np.float32)
            cc=0
            if self.detL.imgp is not None:
                M1=cv2.moments(imgpL)
                c+=[M1['m10']/M1['m00'],M1['m01']/M1['m00']]
                cc+=1
            if self.detR.imgp is not None:
                M2=cv2.moments(imgpR)
                c+=[M2['m10']/M2['m00'],M2['m01']/M2['m00']]
                cc+=1
            if cc!=0:
                c=c/cc
                cv2.circle(imgL,np.int32(c),3,(255,0,0),-1)
                cv2.circle(imgR,np.int32(c),3,(255,0,0),-1)
            pt=None
            if imgpL is not None and imgpR is not None:
                if len(imgpL)==16 and len(imgpR)==16:
                    pts=cv2.triangulatePoints(self.P1,self.P2,imgpL,imgpR)
                    pts[:3]=pts[:3]/pts[3]
                    #print(imgpL[0][0],imgpR[0][0],pts.T[0][:3],c)
                    
                    pt=np.average(pts,axis=1)
                    pt_get_time=time.monotonic()
            curr_time=time.monotonic()
            dt=curr_time-prev_time
            prev_time=curr_time
            state_prev=self.state.copy()
            
            if pt is not None:
                if curr_time-pt_get_time>1:
                    pass
                else:
                    self.state[0]=pt[0] #z
                    self.state[1]=(pt[0]-state_prev[0])/dt#zd
                    self.state[2]=self.mpu.psi_kf#psi
                    self.state[3]=(self.mpu.psi_kf-state_prev[2])/dt#psid
                    self.state[4]=pt[2]#x
                    self.state[5]=(pt[2]-state_prev[4])#xd
                    self.state[6]=self.mpu.phi_kf#phi
                    self.state[7]=(self.mpu.phi_kf-state_prev[6])/dt#phid
                    self.state[8]=pt[1]#y
                    self.state[9]=(pt[1]-state_prev[8])/dt#yd
                    self.state[10]=self.mpu.theta_kf#theta
                    self.state[11]=(self.mpu.theta_kf-state_prev[10])/dt#thetad
                
                self.state_des[0]=0
                self.state_des[1]=0
                self.state_des[2]=0
                self.state_des[3]=0
                self.state_des[4]=-self.des_dist
                self.state_des[5]=0
                self.state_des[6]=0
                self.state_des[7]=0
                self.state_des[8]=0
                self.state_des[9]=0
                self.state_des[10]=0
                self.state_des[11]=0
                
            u=self.get_control()
            self.tx.throttle=u[0][0]
            #self.tx.yaw=u[1][0]
            #self.tx.roll=u[2][0]
            #self.tx.pitch=u[3][0]
            self.tspan.append(curr_time)
            self.throts.append(self.tx.throttle)
            img=np.hstack((imgL,imgR))
            
            print(f"\ndt:{dt}\npt:{pt}\n sig:{[self.tx.throttle,self.tx.pitch,self.tx.yaw,self.tx.roll]}\nstate:{self.state.T}\nstate_des:{self.state_des.T}")
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                cv2.destroyAllWindows()
                self.capL.release()
                self.capR.release()
                with open('pickle.pkl','wb') as f:
                    pickle.dump([self.tspan,pt,self.state,u],f)
                exit(1)

        
        
        

        
        
if __name__=="__main__":
    ctrl=main()
    ctrl.run()

