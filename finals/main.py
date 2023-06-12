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

global phi, psi, theta


class gyro():
    def __init__(self):
        self.sensor = MPU6050()
        self.kf = KalmanFilter(4, 4)
        self.kf.H = np.eye(4)
        self.kf.Q = np.eye(4) * 0.0001
        self.kf.R = np.eye(4) * 10
        self.kf.x = np.array([1, 0, 0, 0]).reshape(4, 1)
        self.kf.P = np.eye(4)

        self.biasGx = -0.00878016
        self.biasGy = 0.00555219
        self.biasGz = -0.00088954

        self.phi, self.theta, self.psi = 0, 0, 0
        self.dt = 0.1
        self.g = 9.81
        self.phis = [self.phi]
        self.thetas = [self.theta]
        self.psis = [self.psi]

    def run(self):
        Ax, Ay, Az, Gx, Gy, Gz = self.sensor.read_value()
        Ax, Ay, Az, Gx, Gy, Gz=np.deg2rad([Ax, Ay, Az, Gx, Gy, Gz])
        self.acc = [Ax, Ay, Az]
        self.ang_rate = [Gx, Gy, Gz]
        self.kf.A = np.eye(4) + self.dt * 1 / 2 * np.array(
            [
                [0, -Gx, -Gy, -Gz],
                [Gx, 0, Gz, -Gy],
                [Gy, -Gz, 0, Gx],
                [Gz, Gy, -Gx, 0],
            ]
        )

        self.theta = np.arctan(Ay / Az)
        self.phi = np.arctan(Ax / np.sqrt(Ay**2 + Az**2))
        self.psi = 0

        cosphi = np.cos(self.phi / 2)
        costhe = np.cos(self.theta / 2)
        cospsi = np.cos(self.psi / 2)
        sinphi = np.sin(self.phi / 2)
        sinthe = np.sin(self.theta / 2)
        sinpsi = np.sin(self.psi / 2)

        z = np.array(
            [
                [cosphi * costhe * cospsi + sinphi * sinthe * sinpsi],
                [sinphi * costhe * cospsi - cosphi * sinthe * sinpsi],
                [cosphi * sinthe * cospsi + sinphi * costhe * sinpsi],
                [cosphi * costhe * sinpsi - sinphi * sinthe * cospsi],
            ]
        )
        x = self.kf.run(z)#quat, [w,x,y,z]
        self.phi_kf = np.arctan2(
            2 * (x[2] * x[3] + x[0] * x[1]), 1 - 2 * (x[1] ** 2 + x[2] ** 2)
        )
        self.theta_kf = np.arcsin(2*(x[0]*x[2]-x[1]*x[3]))
        self.psi_kf = np.arctan2(
            2 * (x[1] * x[2] + x[0] * x[3]), 1 - 2 * (x[2] ** 2 + x[3] ** 2)
        )


class LowPassFilter(object):
    def __init__(self, cut_off_freqency, ts):
        # cut_off_freqency: 차단 주파수
        # ts: 주기

        self.ts = ts
        self.cut_off_freqency = cut_off_freqency
        self.tau = self.get_tau()

        self.prev_data = 0.0

    def get_tau(self):
        return 1 / (2 * np.pi * self.cut_off_freqency)

    def filter(self, data):
        val = (self.ts * data + self.tau * self.prev_data) / (self.tau + self.ts)
        self.prev_data = val
        return val


class main(threading.Thread):
    def __init__(self):
        super().__init__()
        
        self.capL=cv2.VideoCapture(-1)
        self.capR=cv2.VideoCapture(2)


        w = 640
        h = 480
        fps = 10

        self.detL = Detector(self.capL)
        self.detR = Detector(self.capR)

        self.detL.set_cap_frame_size(w, h)
        self.detR.set_cap_frame_size(w, h)

        self.detL.set_cap_frame_rate(fps)
        self.detR.set_cap_frame_rate(fps)

        self.tx = symaTX()
        self.tx.radio.begin()
        self.tx.radio.setAutoAck(False)
        self.tx.radio.setAddressWidth(5)
        self.tx.radio.setRetries(15, 15)
        self.tx.radio.setDataRate(RF24_250KBPS)
        self.tx.radio.setPALevel(RF24_PA_HIGH)
        self.tx.radio.setPayloadSize(10)
        self.mpu = gyro()
        time.sleep(0.015)

        addr = [161, 105, 1, 104, 204]
        addr.reverse()
        self.tx.addr = addr
        self.tx.init2()
        self.tx.start()

        self.control_init()
        self.des_dist = 400

        with open('data.pkl','rb') as f:
            data=pickle.load(f)
        self.M1=data['M1']
        self.M2=data['M2']
        self.P1=data['P1']
        self.P2=data['P2']
        self.dist1=data['dist1']
        self.dist2=data['dist2']
        self.R1=data['R1']
        self.R2=data['R2']
        self.R=data['R']
        self.T=data['T']

        self.detL.camera_matrix = self.M1
        self.detR.camera_matrix = self.M2
        self.detL.dist_coeffs = self.dist1
        self.detR.dist_coeffs = self.dist2
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={"float_kind": float_formatter})
        self.u0_lpf=LowPassFilter(1,0.1)
        self.u1_lpf=LowPassFilter(1,0.1)
        self.u2_lpf=LowPassFilter(1,0.1)
        self.u3_lpf=LowPassFilter(1,0.1)
        self.state0_lpf=LowPassFilter(1,0.1)
        self.state2_lpf=LowPassFilter(1,0.1)
        self.state4_lpf=LowPassFilter(1,0.1)
        self.state6_lpf=LowPassFilter(1,0.1)
        self.state8_lpf=LowPassFilter(1,0.1)
        self.state10_lpf=LowPassFilter(1,0.1)
    def control_init(self):
        g = 9.81
        m = 2
        Iz = 1 / 2 * m * 0.1**2
        Ix = 1 / 12 * m * (3 * 0.1**2 + 0.1**2)
        Iy = Ix
        self.A = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -g, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        self.B = np.array(
            [
                [0, 0, 0, 0],
                [1 / m, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1 / Ix, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1 / Iy, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1 / Iz],
            ]
        )
        acceptable_error_states=[10,100,1,1]*3
        acceptable_error_inputs=[10,10,10,10]
        Q=np.diag(np.divide(1,np.square(acceptable_error_states)))
        R = np.diag(np.divide(1,np.square(acceptable_error_inputs)))
        self.K, S, E = control.lqr(self.A, self.B, Q, R)
        self.state_des = np.zeros((12, 1))
        self.state = np.zeros((12, 1))
        self.state_pre=np.zeros((12,1))
        self.tspan = []
        self.throts = []

    def state_feedback(self):
        u = self.K @ (self.state_des - self.state)
        u[0][0] = max(min(u[0][0], 255), 0)
        u[1][0] = max(min(u[1][0], 127), -127)
        u[2][0] = max(min(u[2][0], 127), -127)
        u[3][0] = max(min(u[3][0], 127), -127)
        return u
    
    def get_control(self,u):
        ##roll left 255 right 127
        ##pitch forward 127 backwrad 255
        ##yaw left 255 right 127
        sig=u.copy()
        for i in range(3):
            if sig[i + 1] <= 0:
                sig[i + 1] = abs(sig[i + 1])
            else:
                sig[i + 1] = sig[i + 1] + 128
        sig = np.int16(sig)
        sig %= 256
        return sig

    def rot(self, a, b, g):
        return np.array(
            [
                [
                    cos(a) * cos(b),
                    cos(a) * sin(b) * sin(g) - sin(a) * cos(g),
                    cos(a) * sin(b) * cos(g) + sin(a) * sin(g),
                ],
                [
                    sin(a) * cos(b),
                    sin(a) * sin(b) * sin(g) + cos(a) * cos(g),
                    sin(a) * sin(b) * cos(g) - cos(a) * sin(g),
                ],
                [-sin(b), cos(b) * sin(g), cos(b) * cos(g)],
            ]
        )

    def run(self):
        prev_time=time.monotonic()
        us,states=[],[]
        states_pre=[]
        dist=self.des_dist
        v1=np.array([0,0,1])
        cam_dist=300
        while time.monotonic()-prev_time<5:
            self.tx.throttle=0
            self.tx.pitch=255
            self.tx.roll=127
            self.tx.yaw=255
        while time.monotonic()-prev_time<10:
            self.tx.throttle=130
            self.tx.pitch=0
            self.tx.roll=0
            self.tx.yaw=0
        while True:
            rvecL, tvecL, imgL, resL = self.detL.run(draw=True)
            rvecR, tvecR, imgR, resR = self.detR.run(draw=True)
            imgpL = self.detL.imgp
            imgpR = self.detR.imgp

            c1=tvecL[:2]
            c2=tvecR[:2]

            if resL and resR:
                cd1=self.M1@tvecL
                cd2=self.M2@self.R@tvecR
                ang1=np.arccos((cd1.T@v1)/(np.linalg.norm(cd1)*np.linalg.norm(v1)))
                ang2=np.arccos((cd2.T@v1)/(np.linalg.norm(cd2)*np.linalg.norm(v1)))
                gamma=np.pi-ang1-ang2
                dist_v=cam_dist*np.sin(ang2)/np.sin(gamma)*np.sin(ang1)
                dist_off=cam_dist/2-cam_dist*np.sin(ang2)/np.sin(gamma)
                dist=np.sqrt(dist_v**2+dist_off**2)
            
            curr_time = time.monotonic()
            dt = curr_time - prev_time
            prev_time = curr_time
            state_prev = self.state_pre.copy()
            
            self.mpu.dt=dt
            self.mpu.run()
            theta,phi=self.mpu.theta_kf,self.mpu.phi_kf
            psi=0

            if c2 is not None:

                self.state_pre[0] = c1[1]  # z
                self.state_pre[1] = (c1[1] - state_prev[0]) / dt  # zd
                self.state_pre[2] = psi-rvecL[1]  # psi
                self.state_pre[3] = (psi - state_prev[2]) / dt   # psid
                self.state_pre[4] = dist  # x
                self.state_pre[5] = dist - state_prev[4]  # xd
                self.state_pre[6] = phi-rvecL[2]  # phi
                self.state_pre[7] = (phi - state_prev[6]) / dt  # phid
                self.state_pre[8] = c1[0]  # y
                self.state_pre[9] = (c1[0] - state_prev[8]) / dt  # yd
                self.state_pre[10] = theta-rvecL[0]  # theta
                self.state_pre[11] = (theta - state_prev[10]) / dt  # thetad

                self.state_des[0] = 0
                self.state_des[1] = 0
                self.state_des[2] = 0
                self.state_des[3] = 0
                self.state_des[4] = self.des_dist
                self.state_des[5] = 0
                self.state_des[6] = 0
                self.state_des[7] = 0
                self.state_des[8] = 0
                self.state_des[9] = 0
                self.state_des[10] = 0
                self.state_des[11] = 0

            self.state[0]=self.state0_lpf.filter(self.state_pre[0])
            self.state[2]=self.state2_lpf.filter(self.state_pre[2])
            self.state[4]=self.state4_lpf.filter(self.state_pre[4])
            self.state[6]=self.state6_lpf.filter(self.state_pre[6])
            self.state[8]=self.state8_lpf.filter(self.state_pre[8])
            self.state[10]=self.state10_lpf.filter(self.state_pre[10])
            
            u_raw=self.state_feedback()
            u = self.get_control(u_raw)
            
            
            
            u[0][0]=max(u[0][0],10)
            
            self.tx.throttle = u[0][0]
            self.tx.roll=u[1][0]
            self.tx.pitch=u[2][0]
            self.tx.yaw=u[3][0]
            self.tspan.append(curr_time)
            self.throts.append(self.tx.throttle)

            print(
                f"\ndt:{dt}\npt:{c1.T}\nthrottle:{u_raw[0][0]},    pitch:{u_raw[1][0]},  yaw:{u_raw[2][0]},  roll:{u_raw[3][0]}"
                )
            print(dict(zip(["z","zd",  "psi", "psid",    "x",   "xd", "phi",  "phid",    "y",   "yd",  "theta",   "thetad"],*self.state.T)))
            print(f'dist: {dist}')
            states.append(self.state)
            states_pre.append(self.state_pre)
            us.append(u)
            #img = np.hstack((imgL, imgR))
            #cv2.imshow("img", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                self.capL.release()
                self.capR.release()
                with open("pickle.pkl", "wb") as f:
                    pickle.dump([self.tspan, states,states_pre, us], f)
                print('pkl writed')
                exit(1)


if __name__ == "__main__":
    ctrl = main()
    ctrl.run()
