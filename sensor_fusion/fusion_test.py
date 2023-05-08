from kalman_filter import KalmanFilter
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import scipy.io
from scipy.spatial.transform import Rotation as R

def quat2euler(x):
    phi=np.arctan2(2*(x[2]*x[3]+x[0]*x[1]),1-2*(x[1]**2+x[2]**2))
    theta=-np.arcsin(2*(x[1]*x[3]-x[0]*x[2]))
    psi=np.arctan2(2*(x[1]*x[2]+x[0]*x[3]),1-2*(x[2]**2+x[3]**2))
    euler=np.array([phi,theta,psi])
    return euler

def euler2quat(x):
    phi,theta,psi=x
    cosphi=np.cos(phi/2)
    costhe=np.cos(theta/2)
    cospsi=np.cos(psi/2)
    sinphi=np.sin(phi/2)
    sinthe=np.sin(theta/2)
    sinpsi=np.sin(psi/2)
    
    z=np.array([
        [cosphi*costhe*cospsi+sinphi*sinthe*sinpsi],
        [sinphi*costhe*cospsi-cosphi*sinthe*sinpsi],
        [cosphi*sinthe*cospsi+sinphi*costhe*sinpsi],
        [cosphi*costhe*sinpsi-sinphi*sinthe*cospsi]
    ])
    return z

if True:
    sensor=np.genfromtxt('sensor.csv',delimiter=',')
else:
    accelMat=scipy.io.loadmat('ArsAccel.mat')
    gyroMat=scipy.io.loadmat('ArsGyro.mat')
    ax,ay,az=accelMat['fx'],accelMat['fy'],accelMat['fz']
    gx,gy,gz=gyroMat['wx'],gyroMat['wy'],gyroMat['wz']
    sensor=np.hstack([ax,ay,az,gx,gy,gz])

    

kf=KalmanFilter(4,4)
kf.H=np.eye(4)
kf.Q=np.diag([.0016610794813841504,
 1.5349546583854867e-07,
 6.004063634161956e-08,
 2.7761794892813822e-08])
kf.R=np.diag([10,10,10,10])*10
kf.x=np.array([1,0,0,0]).reshape((4,1))
kf.P=np.eye(4)

biasGx=-0.00878016
biasGy= 0.00555219
biasGz=-0.00088954

init_acc,_=R.align_vectors(np.array([0,0,1]).reshape(1,3),sensor[0][:3].reshape(1,3))


phi,theta,psi=0,0,0 
dt=0.01
g=9.81
phis=np.array([phi])
thetas=np.array([theta])
psis=np.array([psi])
phi_gyro,theta_gyro,psi_gyro=0,0,0
phis_gryo=[0]
thetas_gyro=[0]
psis_gyro=[0]
sensor_value=[]

for i in range(len(sensor)):
    Ax,Ay,Az,Gx,Gy,Gz=sensor[i]
    Ax,Ay,Az=init_acc.apply([Ax,Ay,Az])
    Ax,Ay,Az=np.array([Ax,Ay,Az])/np.linalg.norm([Ax,Ay,Az])*9.81
    Gx=Gx-biasGx
    Gy=Gy-biasGy
    Gz=Gz-biasGz
    Gx,Gy,Gz=np.deg2rad([Gx,Gy,Gz])

    kf.A=np.eye(4)+dt*1/2*np.array([
        [0,-Gx,-Gy,-Gz],
        [Gx,0,Gz,-Gy],
        [Gy,-Gz,0,Gx],
        [Gz,Gy,-Gx,0]
    ])
    phi_gyro,theta_gyro,psi_gyro=np.array([[1,np.sin(phi_gyro)*np.tan(theta_gyro),np.cos(phi_gyro)*np.tan(theta_gyro)],
                        [0,np.cos(phi_gyro),-np.sin(phi_gyro)],
                        [0,np.sin(phi_gyro)/np.cos(theta_gyro),np.cos(phi_gyro)/np.cos(theta_gyro)]])@np.vstack([Gx,Gy,Gz])[:,0]*dt+np.array([phi_gyro,theta_gyro,psi_gyro])
    phis_gryo.append(phi_gyro)
    thetas_gyro.append(theta_gyro)
    psis_gyro.append(psi_gyro)
    
    theta=np.arctan(Ay/Az)
    phi=np.arctan(Ax/np.sqrt(Ay**2+Az**2))
    psi=0
    
    cosphi=np.cos(phi/2)
    costhe=np.cos(theta/2)
    cospsi=np.cos(psi/2)
    sinphi=np.sin(phi/2)
    sinthe=np.sin(theta/2)
    sinpsi=np.sin(psi/2)
    
    z=np.array([
        [cosphi*costhe*cospsi+sinphi*sinthe*sinpsi],
        [sinphi*costhe*cospsi-cosphi*sinthe*sinpsi],
        [cosphi*sinthe*cospsi+sinphi*costhe*sinpsi],
        [cosphi*costhe*sinpsi-sinphi*sinthe*cospsi]
    ])
    x=kf.run(z)
    phi=np.arctan2(2*(x[2]*x[3]+x[0]*x[1]),1-2*(x[1]**2+x[2]**2))
    theta=-np.arcsin(2*(x[1]*x[3]-x[0]*x[2]))
    psi=np.arctan2(2*(x[1]*x[2]+x[0]*x[3]),1-2*(x[2]**2+x[3]**2))
    phis=np.append(phis,phi,axis=0)
    thetas=np.append(thetas,theta,axis=0)
    psis=np.append(psis,psi,axis=0)

    
euler=np.vstack([phis,thetas,psis]) 
gyro=np.vstack([phis_gryo,thetas_gyro,psis_gyro])

euler=np.rad2deg(euler[:,1:])
gyro=np.rad2deg(gyro[:,1:])

plt.figure(1)
plt.subplot(3,1,1)
plt.title('phi')
plt.plot(euler[0])
plt.plot(gyro[0])
plt.legend(['kf','meas'])
plt.subplot(3,1,2)
plt.title('theta')
plt.plot(euler[1])
plt.plot(gyro[1])
plt.subplot(3,1,3)
plt.title('psi')
plt.plot(gyro[2])
plt.plot(euler[2])
plt.show()