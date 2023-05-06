from read_sensor import MPU6050
from kalman_filter import KalmanFilter
from scipy.spatial.transform import Rotation as R
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

sensor=MPU6050()
kf=KalmanFilter(4,4)
kf.H=np.eye(4)
kf.Q=np.eye(4)*0.0001
kf.R=np.eye(4)*10
kf.x=np.array([0,0,0,1]).T
kf.P=np.eye(4)

phi,theta,psi=0,0,0
dt=0.1
g=9.81
phis=[phi]
thetas=[theta]
psis=[psi]

while len(phis)<600:
    Ax,Ay,Az,Gx,Gy,Gz=sensor.read_value()
    kf.A=np.eye(4)+dt*1/2*np.array([
        [Gx,0,Gz,-Gy],
        [Gy,-Gz,0,Gx],
        [Gz,Gy,-Gz,0],
        [0,-Gx,-Gy,-Gz]
    ])
    theta=np.arcsin(Ax/g)
    phi=np.arcsin(-Ay/(g*np.cos(theta)))
    z=R.from_euler('zyx',[theta,phi,0])
    x=kf.run(z)
    phi,theta,psi=R.from_quat(x).as_euler('zyx',degrees=True)
    phis.append(phi)
    thetas.append(theta)
    psis.append(psi)
    sleep(dt)
    
euler=np.concatenate((phis,thetas,psis))
with open("euler.txt",'w') as f:
    f.write(euler)

plt.figure(1)
plt.subplot(3,1,1)
plt.title('phi')
plt.plot(euler[0,:])
plt.subplot(3,1,2)
plt.title('theta')
plt.plot(euler[1,:])
plt.subplot(3,1,3)
plt.title('psi')
plt.plot(euler[2,:])
plt.show()