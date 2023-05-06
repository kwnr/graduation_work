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
phi_gyro,theta_gyro,psi_gyro=0,0,0
phis_gryo=[0]
thetas_gyro=[0]
psis_gyro=[0]

while len(phis)<600:
    Ax,Ay,Az,Gx,Gy,Gz=sensor.read_value()
    kf.A=np.eye(4)+dt*1/2*np.array([
        [Gx,0,Gz,-Gy],
        [Gy,-Gz,0,Gx],
        [Gz,Gy,-Gz,0],
        [0,-Gx,-Gy,-Gz]
    ])
    phi_gyro,theta_gyro,psi_gyro=np.array([[1,np.sin(phi_gyro)*np.tan(theta_gyro),np.cos(phi_gyro)*np.tan(theta_gyro)],
                        [0,np.cos(phi_gyro),-np.sin(phi_gyro)],
                        [0,np.sin(phi_gyro)/np.cos(theta_gyro),np.cos(phi_gyro)/np.cos(theta_gyro)]])@np.array([Gx,Gy,Gz]).T
    phis_gryo.append(phis_gryo[-1]+phi_gyro*dt)
    thetas_gyro.append(thetas_gyro[-1]+theta_gyro*dt)
    psis_gyro.append(psis_gyro[-1]+psi_gyro*dt)
    
    theta=np.arcsin(Ax/g)
    phi=np.arcsin(-Ay/(g*np.cos(theta)))
    z=R.from_euler('zyx',[theta,phi,0])
    x=kf.run(z.as_quat())
    phi,theta,psi=R.from_quat(x).as_euler('zyx',degrees=True)
    phis.append(phi)
    thetas.append(theta)
    psis.append(psi)
    print(f'{phi}  {theta}  {psi}')
    sleep(dt)
    
euler=np.concatenate((phis,thetas,psis),axis=1)
gyro=np.concatenate((phis_gryo,thetas_gyro,psis_gyro),axis=1)
np.savetxt("euler.csv",euler,delimiter=',')
np.savetxt("gyro.csv",gyro,delimiter=',')

plt.figure(1)
plt.subplot(3,1,1)
plt.title('phi')
plt.plot(euler[:,0])
plt.plot(gyro[:,0])
plt.subplot(3,1,2)
plt.title('theta')
plt.plot(euler[:,1])
plt.plot(gyro[:,1])
plt.subplot(3,1,3)
plt.title('psi')
plt.plot(euler[:,2])
plt.plot(gyro[:,2])
plt.show()