from read_sensor import MPU6050
from kalman_filter import KalmanFilter
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
sensor_value=[]

while len(phis)<600:
    Ax,Ay,Az,Gx,Gy,Gz=sensor.read_value()
    Gx,Gy,Gz=np.deg2rad([Gx,Gy,Gz])
    sensor_value.append([Ax,Ay,Az,Gx,Gy,Gz])
    kf.A=np.eye(4)+dt*1/2*np.array([
        [0,-Gx,-Gy,-Gz],
        [Gx,0,Gz,-Gy],
        [Gy,-Gz,0,Gx],
        [Gz,Gy,-Gx,0]
    ])
    phi_gyro,theta_gyro,psi_gyro=np.array([[1,np.sin(phi_gyro)*np.tan(theta_gyro),np.cos(phi_gyro)*np.tan(theta_gyro)],
                        [0,np.cos(phi_gyro),-np.sin(phi_gyro)],
                        [0,np.sin(phi_gyro)/np.cos(theta_gyro),np.cos(phi_gyro)/np.cos(theta_gyro)]])@np.array([Gx,Gy,Gz]).T
    phis_gryo.append(phis_gryo[-1]+phi_gyro*dt)
    thetas_gyro.append(thetas_gyro[-1]+theta_gyro*dt)
    psis_gyro.append(psis_gyro[-1]+psi_gyro*dt)
    
    theta=np.arcsin(Ax/g)
    phi=np.arcsin(-Ay/(g*np.cos(theta)))
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
    phis.append(phi)
    thetas.append(theta)
    psis.append(psi)
    print(f'{phi}  {theta}  {psi}')
    sleep(dt)
    
euler=np.vstack([phis,thetas,psis]) 
gyro=np.vstack([phis_gryo,thetas_gyro,psis_gyro])
sensor_val=np.array(sensor_value)
np.savetxt("euler.csv",euler,delimiter=',')
np.savetxt("gyro.csv",gyro,delimiter=',')
np.savetxt("sensor.csv",sensor_value,delimiter=',')

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