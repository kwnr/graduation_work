from read_sensor import MPU6050
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

sensor=MPU6050()
phi_gyro,theta_gyro,psi_gyro=0,0,0
phis_gryo=[0]
thetas_gyro=[0]
psis_gyro=[0]
dt=0.1
while len(psis_gyro)<60/dt:
        Ax,Ay,Az,Gx,Gy,Gz=sensor.read_value()
        phi_gyro,theta_gyro,psi_gyro=np.array([[1,np.sin(phi_gyro)*np.tan(theta_gyro),np.cos(phi_gyro)*np.tan(theta_gyro)],
                        [0,np.cos(phi_gyro),-np.sin(phi_gyro)],
                        [0,np.sin(phi_gyro)/np.cos(theta_gyro),np.cos(phi_gyro)/np.cos(theta_gyro)]])@np.array([Gx,Gy,Gz]).T
        phis_gryo.append(phis_gryo[-1]+phi_gyro*dt)
        thetas_gyro.append(thetas_gyro[-1]+theta_gyro*dt)
        psis_gyro.append(psis_gyro[-1]+psi_gyro*dt)
        print(f'phi: {phi_gyro}, theta: {theta_gyro}, psi:{psi_gyro}')
        sleep(dt)
        

plt.figure(1)
plt.subplot(3,1,1)
plt.title('phi')
plt.plot(phis_gryo)
plt.subplot(3,1,2)
plt.title('theta')
plt.plot(thetas_gyro)
plt.subplot(3,1,3)
plt.title('psi')
plt.plot(psis_gyro)
plt.show()