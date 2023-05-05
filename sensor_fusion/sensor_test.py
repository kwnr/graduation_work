from read_sensor import MPU6050
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

sensor=MPU6050()
phi,theta,psi=0,0,0
phis=[0]
thetas=[0]
psis=[0]
dt=0.1;
while len(psis)<60/dt:
        Ax,Ay,Az,Gx,Gy,Gz=sensor.read_value()
        phi,theta,psi=np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
                        [0,np.cos(phi),-np.sin(phi)],
                        [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])@np.array([Gx,Gy,Gz]).T
        phis.append(phis[-1]+phi*dt)
        thetas.append(thetas[-1]+theta*dt)
        psis.append(psis[-1]+psi*dt)
        print(f'phi: {phi}, theta: {theta}, psi:{psi}')
        sleep(dt)
        

plt.figure(1)
plt.subplot(3,1,1)
plt.title('phi')
plt.plot(phis)
plt.subplot(3,1,2)
plt.title('theta')
plt.plot(thetas)
plt.subplot(3,1,3)
plt.title('psi')
plt.plot(psis)
plt.show()