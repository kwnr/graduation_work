from read_sensor import MPU6050
import numpy as np
import matplotlib.pyplot as plt

sensor=MPU6050()
phi,theta,psi=0,0,0
phis=[0]
thetas=[0]
psis=[0]
dt=0.1;
while len(psis)<60/dt:
        Ax,Ay,Az,Gx,Gy,Gz=MPU6050.read_value()
        psi,theta,psi=np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
                        [0,np.cos(phi),-np.sin(phi)],
                        [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])@np.array([Gx,Gy,Gz]).T
        phis.append(phis[-1]+psi*dt)
        thetas.append(thetas[-1]+psi*dt)
        psis.append(psis[-1]+psi*dt)
        

plt.figure(1)
plt.subplot(3,1,1)
plt.title('phi')
plt.plot(phis)
plt.subplot(3,1,2)
plt.title('theta')
plt.plot(theta)
plt.subplot(3,1,3)
plt.title('psi')
plt.plot(psis)
plt.show()