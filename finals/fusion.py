from read_sensor import MPU6050
from kalman_filter import KalmanFilter
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

sensor = MPU6050()
kf = KalmanFilter(4, 4)
kf.H = np.eye(4)
kf.Q = np.eye(4) * 0.0001
kf.R = np.eye(4) * 10
kf.x = np.array([0, 0, 0, 1]).reshape(4, 1)
kf.P = np.eye(4)

biasGx = -0.00878016
biasGy = 0.00555219
biasGz = -0.00088954

phi, theta, psi = 0, 0, 0
dt = 0.1
g = 9.81
phis = [phi]
thetas = [theta]
psis = [psi]
phi_gyro, theta_gyro, psi_gyro = 0, 0, 0
phis_gryo = [0]
thetas_gyro = [0]
psis_gyro = [0]
acc_value = []
sensor_value = []
kf_value = []
gyro_value = []

while len(phis) < 600:
    Ax, Ay, Az, Gx, Gy, Gz = sensor.read_value()
    sensor_value.append([Ax, Ay, Az, Gx, Gy, Gz])

    kf.A = np.eye(4) + dt * 1 / 2 * np.array(
        [[0, -Gx, -Gy, -Gz], [Gx, 0, Gz, -Gy], [Gy, -Gz, 0, Gx], [Gz, Gy, -Gx, 0]]
    )
    phi_gyro, theta_gyro, psi_gyro = np.array(
        [
            [
                1,
                np.sin(phi_gyro) * np.tan(theta_gyro),
                np.cos(phi_gyro) * np.tan(theta_gyro),
            ],
            [0, np.cos(phi_gyro), -np.sin(phi_gyro)],
            [
                0,
                np.sin(phi_gyro) / np.cos(theta_gyro),
                np.cos(phi_gyro) / np.cos(theta_gyro),
            ],
        ]
    ) @ np.vstack([Gx, Gy, Gz])[:, 0] * dt + np.array([phi_gyro, theta_gyro, psi_gyro])
    gyro_value.append([phi_gyro, theta_gyro, psi_gyro])

    theta = np.arctan(Ay / Az)
    phi = np.arctan(Ax / np.sqrt(Ay**2 + Az**2))
    psi = 0
    acc_value.append([theta, phi, psi])

    cosphi = np.cos(phi / 2)
    costhe = np.cos(theta / 2)
    cospsi = np.cos(psi / 2)
    sinphi = np.sin(phi / 2)
    sinthe = np.sin(theta / 2)
    sinpsi = np.sin(psi / 2)

    z = np.array(
        [
            [cosphi * costhe * cospsi + sinphi * sinthe * sinpsi],
            [sinphi * costhe * cospsi - cosphi * sinthe * sinpsi],
            [cosphi * sinthe * cospsi + sinphi * costhe * sinpsi],
            [cosphi * costhe * sinpsi - sinphi * sinthe * cospsi],
        ]
    )
    x = kf.run(z)
    phi = np.arctan2(2 * (x[2] * x[3] + x[0] * x[1]), 1 - 2 * (x[1] ** 2 + x[2] ** 2))
    theta = -np.arcsin(2 * (x[1] * x[3] - x[0] * x[2]))
    psi = np.arctan2(2 * (x[1] * x[2] + x[0] * x[3]), 1 - 2 * (x[2] ** 2 + x[3] ** 2))
    kf_value.append([phi, theta, psi])
    print(f"{phi}  {theta}  {psi}")

phi_euler = [kf_value[i][0][0] for i in range(len(kf_value))]
theta_euler = [kf_value[i][1][0] for i in range(len(kf_value))]
psi_euler = [kf_value[i][2][0] for i in range(len(kf_value))]
phi_gyro = [gyro_value[i][0] for i in range(len(gyro_value))]
theta_gyro = [gyro_value[i][1] for i in range(len(gyro_value))]
psi_gyro = [gyro_value[i][2] for i in range(len(gyro_value))]

plt.figure(1)
plt.subplot(3, 1, 1)
plt.title("phi")
plt.plot(phi_euler)
plt.plot(phi_gyro)
plt.legend(["kf", "gyro"])
plt.subplot(3, 1, 2)
plt.title("theta")
plt.plot(theta_euler)
plt.plot(theta_gyro)
plt.legend(["kf", "gyro"])
plt.subplot(3, 1, 3)
plt.title("psi")
plt.plot(psi_euler)
plt.plot(psi_gyro)
plt.legend(["kf", "gyro"])
plt.savefig("fusion.png")
