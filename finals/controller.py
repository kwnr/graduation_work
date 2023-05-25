import numpy as np
import control

g=9.81
m=2
Iz=1/2*m*0.1**2
Ix=1/12*m*(3*0.1**2+0.1**2)
Iy=Ix
A=np.array([
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

B=np.array([
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
K,S,E=control.lqr(A,B,Q,R)
print(K)
