import numpy as np

class KalmanFilter():
    def __init__(self, input_size, output_size) -> None:
        self.A=np.zeros(input_size)
        self.H=np.zeros((input_size,output_size))
        self.Q=np.zeros(input_size)
        self.R=np.zeros(output_size)
        self.P=np.zeros(input_size)
        self.x=np.zeros((input_size,output_size))
        
    def run(self,z):
        x_pred=self.A@self.x
        P_pred=self.A@self.P@self.A.T+self.Q
        K=P_pred@self.H.T@np.linalg(self.H@P_pred@self.H.T+self.R)
        self.x=x_pred+K@(z-self.H@x_pred)
        self.P=P_pred-K@self.H@P_pred
        return self.x