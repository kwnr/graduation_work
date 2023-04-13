import numpy as np


class KalmanFilter():
    def __init__(self) -> None:
        super().__init__()
        self.dt=1./30.
        self.dt2=1/2*self.dt**2
        self.A_0=np.array([
            [1,0,0,self.dt,0,0,self.dt2,0,0],
            [0,1,0,0,self.dt,0,0,self.dt2,0],
            [0,0,1,0,0,self.dt,0,0,self.dt2],
            [0,0,0,1,0,0,self.dt,0,0],
            [0,0,0,0,1,0,0,self.dt,0],
            [0,0,0,0,0,1,0,0,self.dt],
            [0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,1]
        ])
        self.A=np.block([
            [self.A_0,np.zeros(self.A_0.shape)],
            [np.zeros(self.A_0.shape),self.A_0]
        ])
        self.H=np.block([
            [np.eye(3),np.zeros((3,15))],
            [np.zeros((3,9)),np.eye(3),np.zeros((3,6))]
        ])
        self.Q=np.eye(self.A.shape[0])*5qe-4
        self.R=np.eye(self.H.shape[0])*1e-5
        self.P_prev=np.eye(self.A.shape[0])*0.01
        self.X_pred_prev=np.zeros((18,1))
        self.X_est=None
    def update(self,z):
        self.X_pred=np.dot(self.A,self.X_pred_prev)
        self.P_pred=np.dot(np.dot(self.A,self.P_prev),self.A.T)+self.Q
        self.K=self.P_pred@self.H.T@np.linalg.inv(self.H@self.P_pred@self.H.T+self.R)
        self.X_est=self.X_pred+self.K.dot(z-self.H.dot(self.X_pred))
        self.P_prev=self.P_pred-np.dot(np.dot(self.K,self.H),self.P_pred)
        return self.H.dot(self.X_est)
        
        
        
if __name__=='__main__':
    from aruco_board_detection_fn import BoardDetection
    import matplotlib.pyplot as plt
    import cv2
    import time
    import pandas as pd
    kf=KalmanFilter()
    cap=cv2.VideoCapture(1)
    start_time=time.time()
    df=pd.DataFrame(columns=['t','x','y','z','roll','pitch','yaw','x_est','y_est','z_est','roll_est','pitch_est','yaw_est'])
    df.reset_index(inplace=True,drop=True)
    i=0
    bd=BoardDetection()
    while cap.isOpened():
        _,img=cap.read()  
        X=bd.pipeline(img)
        if X is not None:
            X_est=kf.update(X)
            t=time.time()-start_time
            x,y,z,pitch,yaw,roll=X[:,0]
            x_est,y_est,z_est,pitch_est,yaw_est,roll_est=X_est[:,0] 
            df=pd.concat((df,pd.DataFrame({"i":0,'t':t,'x':x,'y':y,'z':z,'roll':roll,'pitch':pitch,'yaw':yaw,'x_est':x_est,'y_est':y_est,'z_est':z_est,'roll_est':roll_est,'pitch_est':pitch_est,'yaw_est':yaw_est},index=['i'])),ignore_index=True)
        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(df.t,df.roll,'r',df.t,df.roll_est,'b')
    plt.legend(['measured','filtered'])
    plt.title('roll')
    plt.subplot(3,1,2)
    plt.plot(df.t,df.pitch,'r',df.t,df.pitch_est,'b')
    plt.legend(['measured','filtered'])
    plt.title('pitch')
    plt.subplot(3,1,3)
    plt.plot(df.t,df.yaw,'r',df.t,df.yaw_est,'b')
    plt.legend(['measured','filtered'])
    plt.title('yaw')
    plt.figure(2)
    plt.subplot(3,1,1)
    plt.plot(df.t,df.x,'r',df.t,df.x_est,'b')
    plt.legend(['measured','filtered'])
    plt.title('x')
    plt.subplot(3,1,2)
    plt.plot(df.t,df.y,'r',df.t,df.y_est,'b')
    plt.legend(['measured','filtered'])
    plt.title('y')
    plt.subplot(3,1,3)
    plt.plot(df.t,df.z,'r',df.t,df.z_est,'b')
    plt.legend(['measured','filtered'])
    plt.title('z')
    plt.show()
    cap.release()
    
    cv2.destroyAllWindows()