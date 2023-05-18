import time
from pyrf24.rf24 import *
import threading

class symaTX(threading.Thread):
    def __init__(self):
        super().__init__()
        
        self.bound=False
        self.channels=[0x4b,0x30,0x40,0x20]
        self.current_channel=0
        self.channel_counter=0
        self.addr=[0xab,0xac,0xad,0xae,0xaf]
        
        self.throttle=0
        self.pitch=0
        self.yaw=0
        self.roll=0
        self.data=[0]*10
        
        self.radio=RF24(25,0)
        
        self.running=True
        
    def init2(self,tx_addr):
        self.radio.stopListening()
        self.radio.setChannel(self.channels[0])
        self.radio.openWritingPipe(bytes(self.addr))
        for i in range(5):
            self.data[i]=tx_addr[i]
        for i in range(3):
            self.data[i+5]=0xaa
        self.data[8]=0
        self.data[9]=self.checksum(self.data)
        print(self.data)
        tstart=time.monotonic()
        timeout=5
        while time.monotonic()-tstart<timeout:
            self.transmit()
            print(self.data)
        
        self.addr=tx_addr
        self.build_packet()
        self.set_channel()
        self.radio.openWritingPipe(bytes(self.addr))
        self.bound=True
        
    def build_packet(self):
        self.data[0]=self.throttle
        self.data[1]=self.pitch
        self.data[2]=self.roll
        self.data[3]=self.yaw
        self.data[4]=0
        self.data[5]=(self.data[1]>>2)|0xc0
        self.data[6]=(self.data[2]>>2)
        self.data[7]=(self.data[3]>>2)
        self.data[8]=0x00
        self.data[9]=self.checksum(self.data)
            
    def transmit(self):
        self.radio.setChannel(self.channels[self.current_channel])
        self.radio.write(bytes(self.data),False)
        self.channel_counter+=1
        if self.channel_counter%2==0:
            self.current_channel+=1
            self.current_channel=self.current_channel%4
            self.channel_counter=0
            
    
            
    
    def set_channel(self):
        tx0=self.addr[0]
        num_rf_channels=4
        start1 = [0x0a, 0x1a, 0x2a, 0x3a]
        start2 = [0x2a, 0x0a, 0x42, 0x22]
        start3 = [0x1a, 0x3a, 0x12, 0x32]
        tx0&=0x1f
        if tx0 < 0x10:
            if tx0 == 6:
                tx0 = 7
            for i in range(num_rf_channels):
                self.channels[i] = start1[i] + tx0
        elif tx0 < 0x18:
            for i in range(num_rf_channels):
                self.channels[i] = start2[i] + (tx0 & 0x07)
            if tx0 == 0x16:
                self.channels[0] += 1
                self.channels[1] += 1
        elif tx0 < 0x1E:
            for i in range(num_rf_channels):
                self.channels[i] = start3[i] + (tx0 & 0x07)
        elif tx0 == 0x1E:
            self.channels=[0x21, 0x41, 0x18, 0x38]
        else:
            self.channels=[0x21, 0x41, 0x19, 0x39]
    
    def checksum(self,packet):
        sum=packet[0]
        for i in range(1,9):
            sum^=packet[i]
        sum+=0x55
        sum%=256
        return sum
    
    def run(self):
        while self.running:
            if self.bound:
                self.build_packet()
                print(self.data)
                self.transmit()
        
    
class controller(threading.Thread):
    def __init__(self):
        super().__init__()
        self.tx=symaTX()
        self.tx.radio.begin()
        self.tx.radio.setAutoAck(False)
        self.tx.radio.setAddressWidth(5)
        self.tx.radio.setRetries(15,15)
        self.tx.radio.setDataRate(RF24_250KBPS)
        self.tx.radio.setPALevel(RF24_PA_HIGH)
        self.tx.radio.setPayloadSize(10)
        time.wait(0.015)
    
        self.tx.init2([161, 105, 1, 104, 204])
        self.tx.start()
        
if __name__=="__main__":
    tx=symaTX()
    tx=symaTX()
    tx.radio.begin()
    tx.radio.setAutoAck(False)
    tx.radio.setAddressWidth(5)
    tx.radio.setRetries(15,15)
    tx.radio.setDataRate(RF24_250KBPS)
    tx.radio.setPALevel(RF24_PA_HIGH)
    tx.radio.setPayloadSize(10)
    time.sleep(0.015)

    tx.init2([161, 105, 1, 104, 204])
    
    tx.run()
    
            
    
    
    