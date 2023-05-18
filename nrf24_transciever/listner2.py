from pyrf24.rf24 import *
import time
import copy

class symaRX():
    def __init__(self):
        self.bound=False
        self.channels=[0x4b,0x30,0x40,0x20]
        self.current_channel=0
        self.addr=[0xab,0xac,0xad,0xae,0xaf]
        self.available=False
        self.radio=RF24(25,0)
        
    def init2(self):
        self.radio.setChannel(self.channels[0])
        self.radio.openReadingPipe(1,bytes(self.addr))
        self.radio.flush_rx()
        self.radio.startListening()
        self.prev_rx_time=time.monotonic()*1000
        
    def run(self):
        self.curr_rx_time=time.monotonic()*1000
        self.radio.setChannel(self.channels[self.current_channel])
        self.radio.openReadingPipe(1,bytes(self.addr))
        self.radio.startListening()            
        
        if not self.radio.available():
            if self.curr_rx_time-self.prev_rx_time>16:
                self.current_channel+=1
                self.current_channel%=4
                self.radio.setChannel(self.channels[self.current_channel])
                self.prev_rx_time=self.curr_rx_time
                
        else:
            self.data=self.radio.read(10)
            if not self.bound:
                if self.checksum(self.data)==self.data[9] and self.data[5]==0xaa and self.data[6]==0xaa:
                    for i in range(5):
                        self.addr[i]=self.data[4-i]
                    self.set_channel()
                    self.radio.setChannel(self.channels[self.current_channel])
                    self.radio.openReadingPipe(1,bytearray(self.addr))
                    self.bound=True    
            self.available=True
            
            
            
    def read(self):
        self.available=False
        return self.data
    
    def checksum(self,packet):
        sum=packet[0]
        for i in range(1,9):
            sum^=packet[i]
        sum+=0x55
        sum%=256
        return sum
    
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
            
            
if __name__=="__main__":
    rx=symaRX()
    
    
    rx.radio.begin()
    rx.radio.setAutoAck(False)
    rx.radio.setAddressWidth(5)
    rx.radio.setRetries(15,15)
    rx.radio.setDataRate(RF24_250KBPS)
    rx.radio.setPALevel(RF24_PA_HIGH)
    rx.radio.setPayloadSize(10)
    time.sleep(0.012)
    rx.init2()
    while(True):
        rx.run()
        packet=0
        if rx.available:
            packet=rx.read()
            data=[i for i in packet]
            print(data, rx.checksum(data))
