from pyrf24.rf24 import *
import threading
import time

class symaRX(threading.Thread):
    def __init__(self):
        super().__init__()
        self.bound=False
        self.channels=[0x4b,0x30,0x40,0x20]
        self.current_channel=0
        
        self.addr=[0xab,0xac,0xad,0xae,0xaf]
        self.address=int(''.join([hex(self.addr[3-i])[2:] for i in range(5)]),16)
        self.available=False

        self.packet=[0]*10
        
    def to_pointer(self,hex_list):
        return int(''.join([hex(hex_list[3-i])[2:] for i in range(5)]),16)
        
    def init(self,radio:RF24):
        radio.setChannel(self.channels[0])
        radio.openReadingPipe(1,self.to_pointer(self.addr))
        radio.flush_rx()
        radio.startListening()
        self.prev_rx_time=time.time()*1000
    
    def run(self,radio:RF24):
        self.curr_rx_time=time.time()*1000
        radio.setChannel(self.channels[self.current_channel])
        radio.openReadingPipe(1,self.to_pointer(self.addr))
        radio.startListening()
        if not radio.available():
            if self.curr_rx_time-self.prev_rx_time>16:
                self.current_channel+=1
                self.current_channel%=4
                radio.setChannel(self.channels[self.current_channel])
                self.prev_rx_time=self.curr_rx_time
        else:
            self.data=radio.read(10)
            if not self.bound:
                if self.checksum(self.data)==self.data[9] and self.data[6]==0xaa:
                    for i in range(5):
                        self.addr[i]=self.data[4-i]
                    self.set_channel()
                    radio.setChannel(self.channels[self.current_channel])
                    radio.openReadingPipe(1,self.to_pointer(self.addr))
                    self.bound=True
            self.available=True
            
    def read(self):
        for i in range(10):
            self.packet[i]=self.data[i]
        self.available=False
        return self.packet
        
    def checksum(self,packet):
        sum=packet[0]
        for i in range(9):
            sum^=packet[i]
        sum+=0x55
        return sum
    
    def set_channel(self):
        tx0=self.addr[0]
        num_rf_channels=4
        start1 = [0x0a, 0x1a, 0x2a, 0x3a]
        start2 = [0x2a, 0x0a, 0x42, 0x22]
        start3 = [0x1a, 0x3a, 0x12, 0x32]
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

radio=RF24(25,0)
rx=symaRX()

radio.begin()
radio.setAutoAck(False)
radio.setAddressWidth(5)
radio.setRetries(15,15)
radio.setDataRate(RF24_250KBPS)
radio.setPALevel(RF24_PA_HIGH)
radio.setPayloadSize(10)

rx.init(radio)
packet=[]

while True:
    rx.run(radio)
    if rx.available:
        rx.read()
    print(rx.packet)