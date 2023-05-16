import time
from pyrf24.rf24 import *
import threading
ADDRESS_LEN=5
BIND_CHANNEL=9
class symaTX(threading.Thread):
    
    def __init__(self,addr):
        super().__init__()
        self.rx_tx_addr=[0]*ADDRESS_LEN
        self.address=int(addr,16)
        self.pipes=[0xafaeadacab, self.address]
        for i in range(ADDRESS_LEN):
            self.rx_tx_addr[ADDRESS_LEN-1-i]=ord(chr(int(addr[2*i:2*i+2],16)))
        self.chans=self.set_channel()
        self.ch=0
        self.chans_count=len(self.chans)
        self.packet_size=10
        self.packet=[0]*self.packet_size
        self.running=True
        self.bind=False
        self.bind_prev=False
        self.direction=[0,0,0,0]    #[throttle,
        print(f'Address: {self.rx_tx_addr}')
        print(f'Pipes: {self.pipes}')
        print(f'Channels: {self.chans}')
        
        try:
            self.radio = RF24(RPI_V2_GPIO_P1_22, RPI_V2_GPIO_P1_24, BCM2835_SPI_SPEED_8MHZ)
            self.setup()
            pass
        except:
            print ("Error. No nrf module")
            exit(0)
            
    def setup(self):
        self.radio.begin()
        self.radio.setChannel(BIND_CHANNEL)
        self.radio.setDataRate(RF24_250KBPS)
        self.radio.setCRCLength(RF24_CRC_16)
        self.radio.setPayloadSize(self.packet_size)
        self.radio.setAutoAck(0)
        self.radio.setAddressWidth(ADDRESS_LEN)
        self.radio.openWritingPipe(self.pipes[0])
        
    def set_channel(self):
        tx0=self.addr[0]
        start1=[0x0a,0x1a,0x2a,0x3a]
        start2=[0x2a,0x0a,0x42,0x22]
        start3=[0x1a,0x3a,0x12,0x32]
        
        self.current_channel=0
        self.channel_counter=0
        
        tx0&=0x1f
        if tx0<0x10:
            if tx0==0x06:
                tx0=0x07
            for i in range(4):
                self.channels[i]=start1[i]+tx0
        elif tx0<0x18:
            for i in range(4):
                self.channels[i]=start2[i]+(tx0&0x07)
            if tx0==0x16:
                self.channels[0]+=0x01
                self.channels[1]+=0x01
                
        elif tx0<0x1e:
            for i in range(4):
                self.channels[i]=start3[i]+(tx0&0x07)
        elif tx0==0x1e:
            self.channels= [0x21, 0x41, 0x18, 0x38]
        else:
            self.channels[0x21, 0x41, 0x19, 0x39]
            
                
    
    def build_packet(self):
        if self.bind:
            self.packet[0] = self.rx_tx_addr[4]
            self.packet[1] = self.rx_tx_addr[3]
            self.packet[2] = self.rx_tx_addr[2]
            self.packet[3] = self.rx_tx_addr[1]
            self.packet[4] = self.rx_tx_addr[0]
            self.packet[5] = 0xaa
            self.packet[6] = 0xaa
            self.packet[7] = 0xaa
            self.packet[8] = 0x00
        else:
            self.packet[0]=self.direction[0]
            self.packet[1]=self.direction[1]
            self.packet[2]=self.direction[2]
            self.packet[3]=self.direction[3]
            self.packet[4]=0x00
            self.packet[5]=(self.data[1]>>2)|0xc0
            self.packet[6]=(self.data[2]>>2)
            self.packet[7]=(self.data>>2)
            self.packet[8]=0x00

        self.data[9]=self.checksum(self.data)
    
    def run(self):
        while self.running:
            self.build_packet()
            if not self.bind and self.bind_prev:
                self.radio.openWritingPipe(self.pipes[1])
            if self.bind and not self.bind_prev:
                self.radio.setChannel(BIND_CHANNEL)
                self.radio.openWritingPipe(self.pipes[0])
            else:
                self.ch+=1
                self.ch=self.ch%self.chans_count
                self.radio.setChannel(self.chans[self.ch])
            self.radio.write(bytearray(self.packet))
            self.bind_prev=self.bind
            time.sleep(0.000001)
            
    def quit(self):
        self.running=False
    
    def checksum(self, packet):
        csum=packet[0]
        for i in range(1,9):
            csum=csum^packet[i]
        return (csum+0x55)%256
    
class controller(threading.Thread):
    def __init__(self,address):
        self.syma=symaTX(address)
        self.syma.start()
        
    def run(self):
        while(True):
            input='asdf'
            self.syma.direction[0]=10
    


if __name__=="__main__":
    tx=symaTX('a20009890f')