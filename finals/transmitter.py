import time
from pyrf24.rf24 import *
import threading


class symaTX(threading.Thread):
    def __init__(self):
        super().__init__()

        self.bound = False
        self.channels = [22, 38, 54, 70]
        self.chan_len = len(self.channels)
        self.current_channel = 0
        self.channel_counter = 0
        self.addr = [0xAB, 0xAC, 0xAD, 0xAE, 0xAF]

        self.throttle = 0
        self.pitch = 0
        self.yaw = 0
        self.roll = 0
        self.data = [0] * 10

        self.radio = RF24(25, 0)

        self.running = True

    def init2(self):
        self.radio.stopListening()
        self.radio.setChannel(self.channels[0])
        self.radio.openWritingPipe(bytes(self.addr))
        for i in range(5):
            self.data[i] = self.addr[4 - i]
        for i in range(3):
            self.data[i + 5] = 0xAA
        self.data[8] = 0
        self.data[9] = self.checksum(self.data)
        tstart = time.monotonic()
        while time.monotonic() - tstart < 1.4:
            self.bind()
            # print(self.data)
        self.build_packet()
        self.radio.openWritingPipe(bytes(self.addr))
        self.bound = True

    def build_packet(self):
        self.data[0] = self.throttle
        self.data[1] = self.pitch
        self.data[2] = self.yaw
        self.data[3] = self.roll
        self.data[4] = 0
        self.data[5] = 70
        self.data[6] = 32
        self.data[7] = 19
        self.data[8] = 129
        self.data[9] = self.checksum(self.data)

    def bind(self):
        self.radio.setChannel(self.channels[self.current_channel])
        self.radio.write(bytes(self.data), False)
        self.channel_counter += 1
        if self.channel_counter % 2 == 0:
            self.current_channel += 1
            self.current_channel = self.current_channel % self.chan_len
            self.channel_counter = 0

    def transmit(self):
        for i in range(self.chan_len):
            self.radio.setChannel(self.channels[i])
            self.radio.write(bytes(self.data), False)
            # print(self.channels[i],self.data)
            time.sleep(0.4e-3)

    def checksum(self, packet):
        sum = packet[0]
        for i in range(1, 9):
            sum ^= packet[i]
        sum += 0x55
        sum %= 256
        return sum

    def run(self):
        while self.running:
            if self.bound:
                self.build_packet()
                self.transmit()
            time.sleep(0.00001)


class controller(threading.Thread):
    def __init__(self):
        super().__init__()
        self.tx = symaTX()
        self.tx.radio.begin()
        self.tx.radio.setAutoAck(False)
        self.tx.radio.setAddressWidth(5)
        self.tx.radio.setRetries(15, 15)
        self.tx.radio.setDataRate(RF24_250KBPS)
        self.tx.radio.setPALevel(RF24_PA_HIGH)
        self.tx.radio.setPayloadSize(10)
        time.sleep(0.015)

        addr = [161, 105, 1, 104, 204]
        addr.reverse()
        self.tx.addr = addr
        self.tx.init2()
        self.tx.start()

    def run(self):
        # simple up down
        start_time = time.monotonic()
        self.tx.throttle = 0
        self.tx.pitch = 255
        self.tx.roll = 127
        self.tx.yaw = 255
        while True:
            key = input("tpyr:valuem(ex: t:255)")
            key = key.split(":")
            if key[0] == "t":
                self.tx.throttle = key[1]
            elif key[0] == "p":
                self.tx.pitch = key[1]
            elif key[0] == "y":
                self.tx.yaw = key[1]
            elif key[0] == "r":
                self.tx.roll = key[1]


#    def get_control(self):
#        key=

if __name__ == "__main__":
    c = controller()
    c.run()
