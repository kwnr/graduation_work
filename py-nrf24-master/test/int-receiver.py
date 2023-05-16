import argparse
from datetime import datetime
import struct
import sys
import time
import traceback

import pigpio
from nrf24 import *


#
# A simple NRF24L receiver that connects to a PIGPIO instance on a hostname and port, default "localhost" and 8888, and
# starts receiving data on the address specified.  Use the companion program "int-sender.py" or "simple-sender.py" to 
# send data to it from a different Raspberry Pi.
#

# Count of messages received.
count = 0

def gpio_interrupt(gpio, level, tick):
    global count

    # Interrupt cause by data being available.
    print(f'Interrupt: gpio={gpio}, level={("LOW", "HIGH", "NONE")[level]}, tick={tick}')

    # As long as data is ready for processing, process it.
    while nrf.data_ready():
        # Count message and record time of reception.            
        count += 1
        now = datetime.now()
        
        # Read pipe and payload for message.
        pipe = nrf.data_pipe()
        payload = nrf.get_payload()
        hex = ':'.join(f'{i:02x}' for i in payload)

        # Show message received as hex.
        print(f"{now:%Y-%m-%d %H:%M:%S.%f}: pipe: {pipe}, len: {len(payload)}, bytes: {hex}, count: {count}")

        # If the length of the message is 9 bytes and the first byte is 0x01, then we try to interpret the bytes
        # sent as an example message holding a temperature and humidity sent from the "int-sender.py" or 
        # "simple-sender.py" program.
        if len(payload) == 9 and payload[0] == 0x01:
            values = struct.unpack("<Bff", payload)
            print(f'Protocol: {values[0]}, temperature: {values[1]}, humidity: {values[2]}')
        else:
            print('Unknown protocol for data received.')


if __name__ == "__main__":

    print("Python NRF24 Simple Interrupt Based Receiver Example.")
    
    # Parse command line argument.
    parser = argparse.ArgumentParser(prog="simple-receiver.py", description="NRF24 Interrupt Based Receiver Example.")
    parser.add_argument('-n', '--hostname', type=str, default='localhost', help="Hostname for the Raspberry running the pigpio daemon.")
    parser.add_argument('-p', '--port', type=int, default=8888, help="Port number of the pigpio daemon.")
    parser.add_argument('address', type=str, nargs='?', default='1SNSR', help="Address to listen to (3 to 5 ASCII characters)")

    args = parser.parse_args()
    hostname = args.hostname
    port = args.port
    address = args.address

    # Verify that address is between 3 and 5 characters.
    if not (2 < len(address) < 6):
        print(f'Invalid address {address}. Addresses must be between 3 and 5 ASCII characters.')
        sys.exit(1)
    
    # Connect to pigpiod
    print(f'Connecting to GPIO daemon on {hostname}:{port} ...')
    pi = pigpio.pi(hostname, port)
    if not pi.connected:
        print("Not connected to Raspberry Pi ... goodbye.")
        sys.exit()

    # Setup callback for interrupt triggered when data is received.
    pi.callback(24, pigpio.FALLING_EDGE, gpio_interrupt)

    # Create NRF24 object.
    # PLEASE NOTE: PA level is set to MIN because test sender/receivers are often close to each other, and then MIN works better.
    nrf = NRF24(pi, ce=25, payload_size=RF24_PAYLOAD.DYNAMIC, channel=100, data_rate=RF24_DATA_RATE.RATE_250KBPS, pa_level=RF24_PA.MIN)
    nrf.set_address_bytes(len(address))

    # Listen on the address specified as parameter
    nrf.open_reading_pipe(RF24_RX_ADDR.P1, address)
    
    # Display the content of NRF24L01 device registers.
    nrf.show_registers()

    # Enter a loop to prevent us from stopping.
    try:
        print(f'Receive from {address}')        
        while True:
            # Sleep 1 second.
            time.sleep(1)
    except:
        traceback.print_exc()
        nrf.power_down()
        pi.stop()
