# NRF24L01 for Python

This package implement 2.4Ghz communication using NRF24L01+ modules on a Raspberry Pi using Python.

## Changes

* **Version 2.0.0** - Released to PyPi.org on April 8th, 2021.

    This version contains **breaking** changes compared to version 1.1.1.  Make sure to review [**CHANGES.md**](CHANGES.md) and make changes to you client code accordingly.
    
* **Version 1.1.1** - Released to PyPi.org on September 20th, 2020.
* **Version 1.1.0** - Released to PyPi.org on September 20th, 2020.
* **Version 1.0.2** - Released to PyPi.org on May 8th, 2020.
* **Version 1.0.1** - Released to PyPi.org on May 8th, 2020.
* **Version 1.0.0** - Released to PyPi.org on May 8th, 2020.
* **Version 0.8.0** - Released to PyPi.org on May 1st, 2020.
* **Version 0.5.2** - Released to PyPi.org on April 20th, 2020.

## Background

The code is based on a modified version of some example code found on [StackExchange](https://raspberrypi.stackexchange.com/questions/77290/nrf24l01-only-correctly-retrieving-status-and-config-registers).  The author of the original code is also the author of the ```pigpio``` library found here http://abyz.me.uk/rpi/pigpio/.

I have obtained the original authors approval to modify and distribute the code anyway I want.  So, I have created a very basic Python package and published it on PyPI under a MIT license.

The ```nrf24``` packages depends on the ```pigpio``` package that is available via PyPI as well.  Before installing and running any of the code and examples below, please make sure you the ```pigpid``` daemon running on your Raspberry.  This is a library/server that provides access to the GPIO ports of the Raspberry.

Details avalable at http://abyz.me.uk/rpi/pigpio/download.html

Quick installation of `pigpio` on Raspbian:

    sudo apt-get update    
    sudo apt-get install pigpio python-pigpio python3-pigpio

## Installing

You may want to create a virtual environment before installing the `nrf24` package which depends on the `pigpio` package. 

    $ pip install nrf24
    
## Examples

All examples in the `test` folder can be run as command line programs.  They all take optional command line arguments
to specify the `hostname` (default: `localhost`) and the `port` (default: `8888`) of the `pigpio` deamon.  Most of them
also takes one or more addresses to use.  All should have sensible defaults, so running them without arguments should
be an OK first approach to testing your setup.

All test have been run on a Raspberry Pi 4 and a Raspberry Pi Zero Wireless equipped with 2 x NRF24L01+ modules each.

The `int-sender.py` and `int-receiver.py` examples requires extra wiring connenting the IRQ PIN of the NRF24L01+ module
to a GPIO on the Raspberry.  This wiring is shown in **"Raspberry Pi with Single NRF24L01+ Module (IRQ)"**.

The `multi-sender.py` and `multi-receiver.py` examples requires two NRF24L01+ modules.  The wiring for that setup can be
seen in **"Raspberry Pi with Dual NRF24L01+ Modules"** below.

The rest of the Raspberry Pi examples runs with the IRQ wiring as described above, or a simpler wiring like the one shown
in **"Raspberry Pi with Single NRF24L01+ Module"** below.

| Command Line | Comments |
| ------------ | -------- |
| `python test/simple-sender.py` | Emulates a process sending sensor readings every 10 seconds using a **dynamic** payload size (default sending address is `1SNSR`). |
| `python test/simple-receiver.py` | Emulates a receiving process receiving sensor readings from the corresponding sender using a **dynamic** payload size (default listening address `1SNSR`). |
| `python test/fixed-sender.py` | Emulates a process sending sensor readings every 10 seconds using a **fixed** payload size (default sending address is `1SNSR`). |
| `python test/fixed-receiver.py` | Emulates a receiving process receiving sensor readings from the corresponding sender using a **fixed** payload size (default listening address `1SNSR`). |
| `python test/mixed-sender.py` | Shows an example of sending both **fixed** and **dynamic** payload sized messages. Suggested address for fixed messages is `FTEST`, and the suggested address for dynamic messages is `DTEST`. |
| `python test/mixed-receiver.py` | Shows how to configure reading pipes using both **fixed** and **dynamic** message sizes at the same time. |
| `python test/int-sender.py` | Shows how to use interrupt to detect that a message has been sent (default sending address `1SNSR`). |
| `python test/int-receiver.py` | Shows how to use interrupt to detect that a message has been received (default listening address `1SNSR`). |
| `python test/rr-client.py` | Shows example of how to send a request to a server with a reply to address included in the message, and then switching to RX mode to receive the response from the server (default server (TX) address is `1SRVR` and default reply to address (RX) is `1CLNT`) |
| `python test/rr-server.py` | Shows example of a server listening for requests and returning a response to the client (default server (RX) address is `1SRVR`). |
| `python test/ack-sender.py` | Sends message to the receiver every 10 seconds, expecting a payload sent back with the acknowledgement (default sender address `1ACKS`). |
| `python test/ack-receiver.py` | Receives message and sends acknowledgement message with payload (default listen address `1ACKS`).|
| `python test/multi-sender.py` | Sends messages using 2 x NRF24L01+ modules connected to the same Raspberry Pi (defult send addresses `1SRVR` and `2SRVR`). |
| `python test/multi-sender.py` | receives messages using 2 x NRF24L01+ modules connected to the same Raspberry Pi (defult listen addresses `1SRVR` and `2SRVR`). |

If you do not have multiple Raspberry Pi computers, you can run some of the test programs on Arduino. In the `arduino/` directory are sender programs equivalent with
some of those described above.  The wiring for the Arduino Nano can be seen in **"Arduino Nano with DHT22 and NRF24L01+"** below.  Unlike it's Raspberry Pi counterparts
the Arduino examples have been done with an actual DHT22 sensor connected so that we do not need to emulate sensor readings.

| Arduino Code            | Comments                                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `arduino/simple-sender` | Sends temperature and humidity readings to the `simple-receiver.py` counterpart.                                         |
| `arduino/fixed-sender`  | Sends temperature and humidity readings to the `fixed-receiver.py` counterpart.                                          |
| `arduino/mixed-sender`  | Sends temperature and humidity readings to the `mixed-receiver.py` counterpart.                                          |
| `arduino/rr-client`     | Executes request/response calls against its `rr-server.py` counterpart.                                                  |
| `arduino/ack-sender`    | Sends temperature and humidity readings to the `ack-receiver.py` counterpart and receives acknowledgements with payload. |

## Wiring

### Raspberry Pi with Single NRF24L01+ Module (IRQ)

![Raspberry Pi with Single NRF24L01+ Module (IRQ)](https://github.com/bjarne-hansen/py-nrf24/blob/master/doc/pizw-nrf24-1-irq_bb.png "Raspberry Pi with Single NRF24L01+ Module (IRQ)")

### Raspberry Pi with Dual NRF24L01+ Modules

The `multi-sender.py` and `multi-receiver.py` examples requires two NRF24L01+ modules wired to each Raspberry Pi.

![Raspberry Pi with Dual NRF24L01+ Modules](https://github.com/bjarne-hansen/py-nrf24/blob/master/doc/pizw-nrf24-2_bb.png "Raspberry Pi with Dual NRF24L01+ Modules")

### Raspberry Pi with Single NRF24L01+ Module

All the examples, except the `multi-sender.py`, `multi-receiver.py`, `int-sender.py`, and `int-receiver.py` ones will 
run with the following wiring of a single NRF24L01+ module.

![Raspberry Pi with Single NRF24L01+ Module](https://github.com/bjarne-hansen/py-nrf24/blob/master/doc/pizw-nrf24-1_bb.png "Raspberry Pi with Single NRF24L01+ Module")

### Arduino Nano with DHT22 and NRF24L01+

The Arduino examples in `arduino/` can all run with the following wiring.

![Arduino Nano with DHT22 and NRF24L01+](https://github.com/bjarne-hansen/py-nrf24/blob/master/doc/nano-nrf24-1_bb.png "Arduino Nano with DHT22 and NRF24L01+")


    


