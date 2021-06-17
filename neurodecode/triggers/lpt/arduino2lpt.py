"""
Trigger using an ARDUINO to LPT converter. Design of the converter can be found
here: https://github.com/fcbg-hnp/arduino-trigger
"""
import time
import threading

import serial
from serial.tools import list_ports

from .._trigger import _Trigger
from ... import logger


class TriggerArduino2LPT(_Trigger):
    """
    Trigger using an ARDUINO to LPT converter.

    Parameters
    ----------
    delay : int
        The delay in milliseconds until which a new trigger cannot be sent.
    verbose : bool
        If True, display a logger.info message when a trigger is sent.
    """

    def __init__(self, delay=50, verbose=True):
        super().__init__(verbose)

        self._com_port = TriggerArduino2LPT._find_arduino_port()
        self._connect_arduino(self._com_port, baud_rate=115200)

        self._delay = delay / 1000.0
        self.offtimer = threading.Timer(self._delay, self._signal_off)

    def _connect_arduino(self, com_port, baud_rate):
        """
        Connect to the Arduino LPT converter.

        Parameters
        ----------
        com_port : str
            The Arduino COM port.
        baud_rate : int
            The baud rate, determines the communication speed.
        """
        try:
            self.ser = serial.Serial(com_port, baud_rate)
        except serial.SerialException as error:
            logger.error(
                "Disconnect and reconnect the ARDUINO convertor because "
                f"{error}")
            raise Exception from error

        time.sleep(1)
        logger.info(f'Connected to {com_port}.')

    def signal(self, value):
        """
        Send a trigger value.
        """
        if self.offtimer.is_alive():
            logger.warning(
                'You are sending a new signal before the end of the last '
                'signal. Signal ignored. Delay required = {self.delay} ms.')
            return False
        self._set_data(value)
        super().signal(value)
        self.offtimer.start()
        return True

    def _signal_off(self):
        """
        Reset trigger signal to 0 and reset offtimer as Threads are one-call
        only.
        """
        super()._signal_off()
        self.offtimer = threading.Timer(self._delay, self._signal_off)

    def _set_data(self, value):
        """
        Set the trigger signal to value.
        """
        self.ser.write(bytes([value]))

    def close(self):
        """
        Disconnects the Arduino and free the COM port.
        """
        try:
            self.ser.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # --------------------------------------------------------------------
    @staticmethod
    def _find_arduino_port():
        """
        Automatic Arduino COM port detection.
        """
        com_port = None
        for arduino in list_ports.grep(regexp='Arduino'):
            logger.info(f'Found {arduino}')
            com_port = arduino.device
            break
        if com_port is None:
            logger.error('No arduino found.')
            raise IOError

    # --------------------------------------------------------------------
    @property
    def com_port(self):
        """
        The COM port to use.
        """
        return self._com_port

    @com_port.setter
    def com_port(self, com_port):
        self._connect_arduino(com_port, baud_rate=115200)
        self._com_port = com_port

    @property
    def delay(self):
        """
        The delay to wait between 2 .signal() call in milliseconds.
        """
        return self._delay * 1000.0

    @delay.setter
    def delay(self, delay):
        if not self.offtimer.is_alive():
            self._delay = delay / 1000.0
            self.offtimer = threading.Timer(self._delay, self._signal_off)
        else:
            logger.warning(
                'You are changing the delay while an event has been sent less '
                'than {self.delay} ms ago. Skipping.')
