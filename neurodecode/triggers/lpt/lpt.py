"""
Trigger using a computer with an LPT port on the motherboard.
"""
import ctypes
import threading
from pathlib import Path

from .._trigger import _Trigger
from ... import logger


class TriggerLPT(_Trigger):
    """
    Trigger using the LPT port on the motherboard.

    Parameters
    ----------
    portaddr : hex | int
        Port address in hexadecimal format (standard: 0x278, 0x378).
    delay : int
        Delay in milliseconds until which a new trigger cannot be sent.
    verbose : bool
        If True, display a logger.info message when a trigger is sent.
    """

    def __init__(self, portaddr, delay=50, verbose=True):
        super().__init__(verbose)
        self._portaddr = TriggerLPT._check_portaddr(portaddr)

        self._lpt = TriggerLPT._load_dll()
        if self._lpt.init() == -1:
            logger.error(
                'Connecting to LPT port failed. Check the driver status.')
            raise IOError

        self._delay = delay / 1000.0
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def signal(self, value):
        """
        Send a trigger value.
        """
        if self._offtimer.is_alive():
            logger.warning(
                'You are sending a new signal before the end of the last '
                'signal. Signal ignored. Delay required = {self.delay} ms.')
            return False
        self._set_data(value)
        super().signal(value)
        self._offtimer.start()
        return True

    def _signal_off(self):
        """
        Reset trigger signal to 0 and reset offtimer as Threads are one-call
        only.
        """
        self._set_data(0)
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def _set_data(self, value):
        """
        Set the trigger signal to value.
        """
        self._lpt.setdata(self._portaddr, value)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_portaddr(portaddr):
        """
        Checks the portaddr value against usual values.
        """
        if portaddr not in [0x278, 0x378]:
            logger.warning(f'LPT port address {portaddr} is unusual.')

        return int(portaddr)

    @staticmethod
    def _load_dll():
        """
        Load the correct .dll.
        """
        if ctypes.sizeof(ctypes.c_voidp) == 4:
            extension = '32.dll'
        else:
            extension = '64.dll'
        dllname = 'LptControl_Desktop' + extension
        dllpath = Path(__file__).parent / 'libs' / dllname

        if not dllpath.exists():
            logger.error(f"Cannot find the required library '{dllname}'.")
            raise RuntimeError

        logger.info(f"Loading '{dllpath}'.")
        return ctypes.cdll.LoadLibrary(str(dllpath))

    # --------------------------------------------------------------------
    @property
    def portaddr(self):
        """
        Port address.
        """
        return self._portaddr

    @portaddr.setter
    def portaddr(self, portaddr):
        if not self._offtimer.is_alive():
            self._portaddr = TriggerLPT._check_portaddr(portaddr)
        else:
            logger.warning(
                'You are changing the port while an event has been sent less '
                'than {self.delay} ms ago. Skipping.')

    @property
    def delay(self):
        """
        Delay to wait between 2 .signal() call in milliseconds.
        """
        return self._delay * 1000.0

    @delay.setter
    def delay(self, delay):
        if not self._offtimer.is_alive():
            self._delay = delay / 1000.0
            self._offtimer = threading.Timer(self._delay, self._signal_off)
        else:
            logger.warning(
                'You are changing the delay while an event has been sent less '
                'than {self.delay} ms ago. Skipping.')
