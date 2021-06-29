"""
Trigger using a commercial USB to LPT converter.
Drivers can be found here:
https://www-user.tu-chemnitz.de/~heha/bastelecke/Rund%20um%20den%20PC/USB2LPT/

"""
import ctypes
import threading
from pathlib import Path

from .._trigger import _Trigger
from ... import logger


class TriggerUSB2LPT(_Trigger):
    """
    Trigger using a USB to LPT converter.

    Parameters
    ----------
    delay : int
        The delay in milliseconds until which a new trigger cannot be sent.
    verbose : bool
        If True, display a logger.info message when a trigger is sent.
    """

    def __init__(self, delay=50, verbose=True):
        super().__init__(verbose)
        self._lpt = TriggerUSB2LPT._load_dll()
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
        super()._signal_off()
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def _set_data(self, value):
        """
        Set the trigger signal to value.
        """
        self._lpt.setdata(value)

    # --------------------------------------------------------------------
    @staticmethod
    def _load_dll():
        if ctypes.sizeof(ctypes.c_voidp) == 4:
            extension = '32.dll'
        else:
            extension = '64.dll'
        dllname = 'LptControl_USB2LPT' + extension
        dllpath = Path(__file__).parent / 'libs' / dllname

        if not dllpath.exists():
            logger.error(f"Cannot find the required library '{dllname}'.")
            raise RuntimeError

        logger.info(f"Loading '{dllpath}'.")
        return ctypes.cdll.LoadLibrary(str(dllpath))

    # --------------------------------------------------------------------
    @property
    def delay(self):
        """
        The delay to wait between 2 .signal() call in milliseconds.
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
