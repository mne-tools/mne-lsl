"""
Trigger using an LPT port.
"""
import time
import ctypes
import threading
from pathlib import Path

from ._trigger import _Trigger
from ..utils._logs import logger
from ..utils._checks import _check_type
from ..utils._docs import fill_doc, copy_doc
from ..utils._imports import import_optional_dependency


@fill_doc
class TriggerLPT(_Trigger):
    """
    Trigger using the LPT port on the motherboard.

    .. warning:: This class has been adapted from NeuroDecode and has not been
                 tested because computers with build-in LPT port are not common
                 anymore. PCIe-LPT card have not been tested either.
                 When possible, prefer the :ref:`arduino2lpt` or the
                 `~bsl.triggers.software.TriggerSoftware`.

    Parameters
    ----------
    portaddr : hex | `int`
        Port address in hexadecimal format (standard: ``0x278``, ``0x378``).
    %(trigger_lpt_delay)s
    %(trigger_verbose)s
    """

    def __init__(self, portaddr: int, delay: int = 50, *,
                 verbose: bool = True):
        self._portaddr = TriggerLPT._check_portaddr(portaddr)
        _check_type(delay, ('int', ), item_name='delay')
        super().__init__(verbose)

        logger.debug("LPT port address: %d" % self._portaddr)

        self._lpt = TriggerLPT._load_dll()
        if self._lpt.init() == -1:
            logger.error(
                'Connecting to LPT port failed. Check the driver status.')
            raise IOError

        self._delay = delay / 1000.0
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
        _check_type(value, ('int', ), item_name='value')
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

    @copy_doc(_Trigger._set_data)
    def _set_data(self, value: int):
        super()._set_data(value)
        self._lpt.setdata(self._portaddr, value)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_portaddr(portaddr: int) -> int:
        """
        Checks the portaddr value against usual values.
        """
        _check_type(portaddr, ('int', ), item_name='portaddr')
        if portaddr not in [0x278, 0x378]:
            logger.warning('LPT port address %s is unusual.', portaddr)

        return int(portaddr)

    @staticmethod
    def _load_dll():
        """
        Load the correct .dll.
        """
        ext = '32.dll' if ctypes.sizeof(ctypes.c_voidp) == 4 else '64.dll'
        dllname = 'LptControl_Desktop' + ext
        dllpath = Path(__file__).parent / 'lpt_libs' / dllname

        if not dllpath.exists():
            raise RuntimeError("Cannot find the required library '%s'."
                               % dllname)

        logger.info("Loading '%s'.", dllpath)
        return ctypes.cdll.LoadLibrary(str(dllpath))

    # --------------------------------------------------------------------
    @property
    def portaddr(self):
        """
        Port address.

        :type: `int`
        """
        return self._portaddr

    @property
    def delay(self):
        """
        Delay to wait between two ``.signal()`` call in milliseconds.

        :type: `float`
        """
        return self._delay * 1000.0


@fill_doc
class TriggerUSB2LPT(_Trigger):
    """
    Trigger using a USB to LPT converter. Drivers can be found here:
    %(trigger_lpt_usb2lpt_link)s

    .. warning:: This class has been adapted from NeuroDecode and has not been
                 tested.
                 When possible, prefer the :ref:`arduino2lpt` or the
                 `~bsl.triggers.software.TriggerSoftware`.

    Parameters
    ----------
    %(trigger_lpt_delay)s
    %(trigger_verbose)s
    """

    def __init__(self, delay: int = 50, *, verbose: bool = True):
        _check_type(delay, ('int', ), item_name='delay')
        super().__init__(verbose)
        self._lpt = TriggerUSB2LPT._load_dll()
        if self._lpt.init() == -1:
            logger.error(
                'Connecting to LPT port failed. Check the driver status.')
            raise IOError

        self._delay = delay / 1000.0
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
        _check_type(value, ('int', ), item_name='value')
        if self._offtimer.is_alive():
            logger.warning(
                'You are sending a new signal before the end of the last '
                'signal. Signal ignored. Delay required = {self.delay} ms.')
            return False
        self._set_data(value)
        super().signal(value)
        self._offtimer.start()
        return True

    @copy_doc(TriggerLPT._signal_off)
    def _signal_off(self):
        self._set_data(0)
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    @copy_doc(_Trigger._set_data)
    def _set_data(self, value: int):
        super()._set_data(value)
        self._lpt.setdata(value)

    # --------------------------------------------------------------------
    @staticmethod
    def _load_dll():
        """
        Load the correct .dll.
        """
        ext = '32.dll' if ctypes.sizeof(ctypes.c_voidp) == 4 else '64.dll'
        dllname = 'LptControl_USB2LPT' + ext
        dllpath = Path(__file__).parent / 'lpt_libs' / dllname

        if not dllpath.exists():
            raise RuntimeError("Cannot find the required library '%s'."
                               % dllname)

        logger.info("Loading '%s'.", dllpath)
        return ctypes.cdll.LoadLibrary(str(dllpath))

    # --------------------------------------------------------------------
    @property
    @copy_doc(TriggerLPT.delay)
    def delay(self):
        return self._delay * 1000.0


@fill_doc
class TriggerArduino2LPT(_Trigger):
    """
    Trigger using an ARDUINO to LPT converter. Design of the converter can be
    found here: :ref:`arduino2lpt`.

    Parameters
    ----------
    %(trigger_lpt_delay)s
    %(trigger_verbose)s
    """
    BAUD_RATE = 115200

    def __init__(self, delay: int = 50, *, verbose: bool = True):
        import_optional_dependency(
            "serial", extra="Install pyserial for ARDUINO support.")
        _check_type(delay, ('int', ), item_name='delay')
        super().__init__(verbose)

        self._com_port = TriggerArduino2LPT._find_arduino_port()
        self._connect_arduino(self._com_port, baud_rate=self.BAUD_RATE)

        self._delay = delay / 1000.0
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def _connect_arduino(self, com_port: str, baud_rate: int):
        """
        Connect to the Arduino LPT converter.

        Parameters
        ----------
        com_port : `str`
            Arduino COM port.
        baud_rate : `int`
            Baud rate, determines the communication speed.
        """
        import serial

        try:
            self._ser = serial.Serial(com_port, self.BAUD_RATE)
        except serial.SerialException as error:
            logger.error(
                "Disconnect and reconnect the ARDUINO convertor because "
                f"{error}", exc_info=True)
            raise Exception from error

        time.sleep(1)
        logger.info(f'Connected to {com_port}.')

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
        _check_type(value, ('int', ), item_name='value')
        if self._offtimer.is_alive():
            logger.warning(
                'You are sending a new signal before the end of the last '
                'signal. Signal ignored. Delay required = {self.delay} ms.')
            return False
        self._set_data(value)
        super().signal(value)
        self._offtimer.start()
        return True

    @copy_doc(TriggerLPT._signal_off)
    def _signal_off(self):
        self._set_data(0)
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    @copy_doc(_Trigger._set_data)
    def _set_data(self, value: int):
        """
        Set the trigger signal to value.
        """
        super()._set_data(value)
        self._ser.write(bytes([value]))

    def close(self):
        """
        Disconnects the Arduino and free the COM port.
        """
        try:
            self._ser.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # --------------------------------------------------------------------
    @staticmethod
    def _find_arduino_port() -> str:
        """
        Automatic Arduino COM port detection.
        """
        from serial.tools import list_ports

        com_port = None
        for arduino in list_ports.grep(regexp='Arduino'):
            logger.info(f'Found {arduino}')
            com_port = arduino.device
            break
        if com_port is None:
            logger.error('No arduino found.')
            raise IOError

        return com_port

    # --------------------------------------------------------------------
    @property
    def com_port(self):
        """
        COM port to use.

        :setter: Change the COM port and connect the trigger.
        :type: `str`
        """
        return self._com_port

    @property
    @copy_doc(TriggerLPT.delay)
    def delay(self):
        return self._delay * 1000.0
