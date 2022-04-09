import threading
import time

from ._trigger import _Trigger
from ..utils._docs import fill_doc, copy_doc
from ..utils._checks import _check_type
from ..utils._imports import import_optional_dependency
from ..utils._logs import logger


@fill_doc
class ParallelPortTrigger(_Trigger):
    """
    Trigger using a parallel port (also called LPT port).

    Parameters
    ----------
    address : int (hex) | str
        The address of the parallel port on the system. If ``'arduino'``, uses
        an Arduino to LPT converter. Design of the converter can be found here
        :ref:`arduino2lpt`.
    delay : int
        Delay in milliseconds until which a new trigger cannot be sent. During
        this time, the pins remains in the same state.
    %(trigger_verbose)s

    Notes
    -----
    The address is specific to the system. Typical addresses are:

    - On Linux::

          LPT1 = /dev/parport0
          LPT2 = /dev/parport1
          LPT3 = /dev/parport2

    - On Windows, commom port addresses::

          LPT1 = 0x0378 or 0x03BC
          LPT2 = 0x0278 or 0x0378
          LPT3 = 0x0278

    - macOS does not have support for parallel ports.
    """

    def __init__(self, address, delay: int = 50, *, verbose: bool = True):
        _check_type(address, ('int', str), item_name='address')
        _check_type(delay, ('int', ), item_name='delay')
        super().__init__(verbose)

        self._address = address
        self._delay = delay / 1000.0

        # Initialize port
        if self._address == 'arduino':
            self._connect_arduino()
        else:
            self._connect_pport()

        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def _connect_arduino(self, baud_rate: int = 115200):
        """
        Connect to the Arduino LPT converter.
        """
        # Imports
        import_optional_dependency(
            "serial", extra="Install 'pyserial' for ARDUINO support.")

        from serial import Serial, SerialException
        from serial.tools import list_ports

        # Look for arduino
        logger.info('ParallelPort trigger is using an Arduino converter.')
        com_port = None
        for arduino in list_ports.grep(regexp='Arduino'):
            logger.info("Found '%s'.", arduino)
            com_port = arduino.device
            break
        if com_port is None:
            raise IOError('No arduino card was found.')

        # Connect to arduino
        try:
            self._port = Serial(com_port, baud_rate)
        except SerialException as error:
            logger.error(
                "Disconnect and reconnect the ARDUINO convertor because "
                f"{error}", exc_info=True)
            raise Exception from error

        time.sleep(1)
        logger.info('Connected to %s.', com_port)

    def _connect_pport(self):
        """
        Connect to the ParallelPort.
        """
        # Imports
        from ..externals import psychopy

        # Connect to ParallelPort
        logger.info('ParallelPort trigger is using an on-board port.')
        try:
            self._port = psychopy.parallel.ParallelPort(self._address)
        except PermissionError as error:
            logger.error(
                'To fix a PermissionError, try adding your user into the '
                'group with access to the port or try changing the chmod on '
                'the port.')
            raise Exception from error

        time.sleep(1)
        logger.info('Connected to %s.', self._address)

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
        _check_type(value, ('int', ), item_name='value')
        if self._offtimer.is_alive():
            logger.warning(
                'You are sending a new signal before the end of the last '
                'signal. Signal ignored. Delay required = %.1f ms.',
                self.delay)
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
        if self._address == 'arduino':
            self._port.write(bytes([value]))
        else:
            self._port.setData(value)

    def close(self):
        """
        Disconnects the parallel port. This method should free the parallel
        port and let other application or python process use it.
        """
        if self._address == 'arduino':
            try:
                self._port.close()
            except Exception:
                pass
        else:
            if hasattr(self, '_port'):
                del self._port

    def __del__(self):
        self.close()

    # --------------------------------------------------------------------
    @property
    def address(self):
        """
        Port address.

        :type: int | str
        """
        return self._address

    @property
    def delay(self):
        """
        Delay to wait between two :meth:`~ParallelPortTrigger.signal` call in
        milliseconds.

        :type: float
        """
        return self._delay * 1000.0
