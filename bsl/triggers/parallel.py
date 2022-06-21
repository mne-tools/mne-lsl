import threading
import time
from typing import Union

from ..utils._checks import _check_type
from ..utils._docs import copy_doc, fill_doc
from ..utils._imports import import_optional_dependency
from ..utils._logs import logger
from ._base import BaseTrigger


@fill_doc
class ParallelPortTrigger(BaseTrigger):
    """Trigger using a parallel port (also called LPT port).

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

    - On Windows, common port addresses::

          LPT1 = 0x0378 or 0x03BC
          LPT2 = 0x0278 or 0x0378
          LPT3 = 0x0278

    - macOS does not have support for parallel ports.
    """

    def __init__(
        self,
        address: Union[int, str],
        delay: int = 50,
        *,
        verbose: bool = True,
    ):
        _check_type(address, ("int", str), item_name="address")
        _check_type(delay, ("int",), item_name="delay")
        super().__init__(verbose)

        self._address = address
        self._delay = delay / 1000.0

        # Initialize port
        if self._address == "arduino":
            self._connect_arduino()
        else:
            self._connect_pport()

        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def _connect_arduino(self, baud_rate: int = 115200) -> None:
        """Connect to the Arduino LPT converter."""
        # Imports
        import_optional_dependency(
            "serial", extra="Install 'pyserial' for ARDUINO support."
        )

        from serial import Serial, SerialException
        from serial.tools import list_ports

        # Look for arduino
        logger.info("ParallelPort trigger is using an Arduino converter.")
        com_port = None
        for arduino in list_ports.grep(regexp="Arduino"):
            logger.info("Found '%s'.", arduino)
            com_port = arduino.device
            break
        if com_port is None:
            raise IOError("No arduino card was found.")

        # Connect to arduino
        try:
            self._port = Serial(com_port, baud_rate)
        except SerialException as error:
            logger.error(
                "Disconnect and reconnect the ARDUINO converter because "
                f"{error}",
                exc_info=True,
            )
            raise Exception from error

        time.sleep(1)
        logger.info("Connected to %s.", com_port)

    def _connect_pport(self) -> None:
        """Connect to the ParallelPort."""
        # Imports
        psychopy = import_optional_dependency("psychopy", raise_error=False)
        if psychopy is None:
            import platform

            if platform.system() == "Linux":
                import_optional_dependency("parallel", raise_error=True)
            from ..externals.psychopy.parallel import ParallelPort
        else:
            from psychopy.parallel import ParallelPort
        if ParallelPort is None:
            raise RuntimeError(
                "PsychoPy parallel module has been imported but no parallel "
                "port driver was found. psychopy.parallel supports Linux with "
                "pyparallel and Windows with either inpout32, inpout64 or "
                "dlportio. macOS is not supported."
            )

        # Connect to ParallelPort
        logger.info("ParallelPort trigger is using an on-board port.")
        try:
            self._port = ParallelPort(self._address)
        except PermissionError as error:
            logger.error(
                "To fix a PermissionError, try adding your user into the "
                "group with access to the port or try changing the chmod on "
                "the port."
            )
            raise Exception from error

        time.sleep(1)
        logger.info("Connected to %s.", self._address)

    @copy_doc(BaseTrigger.signal)
    def signal(self, value: int) -> None:
        super().signal(value)
        if self._offtimer.is_alive():
            logger.warning(
                "You are sending a new signal before the end of the last "
                "signal. Signal ignored. Delay required = %.1f ms.",
                self.delay,
            )
            return None
        self._set_data(value)
        self._offtimer.start()

    def _signal_off(self) -> None:
        """Reset trigger signal to 0 and reset offtimer.

        The offtimer reset is required because threads are one-call only.
        """
        self._set_data(0)
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    @copy_doc(BaseTrigger._set_data)
    def _set_data(self, value: int) -> None:
        super()._set_data(value)
        if self._address == "arduino":
            self._port.write(bytes([value]))
        else:
            self._port.setData(value)

    def close(self) -> None:
        """Disconnects the parallel or serial port.

        This method should free the parallel or serial port and let other
        application or python process use it.
        """
        if self._address == "arduino":
            try:
                self._port.close()
            except Exception:
                pass
        else:
            if hasattr(self, "_port"):
                del self._port

    def __del__(self):  # noqa: D105
        self.close()

    # --------------------------------------------------------------------
    @property
    def address(self) -> Union[int, str]:
        """Port address.

        :type: int | str
        """
        return self._address

    @property
    def delay(self) -> float:
        """Delay to wait between two :meth:`~ParallelPortTrigger.signal` call
        in milliseconds.

        :type: float
        """
        return self._delay * 1000.0
