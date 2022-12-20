"""Trigger using an parallel port."""

import re
import threading
import time
from typing import Union

from ..utils._checks import _check_type
from ..utils._docs import copy_doc
from ..utils._imports import import_optional_dependency
from ..utils._logs import logger
from ._base import BaseTrigger


class ParallelPortTrigger(BaseTrigger):
    """Trigger using a parallel port (also called LPT port).

    Parameters
    ----------
    address : int (hex) | str
        The address of the parallel port on the system.
        If an :ref:`arduino2lpt` is used, the address must be the COM port
        (e.g. ``"COM5"``) or ``"arduino"`` for automatic detection.
    delay : int
        Delay in milliseconds until which a new trigger cannot be sent. During
        this time, the pins of the LPT port remain in the same state.

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

    - macOS does not have support for built-in parallel ports.
    """

    def __init__(
        self,
        address: Union[int, str],
        delay: int = 50,
    ):
        _check_type(address, ("int", str), item_name="address")
        _check_type(delay, ("int",), item_name="delay")

        self._address = address
        self._delay = delay / 1000.0

        # initialize port
        if isinstance(self._address, str):
            pattern = re.compile(r"^COM\d{1}$")
            if pattern.match(self._address) or self._address == "arduino":
                import_optional_dependency(
                    "serial", extra="Install 'pyserial' for ARDUINO support."
                )
                self._address = (
                    self._address
                    if pattern.match(self._address)
                    else self._search_arduino()
                )
                self._connect_arduino()
            else:
                self._connect_pport()
        else:
            self._connect_pport()

        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def _search_arduino() -> str:
        """Look for a connected Arduino to LPT converter."""
        from serial.tools import list_ports

        for arduino in list_ports.grep(regexp="Arduino"):
            logger.info("[Trigger] Found arduino to LPT on '%s'.", arduino)
            return arduino.device
        else:
            raise IOError("[Trigger] No arduino card was found.")

    def _connect_arduino(self, baud_rate: int = 115200) -> None:
        """Connect to an Arduino to LPT converter."""
        from serial import Serial, SerialException

        try:
            self._port = Serial(self._address, baud_rate)
        except SerialException as error:
            logger.error(
                "[Trigger] Could not connect to arduino to LPT on '%s'.",
                self._address,
            )
            raise Exception from error

        time.sleep(1)
        logger.info(
            "[Trigger] Connected to arduino to LPT on '%s'.", self._address
        )

    def _connect_pport(self) -> None:
        """Connect to the ParallelPort."""
        import platform

        if platform.system() == "Linux":
            import_optional_dependency(
                "parallel",
                extra="Install 'pyparallel' for LPT support on Linux.",
            )

        from .io import ParallelPort

        if ParallelPort is None:
            raise RuntimeError(
                "PsychoPy parallel module has been imported but no parallel "
                "port driver was found. psychopy.parallel supports Linux with "
                "pyparallel and Windows with either inpout32, inpout64 or "
                "dlportio. macOS is not supported."
            )

        try:
            self._port = ParallelPort(self._address)
        except PermissionError as error:
            logger.error(
                "[Trigger] Could not connect to parallel port on '%s'. "
                "To fix a PermissionError, try adding your user into the "
                "group with access to the port or try changing the chmod on "
                "the port.",
                self._address,
            )
            raise Exception from error

        time.sleep(1)
        logger.info(
            "[Trigger] Connected to parallel port on '%s'.", self._address
        )

    @copy_doc(BaseTrigger.signal)
    def signal(self, value: int) -> None:
        super().signal(value)
        if self._offtimer.is_alive():
            logger.warning(
                "You are sending a new signal before the end of the last "
                "signal. Signal ignored. Delay required = %.1f ms.",
                self.delay,
            )
        else:
            self._set_data(value)
            self._offtimer.start()

    def _signal_off(self) -> None:
        """Reset trigger signal to 0 and reset offtimer.

        The offtimer reset is required because threads are one-call only.
        """
        self._set_data(0)
        self._offtimer = threading.Timer(self._delay, self._signal_off)

    def _set_data(self, value: int) -> None:
        """Set data on the pin."""
        if self._address.startswith("COM"):
            self._port.write(bytes([value]))
        else:
            self._port.setData(value)

    def close(self) -> None:
        """Disconnects the parallel port.

        This method should free the parallel or serial port and let other
        application or python process use it.
        """
        if self._address.startswith("COM"):
            try:
                self._port.close()
            except Exception:
                pass
        if hasattr(self, "_port"):
            try:
                del self._port
            except Exception:
                pass

    def __del__(self):  # noqa: D105
        self.close()

    # --------------------------------------------------------------------
    @property
    def address(self) -> Union[int, str]:
        """The address of the parallel port on the system.

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
