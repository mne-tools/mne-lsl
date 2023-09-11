"""Trigger using an parallel port."""

import threading
import time
from platform import system
from typing import Optional, Union

from ..utils._checks import check_type, check_value, ensure_int
from ..utils._docs import copy_doc
from ..utils._imports import import_optional_dependency
from ..utils.logs import logger
from ._base import BaseTrigger


class ParallelPortTrigger(BaseTrigger):
    """Trigger using a parallel port (also called LPT port).

    Parameters
    ----------
    address : int (hex) | str
        The address of the parallel port on the system.
        If an :ref:`arduino_lpt:Arduino to parallel port (LPT) converter` is used, the
        address must be the serial port address or ``"arduino"`` for automatic
        detection.
    port_type : str | None
        Either ``'arduino'`` or ``'pport'`` depending on the connection.
        If None, BSL attempts to infers the type of port from the address.
    delay : int
        Delay in milliseconds until which a new trigger cannot be sent. During
        this time, the pins of the LPT port remain in the same state.

    Notes
    -----
    The address is specific to the system. Typical parallel port addresses are:

    - On Linux::

          LPT1 = /dev/parport0
          LPT2 = /dev/parport1
          LPT3 = /dev/parport2

    - On Windows::

          LPT1 = 0x0378 or 0x03BC
          LPT2 = 0x0278 or 0x0378
          LPT3 = 0x0278

    - macOS does not have support for built-in parallel ports.
    """

    def __init__(
        self,
        address: Union[int, str],
        port_type: Optional[str] = None,
        delay: int = 50,
    ):
        check_type(address, ("int", str), "address")
        if not isinstance(address, str):
            address = ensure_int(address)
        delay = ensure_int(delay, "delay")
        self._delay = delay / 1000.0
        if port_type is None:
            self._port_type = ParallelPortTrigger._infer_port_type(address)
        else:
            check_type(port_type, (str,), "port_type")
            check_value(port_type, ("arduino", "pport"), "port_type")
            self._port_type = port_type

        # initialize port
        if self._port_type == "arduino":
            import_optional_dependency(
                "serial", extra="Install 'pyserial' for ARDUINO support."
            )
            if address == "arduino":
                self._address = ParallelPortTrigger._search_arduino()
            else:
                self._address = address
            self._connect_arduino()

        elif self._port_type == "pport":
            if system() == "Linux":
                import_optional_dependency(
                    "parallel",
                    extra="Install 'pyparallel' for LPT support on Linux.",
                )
            self._address = address
            self._connect_pport()

        self._signal_off()  # set pins to 0 and define self._offtimer

    @staticmethod
    def _infer_port_type(address: Union[int, str]) -> str:
        """Infer the type of port from the address."""
        if address == "arduino":
            return "arduino"
        if isinstance(address, int):
            return "pport"

        if system() == "Linux":
            if address.startswith("/dev/parport"):
                return "pport"
            if address.startswith("/dev/ttyACM"):
                return "arduino"
            else:
                raise RuntimeError(
                    "[Trigger] Could not infer the port type from the address "
                    f"'{address}'. Please provide the 'port_type' argument "
                    "when creating the ParallelPortTrigger object. "
                )
        elif system() == "Darwin":
            return "arduino"
        elif system() == "Windows":
            if address.startswith("COM"):
                return "arduino"
            else:
                return "pport"

    @staticmethod
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
        except SerialException:
            msg = "[Trigger] Could not access arduino to LPT on " f"'{self._address}'."
            if system() == "Linux":
                msg += (
                    " Make sure you have the permission to access this "
                    "address, e.g. by adding your user account to the "
                    "'dialout' group: 'sudo usermod -a -G dialout <username>'."
                )
            raise SerialException(msg)

        time.sleep(1)
        logger.info("[Trigger] Connected to arduino to LPT on '%s'.", self._address)

    def _connect_pport(self) -> None:
        """Connect to the ParallelPort."""
        from .io import ParallelPort

        if ParallelPort is None and system() == "Darwin":
            raise RuntimeError(
                "[Trigger] macOS does not support built-in parallel port. "
                "Please use an arduino to LPT converter for hardware triggers "
                "or bsl.triggers.LSLTrigger for software triggers."
            )
        elif ParallelPort is None and system() != "Linux":
            raise RuntimeError(
                "[Trigger] Windows supports built-in parallel port via "
                "inpout32, inpout64 or dlportio. Neither of this driver was "
                "found."
            )

        try:
            self._port = ParallelPort(self._address)
        except Exception:
            msg = (
                "[Trigger] Could not access the parallel port on " f"'{self._address}'."
            )
            if system() == "Linux":
                msg += (
                    " Make sure you have the permission to access this "
                    "address, e.g. by adding your user account to the 'lp' "
                    "group: 'sudo usermod -a -G lp <username>'. Make sure the "
                    "'lp' module is removed and the 'ppdev' module is loaded: "
                    "'sudo rmmod lp' & 'sudo modprobe ppdev'. You can "
                    "configure the module loaded by default in "
                    "'/etc/modprobe.d/'."
                )
            raise RuntimeError(msg)

        time.sleep(1)
        logger.info("[Trigger] Connected to parallel port on '%s'.", self._address)

    @copy_doc(BaseTrigger.signal)
    def signal(self, value: int) -> None:
        super().signal(value)
        if self._offtimer.is_alive():
            logger.warning(
                "[Trigger] You are sending a new signal before the end of the "
                "last signal. Signal ignored. Delay required = %.1f ms.",
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
        if self._port_type == "arduino":
            self._port.write(bytes([value]))
        else:
            self._port.setData(value)

    def close(self) -> None:
        """Disconnect the parallel port.

        This method should free the parallel or serial port and let other
        application or python process use it.
        """
        if self._port_type == "arduino":
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
        """Delay (ms) to wait between two :meth:`~ParallelPortTrigger.signal`.

        :type: float
        """
        return self._delay * 1000.0

    @property
    def port_type(self) -> str:
        """Type of connection port.

        :type: str
        """
        return self._port_type
