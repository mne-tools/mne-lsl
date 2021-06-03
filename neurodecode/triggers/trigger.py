"""
Send trigger events to parallel port (LPT).

See sample code at the end.

Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)
"""

import sys
import time
import pylsl
import ctypes
import threading
import multiprocessing as mp
from pathlib import Path

from .. import logger
from ..utils.lsl import start_client


class Trigger(object):
    """
    Class for sending trigger events.

    Parameters
    ----------
    lpttype : str
        - 'DESKTOP': Desktop native LPT
        - 'USB2LPT': Commercial USB2LPT adapter
        - 'SOFTWARE': Software trigger
        - 'ARDUINO': Arduino trigger
        - 'FAKE': Mock trigger device for testing

    portaddr : hex
        The port address in hexadecimal format (standard: 0x278, 0x378)
        When using USB2LPT, the port number (e.g. 0x378) can be searched automatically.
        When using Desktop's LPT, the port number must be specified during initialization.
    verbose : bool
        The verbosity, True display logging info output.
    state : multiprocessing.value
        For GUI usage.
    """

    def __init__(self, lpttype='SOFTWARE', portaddr=None,
                 verbose=True, state=mp.Value('i', 1)):

        self.evefile = None
        self.offtimer = None
        self._lpttype = lpttype.upper().strip()
        self.verbose = verbose

        if self._lpttype in ['USB2LPT', 'DESKTOP']:
            if portaddr not in [0x278, 0x378]:
                logger.warning(f'LPT port address {portaddr} is unusual.')

            self.portaddr = portaddr
            dllname = self._find_dllname()
            self.lpt = self._load_dll(dllname)

        elif self._lpttype == 'ARDUINO':
            BAUD_RATE = 115200
            com_port = self._find_arduino_port()
            self._connect_arduino(com_port, BAUD_RATE)

        elif self._lpttype == 'SOFTWARE':
            logger.info('Using software trigger')
            self.evefile = self._find_evefile(state)

        elif self._lpttype == 'FAKE' or self._lpttype is None or self._lpttype is False:
            logger.warning('Using a fake trigger.')
            self._lpttype = 'FAKE'
            self.lpt = None

        else:
            logger.error(f'Unrecognized lpttype device name {lpttype}')
            sys.exit(-1)

    def _find_dllname(self):
        """
        Name the required dll libraries in case of USB2LPT or DESKTOP trigger.

        Returns
        -------
        string
            The dll library name to load
        """
        if ctypes.sizeof(ctypes.c_voidp) == 4:
            extension = '32.dll'
        else:
            extension = '64.dll'

        if self._lpttype == 'DESKTOP':
            dllname = 'LptControl_Desktop' + extension
        elif self._lpttype == 'USB2LPT':
            dllname = 'LptControl_USB2LPT' + extension

        return dllname

    def _load_dll(self, dllname):
        """
        Load the dll library.

        Parameters
        ----------
        dllname : str
            The dll lib's name.

        Returns
        -------
        lib
            The loaded library
        """
        # Define the dll library path
        dllpath = Path(__file__).parent / 'libs' / dllname

        # Ensure that the dll exists
        if not dllpath.exists():
            logger.error(f'Cannot find the required library {dllname}')
            raise RuntimeError

        logger.info(f'Loading {dllpath}')

        return ctypes.cdll.LoadLibrary(str(dllpath))

    def _find_arduino_port(self):
        """
        Automatic Arduino comPort detection.
        """
        import serial.tools.list_ports

        arduinos = [x for x in serial.tools.list_ports.grep('Arduino')]

        if len(arduinos) == 0:
            logger.error('No Arduino found. Stop.')
            sys.exit()

        for i, a in enumerate(arduinos):
            logger.info(f'Found {a[0]}')
        try:
            com_port = arduinos[0].device
        except AttributeError:  # depends on Python distribution
            com_port = arduinos[0][0]

        return com_port

    def _connect_arduino(self, com_port, baud_rate):
        """
        Connect to the Arduino USB2LPT converter.

        Parameters
        ----------
        com_port : str
            The Arduino comPort
        baud_rate : int
            The baud rate, determined the communication speed
        """
        import serial

        try:
            self.ser = serial.Serial(com_port, baud_rate)
        except serial.SerialException as error:
            raise Exception(
                f"Disconnect and reconnect the ARDUINO convertor because {error}")

        time.sleep(1)  # doesn't work without this delay. why?
        logger.info(f'Connected to {com_port}.')

    def _find_evefile(self, state):
        """
        Find the event file name from LSL Server in case of SOFTWARE trigger.
        """
        LSL_SERVER = 'StreamRecorderInfo'

        inlet = start_client(LSL_SERVER, state)
        evefile = inlet.info().source_id()
        logger.info(f'Event file is: {evefile}')

        return evefile

    def init(self, duration):
        """
        Initialize the trigger's duration.

        Parameters
        ----------
        duration : int
            The event's duration in ms.

        Returns
        -------
        bool
            True if trigger is ready to use, False otherwise.
        """
        if self._lpttype == 'SOFTWARE':
            #  Open the file
            try:
                self.evefile = open(self.evefile, 'a')
            # Close it before if already opened.
            except IOError:
                self.evefile.close()
                self.evefile = open(self.evefile, 'a')

            logger.info('Ignoring delay parameter for software trigger.')
            return True
        elif self._lpttype == 'FAKE':
            return True
        else:
            self.delay = duration / 1000.0

            if self._lpttype in ['DESKTOP', 'USB2LPT']:
                if self.lpt.init() == -1:
                    logger.error(
                        'Connecting to LPT port failed. '
                        'Check the driver status.')
                    self.lpt = None
                    return False

            self.offtimer = threading.Timer(self.delay, self._signal_off)

            return True

    def _signal_off(self):
        """
        Set data to zero (all bits off).
        """
        if self._lpttype == 'SOFTWARE':
            return self._write_software_event(0)
        elif self._lpttype == 'FAKE':
            logger.info('FAKE trigger off')
            return True
        else:
            self._set_data(0)
            self.offtimer = threading.Timer(self.delay, self._signal_off)

    def signal(self, value):
        """
        Sends the value to the parallel port and sets to 0 after a set period.
        The value should be an integer in the range of 0-255.

        Parameters
        ----------
        value : int
            The trigger event to write to the file.

        Returns
        -------
        bool
            True if trigger event has been properly sent
        """
        if self._lpttype == 'SOFTWARE':
            if self.verbose is True:
                logger.info(f'Sending software trigger {value}')
            return self._write_software_event(value)
        elif self._lpttype == 'FAKE':
            logger.info(f'Sending FAKE trigger signal {value}')
            return True
        else:
            if not self.offtimer:
                logger.error(
                    'First, initialize the event duration with init().')
                return False
            if self.offtimer.is_alive():
                logger.warning(
                    'You are sending a new signal before the end of the last signal. Signal ignored.')
                logger.warning(f'Delay required = {self.delay} us.')
                return False
            self._set_data(value)
            if self.verbose is True:
                logger.info(f'Sending {value}')
            self.offtimer.start()
            return True

    def _write_software_event(self, value):
        """
        Write to file in case of SOFTWARE trigger.

        Parameters
        ----------
        value : int
            The trigger event to write to the file.
        """
        assert self._lpttype == 'SOFTWARE'
        self.evefile.write('%.6f\t0\t%d\n' % (pylsl.local_clock(), value))
        return True

    def _set_data(self, value):
        """
        Set the trigger's value to the LPT port.

        Parameters
        ----------
        value : int
            The trigger event to write to the file.
        """
        if self._lpttype == 'SOFTWARE':
            logger.error('_set_data() not supported for software trigger.')
            return False
        elif self._lpttype == 'FAKE':
            logger.info(f'FAKE trigger value {value}')
            return True
        else:
            if self._lpttype == 'USB2LPT':
                self.lpt.setdata(value)
            elif self._lpttype == 'DESKTOP':
                self.lpt.setdata(self.portaddr, value)
            elif self._lpttype == 'ARDUINO':
                self.ser.write(bytes([value]))
            else:
                raise RuntimeError('Wrong trigger device.')

    def set_pin(self, pin):
        """
        Set a specific pin to 1.
        """
        if self._lpttype == 'SOFTWARE':
            logger.error('set_pin() not supported for software trigger.')
            return False
        elif self._lpttype == 'FAKE':
            logger.info(f'FAKE trigger pin {pin}')
            return True
        else:
            self._set_data(2 ** (pin - 1))

    @property
    def type(self):
        """
        The trigger type
        """
        return self._lpttype

    @type.setter
    def type(self, new_type):
        logger.warning("The trigger type cannot be modify directly, "
                       "instead instance a new Trigger.")

    def __del__(self):
        if self.evefile is not None and not self.evefile.closed:
            self.evefile.close()

        if self._lpttype == 'ARDUINO':
            try:
                self.ser.close()
            except:
                pass
