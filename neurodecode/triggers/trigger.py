from __future__ import print_function, division

"""
Send trigger events to parallel port.

See sample code at the end.

Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import os
import sys
import time
import pylsl
import ctypes
import threading
import multiprocessing as mp
import neurodecode.utils.cnbi_lsl as cnbi_lsl
import neurodecode.utils.pycnbi_utils as pu
from neurodecode import logger
from builtins import input, bytes

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
        The verbosity, True display logging info output
    state : multiprocessing.value
        For GUI usage
    """
    #----------------------------------------------------------------------
    def __init__(self, lpttype='SOFTWARE', portaddr=None, verbose=True, state=mp.Value('i', 1)):
        
        self.evefile = None
        self.offtimer = None
        self._lpttype = lpttype
        self.verbose = verbose

        if self._lpttype in ['USB2LPT', 'DESKTOP']:
            if portaddr not in [0x278, 0x378]:
                logger.warning('LPT port address %d is unusual.' % portaddr)
                
            self.portaddr = portaddr
            dllname = self._find_dllname()
            self.lpt = self._load_dll(dllname)

        elif self._lpttype == 'ARDUINO':
            BAUD_RATE = 115200
            com_port = self._find_arduino_port()
            self._connect_arduino(com_port, BAUD_RATE)

        elif self._lpttype == 'SOFTWARE':
            logger.info('Using software trigger')
            evefile = self._find_evefile(state)
            self.evefile = open(evefile, 'a')

            #if check_lsl_offset:
                #self._compute_lsl_offset(evefile)

        elif self._lpttype == 'FAKE' or self._lpttype is None or self._lpttype is False:
            logger.warning('Using a fake trigger.')
            self._lpttype = 'FAKE'
            self.lpt = None

        else:
            logger.error('Unrecognized lpttype device name %s' % lpttype)
            sys.exit(-1)

    #----------------------------------------------------------------------
    def _find_evefile(self, state):
        """
        Find the event file name from LSL Server in case of SOFTWARE trigger.
        """
        LSL_SERVER = 'StreamRecorderInfo'
        
        inlet = cnbi_lsl.start_client(LSL_SERVER, state)
        evefile = inlet.info().source_id()
        logger.info('Event file is: %s' % evefile) 
        
        return evefile
        
    #----------------------------------------------------------------------
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
            raise Exception("Disconnect and reconnect the ARDUINO convertor because {}".format(error))            
        
        time.sleep(1)  # doesn't work without this delay. why?
        logger.info('Connected to %s.' % com_port)
    
    #----------------------------------------------------------------------
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
            logger.info('Found %s' % a[0])
        try:
            com_port = arduinos[0].device
        except AttributeError: # depends on Python distribution
            com_port = arduinos[0][0]
        
        return com_port
    
    #----------------------------------------------------------------------
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
        f = os.path.dirname(__file__) + '/libs/' + dllname
        
        # Ensure that the dll exists
        if os.path.exists(f):
            dllpath = f
        else:
            logger.error('Cannot find the required library %s' % dllname)
            raise RuntimeError
        
        logger.info('Loading %s' % dllpath)
        
        return ctypes.cdll.LoadLibrary(dllpath)

    #----------------------------------------------------------------------
    def _find_dllname(self):
        """
        Name the required dll libraries in case of USB2LPT or DESKTOP trigger.
        
        Returns
        -------
        string
            The dll library name to load
        """
        if ctypes.sizeof(ctypes.c_voidp) == 4:
            extension =  '32.dll'
        else:
            extension =  '64.dll'
        
        dllname = 'LptControl_' + self._lpttype + extension 

        return dllname
    
    #----------------------------------------------------------------------
    def __del__(self):
        if self.evefile is not None and not self.evefile.closed:
            self.evefile.close()
            
    #----------------------------------------------------------------------
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
            logger.info('Ignoring delay parameter for software trigger.')
            return True
        elif self._lpttype == 'FAKE':
            return True
        else:
            self.delay = duration / 1000.0

            if self._lpttype in ['DESKTOP', 'USB2LPT']:
                if self.lpt.init() == -1:
                    logger.error('Connecting to LPT port failed. Check the driver status.')
                    self.lpt = None
                    return False

            self.offtimer = threading.Timer(self.delay, self._signal_off)

            return True

    #----------------------------------------------------------------------
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

    #----------------------------------------------------------------------
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
            logger.info('FAKE trigger value %s' % value)
            return True
        else:
            if self._lpttype == 'USB2LPT':
                self.lpt.setdata(value)
            elif self._lpttype == 'DESKTOP':
                self.lpt.setdata(self.portaddr, value)
            elif self._lpttype == 'ARDUINO':
                self.ser.write(bytes([value]))
            else:
                raise RuntimeError('Wrong trigger device')

    #----------------------------------------------------------------------
    def signal(self, value):
        """
        Sends the value to the parallel port and sets to 0 after a set period.
        The value shuold be an integer in the range of 0-255.
        
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
                logger.info('Sending software trigger %s' % value)
            return self._write_software_event(value)
        elif self._lpttype == 'FAKE':
            logger.info('Sending FAKE trigger signal %s' % value)
            return True
        else:
            if not self.offtimer:
                logger.error('First, initialize the event duration with init()')
                return False
            if self.offtimer.is_alive():
                logger.warning('You are sending a new signal before the end of the last signal. Signal ignored.')
                logger.warning('Delay required = {}'.format(self.delay))
                return False
            self._set_data(value)
            if self.verbose is True:
                logger.info('Sending %s' % value)
            self.offtimer.start()
            return True

    #----------------------------------------------------------------------
    def _signal_off(self):
        """
        Set data to zero (all bits off)
        """
        if self._lpttype == 'SOFTWARE':
            return self._write_software_event(0)
        elif self._lpttype == 'FAKE':
            logger.info('FAKE trigger off')
            return True
        else:
            self._set_data(0)
            self.offtimer = threading.Timer(self.delay, self._signal_off)

    #----------------------------------------------------------------------
    def set_pin(self, pin):
        """
        Set a specific pin to 1.
        """
        if self._lpttype == 'SOFTWARE':
            logger.error('set_pin() not supported for software trigger.')
            return False
        elif self._lpttype == 'FAKE':
            logger.info('FAKE trigger pin %s' % pin)
            return True
        else:
            self._set_data(2 ** (pin - 1))


    #----------------------------------------------------------------------
    @property
    def type(self):
        """
        The trigger type
        """
        return self._lpttype
    
    #----------------------------------------------------------------------
    @type.setter
    def type(self, new_type):
        logger.warning("The trigger type cannot be modify directly, instead instance a new Trigger.")
        
    
#----------------------------------------------------------------------
if __name__ == '__main__':
    
    lpttype = input("Provide the type of LPT connector: DESKTOP, USB2LPT, SOFTWARE, ARDUINO, FAKE \n>>")
    
    trigger = Trigger(lpttype=lpttype)
    
    if not trigger.init(500):
        raise RuntimeError('LPT port cannot be opened. Using mock trigger.')

    print('Type quit or Ctrl+C to finish.')
    while True:
        val = input('Trigger value? \n>> ')
        if val.strip() == '':
            continue
        if val == 'quit':
            break
        if 0 <= int(val) <= 255:
            trigger.signal(int(val))
            print('Sent %d' % int(val))
        else:
            print('Ignored %s' % val)
