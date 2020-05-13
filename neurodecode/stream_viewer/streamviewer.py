import sys
from PyQt5.QtWidgets import QApplication

from neurodecode import logger
from neurodecode.stream_viewer._scope import _Scope

import neurodecode.utils.pycnbi_utils as pu

class StreamViewer:
    """
    Class for displaying the signals streamed from an amplifier in real time.
    
    ...

    Parameters
    ----------
    amp_name : str
        The amplifier's name to connect to
    amp_serial : str
        The amplifier's serial number
    
    
    Attributes
    ----------
    amp_name : str
        The amplifier's name to connect to
    amp_serial : str
        The amplifier's serial number
        
    Methods
    -------
    run()
        Launch the GUI and plot the streamed data
    search_stream()
        Look for an available stream on the LSL server
    """
    #----------------------------------------------------------------------
    def __init__(self, amp_name=None, amp_serial=None):
        """Constructor
        
        Parameters
        ----------
        amp_name : str
            The amplifier's name to connect to
        amp_serial : str
            The amplifier's serial number
        """
        self.amp_name = amp_name
        self.amp_serial = amp_serial
    
    #----------------------------------------------------------------------
    def run(self):
        """
        Connect to the selected amplifier and plot the streamed data 
        
        If not amp infos are provided, look for available streams on the LSL server.
        """
        if (self.amp_name is None):
            self.search_stream()
        
        logger.info('Connecting to a stream %s (Serial %s).' % (amp_name, amp_serial))
        
        app = QApplication(sys.argv)
        ex = _Scope(amp_name, amp_serial)
        sys.exit(app.exec_())
    
    #----------------------------------------------------------------------
    def search_stream(self):
        """
        Select an available stream on the LSL server to connect to.
        
        Assign the found amp name and serial number to internal attributes
        """
        self.amp_name, self.amp_serial = pu.search_lsl()
        
    
#----------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 2:
        amp_name = sys.argv[1]
        amp_serial = None
    elif len(sys.argv) == 3:
        amp_name, amp_serial = sys.argv[1:3]
    else:
        amp_name, amp_serial = pu.search_lsl()
    if amp_name == 'None':
        amp_name = None
    
    stream_viewer = StreamViewer(amp_name, amp_serial)
    stream_viewer.run()
