import sys
from PyQt5.QtWidgets import QApplication

from neurodecode import logger
from neurodecode.stream_viewer._scope import _Scope

import neurodecode.utils.pycnbi_utils as pu

class StreamViewer:
    """
    Class for displaying in real time the signals coming from an lsl stream.

    Parameters
    ----------
    amp_name : str
        The amplifier's name to connect to
    amp_serial : str
        The amplifier's serial number
    """
    #----------------------------------------------------------------------
    def __init__(self, amp_name=None, amp_serial=None):
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
        
        logger.info('Connecting to a stream %s (Serial %s).' % (self.amp_name, self.amp_serial))
        
        app = QApplication(sys.argv)
        ex = _Scope(self.amp_name)
        sys.exit(app.exec_())

    #----------------------------------------------------------------------
    def search_stream(self):
        """
        Select an available stream on the LSL server to connect to.
        
        Assign the found amp name and serial number to the internal attributes
        """
        self.amp_name, self.amp_serial = pu.search_lsl()
        
    
#----------------------------------------------------------------------
if __name__ == '__main__':
    
    amp_name = None
    
    if len(sys.argv) > 2:
        raise RuntimeError("Too many arguments provided, maximum is 1.")
    
    if len(sys.argv) > 1:
        amp_name = sys.argv[1]
        
    if len(sys.argv) == 1:
        amp_name, amp_serial = pu.search_lsl()
    
    stream_viewer = StreamViewer(amp_name)
    stream_viewer.run()
