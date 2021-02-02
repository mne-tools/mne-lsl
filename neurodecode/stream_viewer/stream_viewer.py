import sys
from PyQt5.QtWidgets import QApplication

from neurodecode import logger
from neurodecode.stream_viewer._scope import _Scope

import neurodecode.utils.pycnbi_utils as pu

class StreamViewer:
    """
    Class for displaying in real time the signals coming from a lsl stream.

    Parameters
    ----------
    amp_name : str
        The amplifier's name to connect to
    """
    #----------------------------------------------------------------------
    def __init__(self, amp_name=None):
        self.amp_name = amp_name
    
    #----------------------------------------------------------------------
    def start(self):
        """
        Connect to the selected amplifier and plot the streamed data 
        
        If not amp infos are provided, look for available streams on the LSL server.
        """
        if (self.amp_name is None):
            self.search_stream()
        
        logger.info('Connecting to the stream: {}'.format(self.amp_name))
        
        app = QApplication(sys.argv)
        ex = _Scope(self.amp_name)
        sys.exit(app.exec_())

    #----------------------------------------------------------------------
    def search_stream(self):
        """
        Select an available stream on the LSL server to connect to.
        
        Assign the found amp name and serial number to the internal attributes
        """
        self.amp_name, _ = pu.search_lsl()
        
    
#----------------------------------------------------------------------
if __name__ == '__main__':
    
    amp_name = None
    
    if len(sys.argv) > 2:
        raise RuntimeError("Too many arguments provided, maximum is 1.")
    
    if len(sys.argv) > 1:
        amp_name = sys.argv[1]
        
    stream_viewer = StreamViewer(amp_name)
    stream_viewer.start()
