import sys
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from pycnbi import add_logger_handler

########################################################################
class WriteStream():
    """
    The new Stream Object which replaces the default stream associated with sys.stdout.
    It puts data in a queue!
    """
    #----------------------------------------------------------------------
    def __init__(self, queue):
        """
        Constructor
        """
        self.queue = queue

    #----------------------------------------------------------------------
    def write(self, text):
        """
        Overload sys.stdout write function
        """
        self.queue.put(text)
    
    #----------------------------------------------------------------------
    def flush(self):
        """
        Overload sys.stdout flush function
        """
        #if self.queue.empty() is False:
            #tmp = self.queue.get()
        pass
        

########################################################################
class MyReceiver(QObject):
    """
    A QObject (to be run in a QThread) which sits waiting for data to come through a Queue.Queue().
    It blocks until data is available, and once it got something from the queue, it sends it to the "MainThread" by emitting a Qt Signal 
    """
    mysignal = pyqtSignal(str)
    
    #----------------------------------------------------------------------
    def __init__(self, queue, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.queue = queue

    #----------------------------------------------------------------------
    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)

#----------------------------------------------------------------------
def redirect_stdout_to_queue(queue):
    """
    Redirect stdout and stderr to a queue (GUI purpose). 
    """
    if queue is not None:
        sys.stdout = WriteStream(queue)
        sys.stderr = WriteStream(queue)
        add_logger_handler(sys.stdout)