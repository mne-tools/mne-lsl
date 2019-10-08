import sys
import multiprocessing as mp
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit

from neurodecode import add_logger_handler
from neurodecode.utils import pycnbi_utils as pu

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
    def __init__(self, queue):
        QObject.__init__(self)
        self.queue = queue

    #----------------------------------------------------------------------
    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal[str].emit(text)

        
########################################################################
class GuiTerminal(QDialog):
    """
    Open a QDialog and display the terminal output of a specific process 
    """

    #----------------------------------------------------------------------
    def __init__(self, logger, verbosity, width):
        """Constructor"""
        super().__init__()
        
        self.textEdit = QTextEdit()
        self.setWindowTitle('Recording')
        self.resize(width, 100)
        self.textEdit.setReadOnly(1)
        
        l = QVBoxLayout()
        l.addWidget(self.textEdit)
        self.setLayout(l)
        
        self.redirect_stdout(logger, verbosity)
        
    # ----------------------------------------------------------------------
    def redirect_stdout(self, logger, verbosity):
        """
        Create Queue and redirect sys.stdout to this queue.
        Create thread that will listen on the other end of the queue, and send the text to the textedit_terminal.
        """
        queue = mp.Queue()

        self.thread = QThread()

        self.my_receiver = MyReceiver(queue)
        self.my_receiver.mysignal[str].connect(self.on_terminal_append)
        self.my_receiver.moveToThread(self.thread)

        self.thread.started.connect(self.my_receiver.run)
        self.thread.start()
        self.textEdit.insertPlainText('Waiting for the recording to start...\n')
        self.show()
        
    
    @pyqtSlot(str)
    #----------------------------------------------------------------------
    def on_terminal_append(self, text):
        """
        Writes to the QtextEdit_terminal the redirected stdout.
        """
        self.textEdit.moveCursor(QTextCursor.End)
        self.textEdit.insertPlainText(text)

########################################################################
class search_lsl_streams_thread(QThread):
    """
    Look for available lsl streams and emit the signal to share the list
    """
    
    signal_lsl_found = pyqtSignal(list)
    
    #----------------------------------------------------------------------
    def __init__(self, state, logger):
        """
        Constructor
        """
        super().__init__()
        self.state = state
        self.logger = logger
    
    #----------------------------------------------------------------------
    def run(self):
        amp_list, streamInfos = pu.list_lsl_streams(state=self.state, logger=self.logger, ignore_markers=False)
        
        with self.state.get_lock():
            self.state.value = 0
            
        if amp_list:
            self.signal_lsl_found[list].emit(amp_list)

#----------------------------------------------------------------------
def redirect_stdout_to_queue(logger, queue, verbosity):
    """
    Redirect stdout and stderr to a queue (GUI purpose). 
    """
    if queue is not None:

        sys.stdout = WriteStream(queue)
        # sys.stderr = WriteStream(queue)
        add_logger_handler(logger, sys.stdout, verbosity)
