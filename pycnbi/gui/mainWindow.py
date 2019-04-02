#!/usr/bin/env python
#coding:utf-8

"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@fcbg.ch>
  Purpose: Defines the mainWindow class for the PyCNBI GUI.
  Created: 2/22/2019
"""

#import unittest
import sys
from importlib import import_module
from os.path import expanduser
from queue import Queue

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QAction, QLabel
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QColor, QTextCursor, QFont


from ui_mainwindow import Ui_MainWindow
from Streams import WriteStream, MyReceiver

DEFAULT_PATH = "C:/Users/adesvachez/git/pycnbi_local/z2"

class MainWindow(QMainWindow):    
    #----------------------------------------------------------------------
    def __init__(self):
        """
        Constructor.
        """
        super(MainWindow, self).__init__()
        self.load_UiFromFile()
        self.redirect_StdOut()
        self.connect_Signals_To_Slots()
        
        # Default path 
        self.ui.lineEdit_pathSearch.insert(DEFAULT_PATH)
        # Terminal
        self.ui.textEdit_terminal.setReadOnly(1)
        font = QFont()
        font.setPointSize(6)
        self.ui.textEdit_terminal.setFont(font)
        
        
        
    #----------------------------------------------------------------------
    def redirect_StdOut(self):
        """
        Create Queue and redirect sys.stdout to this queue.
        Create thread that will listen on the other end of the queue, and send the text to the textedit_terminal.
        """
        queue = Queue()
        sys.stdout = WriteStream(queue)
        sys.stderr = WriteStream(queue)
        
        self.thread = QThread()
        self.my_receiver = MyReceiver(queue)
        self.my_receiver.mysignal.connect(self.on_terminal_append)
        self.my_receiver.moveToThread(self.thread)
        self.thread.started.connect(self.my_receiver.run)
        self.thread.start()
        
        
    #----------------------------------------------------------------------
    def load_UiFromFile(self):
        """
        Loads the UI interface from file.
        """
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        
    #----------------------------------------------------------------------
    def disp_Params(self, cfg):
        """
        Displays the parameters in the corresponding UI scrollArea.
        cfg = config module
        """
        import inspect
        params = inspect.getmembers(cfg)
        # Clear all previous widgets        
        if self.ui.verticalLayout_Basic.count() > 0:
            for i in reversed(range(self.ui.verticalLayout_Basic.count())):
                self.ui.verticalLayout_Basic.itemAt(i).widget().setParent(None)

        if self.ui.verticalLayout_Adv.count() > 0:
            for i in reversed(range(self.ui.verticalLayout_Adv.count())): 
                self.ui.verticalLayout_Adv.itemAt(i).widget().setParent(None)
        
        # Iterates over the classes            
        for par in params:
            param = inspect.getmembers(par[1])
            # Iterates over the dict
            for p in param:
                if '__' in p[0]:
                    break
                value2display = ""
                # Iterates over the tuples
                for i in range(len(p)):
                    value2display += str(p[i])+" "
                
                # Display the parameters according to their types.
                if par[0] == 'Basic':
                    self.ui.verticalLayout_Basic.addWidget(QLabel(value2display))
                elif par[0] == 'Advanced':
                    self.ui.verticalLayout_Adv.addWidget(QLabel(value2display))
                
    #----------------------------------------------------------------------
    def load_Params(self, cfg_file):
        """
        Loads the parameters from source file and displays them.
        """
        cfg_path = self.ui.lineEdit_pathSearch.text()+'/python/'
        
        # Dynamic loading
        sys.path.append(cfg_path)
        cfg_module = import_module(cfg_file)
        
        # Format the lib to fit the previous developed pycnbi code.
        self.cfg = type('cfg', (cfg_module.Advanced, cfg_module.Basic), dict())

        # Check the parameters integrity
        self.cfg = self.m.check_config(self.cfg)
        
        # Display parameters on the GUI
        self.disp_Params(cfg_module)
        
    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_pathSearch(self):
        """
        Opens the File dialog window when the search button is pressed.
        """
        #path_name = QFileDialog.getExistingDirectory(caption="Choose the subject's directory", directory=expanduser("~"))
        path_name = QFileDialog.getExistingDirectory(caption="Choose the subject's directory", directory=DEFAULT_PATH)
        self.ui.lineEdit_pathSearch.insert(path_name)
        
        
    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_Offline(self):
        """ 
        Loads the Offline parameters. 
        """
        import pycnbi.protocols.train_mi as m
        self.m = m
        cfg_file = 'config_train_mi'
        self.load_Params(cfg_file)
        
        
    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_Train(self):
        """
        Loads the Training parameters.
        """
        import pycnbi.decoder.trainer as m
        self.m = m
        cfg_file = 'config_trainer_mi'
        self.load_Params(cfg_file)
        
        
    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_Online(self):
        """
        Loads the Online parameters.
        """
        import pycnbi.protocols.test_mi as m
        self.m = m
        cfg_file = 'config_test_mi'
        self.load_Params(cfg_file)
    
    
    #----------------------------------------------------------------------v
    @pyqtSlot()
    def on_click_Start(self):
        """
        Launch the selected protocol. It can be Offline, Train or Online. 
        """
        self.m.run(self.cfg)
        
    #----------------------------------------------------------------------
    @pyqtSlot(str)
    def on_terminal_append(self, text):
        """
        Writes to the QtextEdit_terminal the redirected stdout.
        """
        self.ui.textEdit_terminal.moveCursor(QTextCursor.End)
        self.ui.textEdit_terminal.insertPlainText(text)     

        
    #----------------------------------------------------------------------
    def connect_Signals_To_Slots(self):
        """Connects the signals to the slots"""
        # Search button
        self.ui.pushButton_Search.clicked.connect(self.on_click_pathSearch)
        # Offline button
        self.ui.pushButton_Offline.clicked.connect(self.on_click_Offline)
        # Train button
        self.ui.pushButton_Train.clicked.connect(self.on_click_Train)
        # Online button
        self.ui.pushButton_Online.clicked.connect(self.on_click_Online)
        # Start button
        self.ui.pushButton_Start.clicked.connect(self.on_click_Start)
        
def main():    
    #unittest.main()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()