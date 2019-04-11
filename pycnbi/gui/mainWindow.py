#!/usr/bin/env python
#coding:utf-8

"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@gmail.com>
  Purpose: Defines the mainWindow class for the PyCNBI GUI.
  Created: 2/22/2019
"""

#import unittest
import sys
from importlib import import_module
from os.path import expanduser
from queue import Queue
import os 
import inspect

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QAction, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QFormLayout, QWidget, QPushButton, QFrame, QSizePolicy
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QLine
from PyQt5.QtGui import QColor, QTextCursor, QFont


from ui_mainwindow import Ui_MainWindow
from pathFinders import PathFolderFinder, PathFileFinder
from streams import WriteStream, MyReceiver

DEFAULT_PATH = os.environ['PYCNBI_SCRIPTS']

########################################################################
class MainWindow(QMainWindow):
    """
    Defines the mainWindow class for the PyCNBI GUI.
    """
    #----------------------------------------------------------------------
    def __init__(self):
        """
        Constructor.
        """
        super(MainWindow, self).__init__()
        self.load_UiFromFile()
        self.redirect_StdOut()
        self.connect_Signals_To_Slots()
        
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
    def clear_Params(self):
        """
        Clear all previously loaded params widgets.
        """
        layout = self.ui.scrollAreaWidgetContents_Basics.layout()
        if layout is not None:
            for i in reversed(range(layout.count())):
                layout.itemAt(i).widget().setParent(None)
        
        layout = self.ui.scrollAreaWidgetContents_2_Adv.layout()
        if layout is not None:
            for i in reversed(range(layout.count())): 
                layout.itemAt(i).widget().setParent(None) 
        
    
    #----------------------------------------------------------------------
    def add_To_ComboBox(self, values, chosenValue):
        """
        Add the possibles values found in the template file to a QComboBox and
        add it to a QFormLayout.
        values = list of values.
        """
        templateChoices = QComboBox()
        
        # Iterates over the possible choices
        for val in values:
            templateChoices.addItem(str(val))
            if val is chosenValue:
                templateChoices.setCurrentText(str(val))
        
        return templateChoices
    
    #----------------------------------------------------------------------
    def extract_SubjectValue(self, key, values):
        """
        Extracts the subject's specific value associated with key.
        key = parameter name.
        values = list of all the parameters values. 
        """        
        for v in values:
            if v[0] is key:
                return v[1]
               
    
    #----------------------------------------------------------------------
    def disp_Params(self, cfg_template_module, cfg_module):
        """
        Displays the parameters in the corresponding UI scrollArea.
        cfg = config module
        """
        
        self.clear_Params()
        # Extract the parameters and their possible values from the template modules.
        params = inspect.getmembers(cfg_template_module)
        
        # Extract the chosen values from the subject's specific module.
        all_chosen_values = inspect.getmembers(cfg_module)
               
        # Iterates over the classes            
        for par in range(2):
            
            param = inspect.getmembers(params[par][1])
            #chosen_values = inspect.getmembers(all_chosen_values[par][1])
            
            # Create layouts            
            layout = QFormLayout()
            
            # Iterates over the list
            for p in param:
                
                # Remove useless attributes
                if '__' in p[0]:
                    continue
                
                # Iterates over the dict
                for key, value in p[1].items():
                    
                    chosen_value = self.extract_SubjectValue(key, all_chosen_values)
                    
                    if 'DIRECTIONS' in key:
                        l = QHBoxLayout()                        
                        for i in range(len(chosen_value)):
                            l.addWidget(self.add_To_ComboBox(value, chosen_value[i]))
                        for i in range(4-len(chosen_value)):
                            l.addWidget(self.add_To_ComboBox(value, None))
                        layout.addRow(key, l)                  
                    elif 'PATH' in key:
                        self.pathFolderFinder = PathFolderFinder(DEFAULT_PATH, chosen_value)
                        layout.addRow(key, self.pathFolderFinder.layout)
                    elif 'DEF' in key:
                        self.pathFileFinder = PathFileFinder(chosen_value)
                        layout.addRow(key, self.pathFileFinder.layout)                        
                    elif value is None:
                        layout.addRow(key, QLineEdit(str(chosen_value))) 
                    else:
                        templateChoices = self.add_To_ComboBox(value, chosen_value)
                        #templateChoices.setSizePolicy(QSizePolicy.Fixed , QSizePolicy.Fixed)
                        layout.addRow(key, templateChoices)
                
                # Add a horizontal line to separate parameters' type.
                if p is not param[-1]:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    separator.setFrameShadow(QFrame.Sunken)
                    layout.addRow(separator)
                   
                # Display the parameters according to their types.
                if params[par][0] == 'Basic':
                    self.ui.scrollAreaWidgetContents_Basics.setLayout(layout)
                elif params[par][0] == 'Advanced':
                    self.ui.scrollAreaWidgetContents_2_Adv.setLayout(layout)
                        
                
    #----------------------------------------------------------------------
    def load_Config(self, cfg_path, cfg_file, subj_file):
        """
        Dynamic loading of a config file.
        Format the lib to fit the previous developed pycnbi code if subject specific file (not for the templates).
        cfg_path: path to the folder containing the config file.
        cfg_file: config file to load.
        subj_file: true or false, if true it means it is the subject specific file. Format it.
        """
        # Dynamic loading
        sys.path.append(cfg_path)
        cfg_module = import_module(cfg_file)
        
        # Format the lib to fit the previous developed pycnbi code if subject specific file.
        #if subj_file:
            #self.cfg = type('cfg', (cfg_module.Advanced, cfg_module.Basic), dict())
        
        return cfg_module
        
    #----------------------------------------------------------------------
    def load_AllParams(self, cfg_template ,cfg_file):
        """
        Loads the params structure and assign the subject/s specific value.
        It also checks the sanity of the loaded params according to the protocol.
        """
        
        # Loads the template
        self.cfg_struct = self.load_Struct_Params(cfg_template)
        
        # Loads the subject's specific values
        self.cfg_subject = self.load_Subject_Params(cfg_file)
        
        # Check the parameters integrity
        self.cfg_subject = self.m.check_config(self.cfg_subject)
        
        # Display parameters on the GUI
        self.disp_Params(self.cfg_struct, self.cfg_subject)
            
    
    #----------------------------------------------------------------------
    def load_Struct_Params(self, cfg_template):
        """
        Load the parameters' structure from file depending on the choosen protocol.
        """
        cfg_template_path = os.environ['PYCNBI_ROOT']+'\pycnbi\config_files'
        cfg_template_module  = self.load_Config(cfg_template_path, cfg_template, False) 
        return cfg_template_module
        
    
    #----------------------------------------------------------------------
    def load_Subject_Params(self, cfg_file):
        """
        Loads the subject specific parameters' values from file and displays them.
        cfg_file: config file to load.
        """
        cfg_path = self.ui.lineEdit_pathSearch.text()+'/python'
        cfg_module  = self.load_Config(cfg_path, cfg_file, True)               
        return cfg_module 
        
    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_pathSearch(self):
        """
        Opens the File dialog window when the search button is pressed.
        """
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
        cfg_template = 'config_structure_train_mi'
        cfg_file = 'config_train_mi'
        self.load_AllParams(cfg_template, cfg_file)
        
        
    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_Train(self):
        """
        Loads the Training parameters.
        """
        import pycnbi.decoder.trainer as m
        self.m = m
        cfg_file = 'config_trainer_mi'
        #self.load_AllParams(cfg_file)
        
        
    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_Online(self):
        """
        Loads the Online parameters.
        """
        import pycnbi.protocols.test_mi as m
        self.m = m
        cfg_file = 'config_test_mi'
        #self.load_AllParams(cfg_file)
    
    
    #----------------------------------------------------------------------v
    @pyqtSlot()
    def on_click_Start(self):
        """
        Launch the selected protocol. It can be Offline, Train or Online. 
        """
        self.m.run(self.cfg_subject)
        
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
        # Search  folder button
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