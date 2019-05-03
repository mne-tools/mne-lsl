"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@gmail.com>
  Purpose: Defines the path search class.
  
  It is required due to the dynamic loading of parameters/widgets.
  One cannot connect the clicked of the button to write in the QLineEdit
  
  Created: 4/09/2019
"""
import os
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QFileDialog, QLineEdit
from PyQt5.QtCore import pyqtSignal, QObject

########################################################################
class PathFolderFinder(QObject):
    """
    Its has a layout containing a QPushButton and a QLineEdit 
    to write the selected folder path from the QFileDialog.
    """

    signal_pathChanged = pyqtSignal(str, str)
    
    # ----------------------------------------------------------------------
    def __init__(self, paramName, defaultPath, defaultValue):
        """
        Constructor
        name = name of the parameter in the module 
        defaultPath = directory when opening QFileDialog
        defaultValue = folder path taken from subject specific file and
        written in the lineEdit at creation
        """
        super().__init__()

        self.name = paramName
        self.defaultPath = defaultPath
        self.layout = QHBoxLayout()

        self.button = QPushButton('Search')
        self.layout.addWidget(self.button)

        self.lineEdit_pathSearch = QLineEdit(defaultValue)
        self.layout.addWidget(self.lineEdit_pathSearch)
        
        self.button.clicked.connect(self.on_click_pathSearch)

    #----------------------------------------------------------------------
    def on_click_pathSearch(self):
        """
        Slot connected to the button clicked signal. It opens a QFileDialog 
        and adds the selected path to the lineEdit.
        """
        path_name = QFileDialog.getExistingDirectory(caption="Choose the subject's directory", directory=self.defaultPath)
        self.lineEdit_pathSearch.setText(path_name)
        self.signal_pathChanged.emit(self.name, path_name)
        
        
########################################################################
class PathFileFinder(QObject):
    """
    Its attribute is a layout containing a QPushButton and a QLineEdit 
    to write the selected file path from the QFileDialog.
    """
    #----------------------------------------------------------------------
    def __init__(self, defaultValue):
        """
        Constructor
        defaultValue = folder path taken from subject specific file and
        written in the lineEdit at creation.
        """
        self.defaultPath =  os.environ['PYCNBI_ROOT']+'\pycnbi\\triggers'
        self.layout = QHBoxLayout()
        
        self.button = QPushButton('Search')
        self.layout.addWidget(self.button)
        
        self.lineEdit_pathSearch = QLineEdit(defaultValue)
        self.layout.addWidget(self.lineEdit_pathSearch)
        
        self.button.clicked.connect(self.on_click_pathSearch)

        
    # ----------------------------------------------------------------------
    def on_click_pathSearch(self):
        """
        Slot connected to the button clicked signal. It opens a QFileDialog 
        and adds the selected path to the lineEdit.
        """
        path_name = QFileDialog.getOpenFileName(caption="Choose the subject's directory", directory=self.defaultPath)
        self.lineEdit_pathSearch.setText(path_name[0])
        

