"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@gmail.com>
  Purpose: Defines the path search class.
  
  It is required due to the dynamic loading of parameters/widgets.
  One cannot connect the clicked of the button to write in the QLineEdit
  
  Created: 4/09/2019
"""
import os
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QFileDialog, QLineEdit, QComboBox, QSpinBox
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot

########################################################################
class Connect_Directions(QObject):
    """
    This class is used to connect the directions modifications at the
    GUI level. It modifies the module according to the newly selected
    parameter value.
    """

    signal_paramChanged = pyqtSignal(str, str)
    l = QHBoxLayout()

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value, all_Values, nb_directions):
        """Constructor
        Creates nb_directions directions, list the possible values and select the chosen_value.
        
        chosen_value = the subject's specific parameter value.
        all_Values = list of all possible values for a parameter
        nb_directions = number of directions to add.
        """
        super().__init__()

        self.paramName = paramName
        
        for i in range(len(chosen_value)):
            self.l.addWidget(self.add_To_ComboBox(all_Values, chosen_value[i]))

        for i in range(nb_directions - len(chosen_value)):
            self.l.addWidget(self.add_To_ComboBox(all_Values, None))


    #----------------------------------------------------------------------
    def add_To_ComboBox(self, values, chosenValue):
        """
        Add the possibles values found in the structure file to a QComboBox and
        add it to a QFormLayout. Highlight the subject's specific value.
        values = list of values.
        chosenValue = subject's specific value.
        """
        templateChoices = QComboBox()
        
        # Iterates over the possible choices
        for val in values:
            templateChoices.addItem(val)
            if val == chosenValue:
                templateChoices.setCurrentText(val)

        templateChoices.currentIndexChanged[str].connect(self.on_modify)
        
        return templateChoices

        
    @pyqtSlot(str)
    # ----------------------------------------------------------------------
    def on_modify(self, new_Value):
        """
        Slot connected to comboBox param value change
        """
        self.signal_paramChanged.emit(self.paramName, new_Value)


########################################################################
class Connect_ComboBox(QObject):
    """
    This class is used to connect the comboBox modifications at the
    GUI level. It modifies the module according to the newly selected
    parameter value.
    """

    signal_paramChanged = pyqtSignal([str, str], [str, list], [str, bool])

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosenValue, all_values):
        """Constructor"""
        super().__init__()

        self.paramName = paramName
        self.templateChoices = self.add_To_ComboBox(all_values, chosenValue)
        
        
    # ----------------------------------------------------------------------
    def add_To_ComboBox(self, values, chosenValue):
        """
        Add the possibles values found in the structure file to a QComboBox and
        add it to a QFormLayout. Highlight the subject's specific value.
        values = list of values.
        chosenValue = subject's specific value.
        """
        templateChoices = QComboBox()
        
        # Iterates over the possible choices
        for val in values:
            templateChoices.addItem(str(val), val)
            if val == chosenValue:
                index = templateChoices.findData(chosenValue)
                if index != -1:
                    templateChoices.setCurrentIndex(index)

        templateChoices.currentIndexChanged[int].connect(self.on_modify)
        
        return templateChoices
    
    
    @pyqtSlot(int)
    # ----------------------------------------------------------------------
    def on_modify(self, index):
        """
        Slot connected to comboBox param value change
        """
        val = self.templateChoices.itemData(index)
        
        if type(val) is list:
            self.signal_paramChanged[str, list].emit(self.paramName, self.templateChoices.itemData(index))
        elif type(val) is bool:
            self.signal_paramChanged[str, bool].emit(self.paramName, self.templateChoices.itemData(index))
        elif type(val) is str:
            self.signal_paramChanged[str, str].emit(self.paramName, self.templateChoices.itemData(index))


########################################################################
class Connect_SpinBox(QObject):
    """
    This class is used to connect the lineEdit modifications at the
    GUI level. It modifies the module according to the newly 
    parameter value.
    """

    signal_paramChanged = pyqtSignal(str, float)
    
    #----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value):
        """Constructor"""
        super().__init__()

        self.paramName = paramName
        self.spinBox = QSpinBox()
        self.spinBox.setValue(chosen_value)
        self.spinBox.editingFinished.connect(self.on_modify)


    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the DoubleSpinBox
        """
        self.signal_paramChanged.emit(self.paramName, self.spinBox.value())



########################################################################


class Connect_LineEdit(QObject):
    """
    This class is used to connect the lineEdit modifications at the
    GUI level. It modifies the module according to the newly 
    parameter value.
    """

    signal_paramChanged = pyqtSignal(str, str)

    #----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value):
        """Constructor"""
        super().__init__()
        
        self.paramName = paramName
        self.lineEdit = QLineEdit(chosen_value)
        self.lineEdit.editingFinished.connect(self.on_modify)

    @pyqtSlot()
    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the lineEdit
        """
        self.signal_paramChanged.emit(self.paramName, self.lineEdit.text())
    
    

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

    @pyqtSlot()
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

    signal_pathChanged = pyqtSignal(str, str)
    
    #----------------------------------------------------------------------
    def __init__(self, paramName, defaultValue):
        """
        Constructor
        name = name of the parameter in the module
        defaultValue = folder path taken from subject specific file and
        written in the lineEdit at creation.
        """
        super().__init__()

        self.name = paramName
        self.defaultPath = os.environ['PYCNBI_ROOT'] + '\pycnbi\\triggers'
        self.layout = QHBoxLayout()
        
        self.button = QPushButton('Search')
        self.layout.addWidget(self.button)
        
        self.lineEdit_pathSearch = QLineEdit(defaultValue)
        self.layout.addWidget(self.lineEdit_pathSearch)
        
        self.button.clicked.connect(self.on_click_pathSearch)

    @pyqtSlot()
    # ----------------------------------------------------------------------
    def on_click_pathSearch(self):
        """
        Slot connected to the button clicked signal. It opens a QFileDialog 
        and adds the selected path to the lineEdit.
        """
        path_name = QFileDialog.getOpenFileName(caption="Choose the subject's directory", directory=self.defaultPath)
        self.lineEdit_pathSearch.setText(path_name[0])
        self.signal_pathChanged.emit(self.name, path_name[0])

