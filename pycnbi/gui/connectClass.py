"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@gmail.com>
  Purpose: Defines the path search class.
  
  It is required due to the dynamic loading of parameters/widgets.
  One cannot connect the clicked of the button to write in the QLineEdit
  
  Created: 4/09/2019
"""

import os
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QWidget, QFrame
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, Qt

########################################################################
class QComboBox_Directions(QComboBox):
    """
    Overloads the QComboBox to contain its position in the directions
    list
    """

    signal_paramChanged = pyqtSignal([int, str])

    #----------------------------------------------------------------------
    def __init__(self, pos):
        """
        Constructor
        
        pos = QComboBox position in the directions list
        """
        super().__init__()
        self.pos = pos
        self.currentIndexChanged[str].connect(self.on_modify)


    # ----------------------------------------------------------------------
    @pyqtSlot(str)
    def on_modify(self, new_Value):
        """
        Slot connected to comboBox param value change. Emits the pos and
        new value.
        """
        self.signal_paramChanged.emit(self.pos, new_Value)


########################################################################
class Connect_Directions(QObject):
    """
    This class is used to connect the directions modifications at the
    GUI level. It modifies the module according to the newly selected
    parameter value.
    """

    signal_paramChanged = pyqtSignal(str, list)

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value, all_Values, nb_directions):
        """Constructor
        Creates nb_directions directions, list the possible values and select the chosen_value.
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter values.
        all_Values = list of all possible values for a parameter
        nb_directions = number of directions to add.
        """
        super().__init__()
        self.paramName = paramName
        self.l = QHBoxLayout()
        self.chosen_value = chosen_value
 
        for i in range(len(chosen_value)):
            self.l.addWidget(self.add_To_ComboBox(all_Values, chosen_value[i], i))

        for i in range(len(chosen_value), nb_directions):
            self.l.addWidget(self.add_To_ComboBox(all_Values, None, i))


    # ----------------------------------------------------------------------
    def add_To_ComboBox(self, values, chosenValue, pos):
        """
        Add the possibles values found in the structure file to a QComboBox and
        add it to a QFormLayout. Highlight the subject's specific value.
        values = list of values.
        chosenValue = subject's specific value.
        pos = QComboBox position in the directions list
        """
        templateChoices = QComboBox_Directions(pos)

        # Iterates over the possible choices
        for val in values:
            templateChoices.addItem(val)
            if val == chosenValue:
                templateChoices.setCurrentText(val)

        templateChoices.signal_paramChanged.connect(self.on_modify)
        
        return templateChoices

        
    @pyqtSlot(int, str)
    # ----------------------------------------------------------------------
    def on_modify(self, pos, new_Value):
        """
        Slot connected to comboBox param value change
        
        pos = QComboBox position in the directions list
        new_value = direction new_value
        """

        if pos >= len(self.chosen_value):
            self.chosen_value.append(new_Value)
        else:
            self.chosen_value[pos] = (new_Value)
            
        self.signal_paramChanged.emit(self.paramName, self.chosen_value)


########################################################################
class Connect_ComboBox(QObject):
    """
    This class is used to connect the comboBox modifications at the
    GUI level. It modifies the module according to the newly selected
    parameter value.
    """

    signal_paramChanged = pyqtSignal([str, str], [str, list], [str, bool], [str, type(None)])

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosenValue, all_values):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter value.
        all_Values = list of all possible values for a parameter
        """
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
        # templateChoices.setEditable(True)
        # templateChoices.setInsertPolicy(3)

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
        self.signal_paramChanged[str, type(val)].emit(self.paramName, val)


########################################################################
class Connect_SpinBox(QObject):
    """
    This class is used to connect the SpinBox modifications at the
    GUI level. It modifies the module according to the newly 
    parameter value.
    """

    signal_paramChanged = pyqtSignal(str, int)
    
    #----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter value.
        """
        super().__init__()

        self.paramName = paramName
        self.w = QSpinBox()
        self.w.setMinimum(-1)
        self.w.setMaximum(10000)
        
        self.w.setValue(chosen_value)
        self.w.editingFinished.connect(self.on_modify)


    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the SpinBox
        """
        self.signal_paramChanged.emit(self.paramName, self.w.value())


########################################################################
class Connect_DoubleSpinBox(QObject):
    """
    This class is used to connect the doubleSpinBox modifications at the
    GUI level. It modifies the module according to the newly 
    parameter value.
    """

    signal_paramChanged = pyqtSignal(str, float)
    
    #----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter value.
        """
        super().__init__()

        self.paramName = paramName
        self.w = QDoubleSpinBox()
        self.w.setSingleStep(0.1)
        self.w.setValue(chosen_value)
        self.w.editingFinished.connect(self.on_modify)


    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the DoubleSpinBox
        """
        self.signal_paramChanged.emit(self.paramName, self.w.value())


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
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter value.
        """
        super().__init__()
        
        self.paramName = paramName
        self.w = QLineEdit(str(chosen_value))
        self.w.editingFinished.connect(self.on_modify)

    @pyqtSlot()
    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the lineEdit
        """
        self.signal_paramChanged.emit(self.paramName, self.w.text())

    
########################################################################
class Connect_Modifiable_List(QObject):
    """
    This class is used in case of lists containing modifiable contents.
    It modifies the module according to the newly parameter value.
    """

    signal_paramChanged = pyqtSignal([str, list])
    paramWidgets = []

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value, content_list):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter value.
        content_list = list of values.
        """

        super().__init__()
        self.paramName = paramName
        self.chosen_value = chosen_value
        self.hlayout = QHBoxLayout()
        self.tempWidgets = []

        # first list
        for k in range(len(chosen_value)):
            vLayout = QVBoxLayout()
            
            #  Check if there is an inner list
            if type(chosen_value[k]) is list:

                # Iterate inside the inner list
                for i in range(len(chosen_value[k])):
                    tempWidget = tempWidget_for_Modifiable_List(chosen_value[k], k, i)
                    tempWidget.signal_paramChanged[list, int].connect(self.on_modify)
                    tempWidget.signal_paramChanged[list, float].connect(self.on_modify)
                    tempWidget.signal_paramChanged[list, str].connect(self.on_modify)
                    self.tempWidgets.append(tempWidget)
                    vLayout.addWidget(tempWidget.w.w)

                self.hlayout.addLayout(vLayout)

                # Add a horizontal line to separate parameters' type.
                if chosen_value[k] != chosen_value[-1]:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.VLine)
                    separator.setFrameShadow(QFrame.Sunken)
                    self.hlayout.addWidget(separator)

            else:
                tempWidget = tempWidget_for_Modifiable_List(chosen_value[k], k, None)
                tempWidget.signal_paramChanged[list, int].connect(self.on_modify)
                tempWidget.signal_paramChanged[list, float].connect(self.on_modify)
                tempWidget.signal_paramChanged[list, str].connect(self.on_modify)
                self.tempWidgets.append(tempWidget)
                self.hlayout.addWidget(tempWidget.w.w)


    @pyqtSlot(list, int)
    @pyqtSlot(list, float)
    @pyqtSlot(list, str)
    # ----------------------------------------------------------------------
    def on_modify(self, position, val):
        """
        Changes the module according to any new value written in the
        modifiable list.
        """
        if position[1] == 'None':
            self.chosen_value[position[0]] = val
        else:
            self.chosen_value[position[0]][int(position[1])] = val
        self.signal_paramChanged[str, list].emit(self.paramName, self.chosen_value)


########################################################################
class tempWidget_for_Modifiable_List(QWidget):
    """
    This class is used by the Connect_Modifiable_List because of the possible
    subList structure. One needs to know in which inner list to change properly
    the module.
    """

    signal_paramChanged = pyqtSignal([list, int], [list, float], [list, str])

    # ----------------------------------------------------------------------
    def __init__(self, listElements, pos1, pos2):
        """Constructor
        
        listElements = element contained in the list to display
        pos1 = position in the outer list
        pos2 = position in the inner list
        layout = layout to add the newly created widgets
        """
        super().__init__()

        self.outerPos = pos1

        if type(listElements) is str:
            self.w = Connect_LineEdit(str(None), listElements)
            self.w.signal_paramChanged.connect(self.on_modify)

        elif type(listElements) is int:
            self.w = Connect_SpinBox(str(None), listElements)
            self.w.signal_paramChanged.connect(self.on_modify)
    
        elif type(listElements) is float:
            self.w = Connect_DoubleSpinBox(str(None), listElements)
            self.w.signal_paramChanged.connect(self.on_modify)

        elif type(listElements[pos2]) is str:
            self.w = Connect_LineEdit(str(pos2), listElements[pos2])
            self.w.signal_paramChanged.connect(self.on_modify)

        elif type(listElements[pos2]) is int:
            self.w = Connect_SpinBox(str(pos2), listElements[pos2])
            self.w.signal_paramChanged.connect(self.on_modify)

        elif type(listElements[pos2]) is float:
            self.w = Connect_DoubleSpinBox(str(pos2), listElements[pos2])
            self.w.signal_paramChanged.connect(self.on_modify)



    @pyqtSlot(str, int)
    @pyqtSlot(str, float)
    @pyqtSlot(str, str)
    # ----------------------------------------------------------------------
    def on_modify(self, inerPos, value):
        """
        Connect to the inner loop widget. Emit the received info + outer loop
        position
        """
        self.signal_paramChanged[list, type(value)].emit([self.outerPos, inerPos], value)


########################################################################
class Connect_Modifiable_Dict(QObject):
    """
    This class is used in case of dicts containing modifiable contents.
    It modifies the module according to the newly parameter value.
    """

    signal_paramChanged = pyqtSignal([str, dict])
    paramWidgets = []

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value, content_dict):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific dict parameter values.
        content_dict = list of params in the dict
        """

        super().__init__()
        self.paramName = paramName
        self.chosen_value = chosen_value
        self.layout = QHBoxLayout()

        for key, value in content_dict.items():

            # Add a horizontal line to separate parameters' type.
            add_v_separator(self.layout)

            if value is int:
                spinBox = Connect_SpinBox(key, chosen_value[key])
                spinBox.signal_paramChanged.connect(self.on_modify)
                self.paramWidgets.append(spinBox)
                label = QLabel(key)
                label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                spinBox.w.setFixedWidth(80)
                self.layout.addWidget(label)
                self.layout.addWidget(spinBox.w)

            elif value is float:
                doublespinBox = Connect_DoubleSpinBox(key, chosen_value[key])
                self.paramWidgets.append(doublespinBox)
                doublespinBox.signal_paramChanged.connect(self.on_modify)
                label = QLabel(key)
                label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                doublespinBox.w.setFixedWidth(80)
                self.layout.addWidget(label)
                self.layout.addWidget(doublespinBox.w)

            elif value is str:
                lineEdit = Connect_LineEdit(key, chosen_value[key])
                lineEdit.signal_paramChanged.connect(self.on_modify)
                self.paramWidgets.append(lineEdit)
                label = QLabel(key)
                label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                lineEdit.w.setFixedWidth(80)
                self.layout.addWidget(label)
                self.layout.addWidget(lineEdit.w)

            # Add a horizontal line to separate parameters' type.
            if key == list(content_dict.items())[-1][0]:
                add_v_separator(self.layout)

        self.layout.addStretch(1)


    @pyqtSlot(str, int)
    @pyqtSlot(str, float)
    @pyqtSlot(str, str)
    # ----------------------------------------------------------------------
    def on_modify(self, key, val):
        """
        Changes the module according to any new value written in the
        modifiable dict.    
        """
        
        self.chosen_value[key] = val
        self.signal_paramChanged[str, dict].emit(self.paramName, self.chosen_value)


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
        
        paramName = name of the parameter in the module
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
        
        paramName = name of the parameter in the module
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

#----------------------------------------------------------------------
def add_v_separator(layout):
    """
    Add a vertical separrator to the layout
    """
    separator = QFrame()
    separator.setFrameShape(QFrame.VLine)
    separator.setFrameShadow(QFrame.Sunken)
    layout.addWidget(separator)

# ----------------------------------------------------------------------
def add_h_separator(layout):
    """
    Add a horizontal separrator to the layout
    """
    separator = QFrame()
    separator.setFrameShape(QFrame.VLine)
    separator.setFrameShadow(QFrame.Sunken)
    layout.addWidget(separator)
