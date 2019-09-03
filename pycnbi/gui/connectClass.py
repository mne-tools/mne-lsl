"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@gmail.com>
  Purpose: Defines the path search class.
  
  It is required due to the dynamic loading of parameters/widgets.
  One cannot connect the clicked of the button to write in the QLineEdit
  
  Created: 4/09/2019
"""

import os
from glob import glob
from pathlib import Path 
from shutil import copy2
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, \
     QFrame, QDialog, QFormLayout, QDialogButtonBox
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, Qt

########################################################################
class QComboBox_Directions(QComboBox):
    """
    Overloads the QComboBox to contain its position in the directions
    list
    """

    signal_paramChanged = pyqtSignal([int, object])

    #----------------------------------------------------------------------
    def __init__(self, pos):
        """
        Constructor
        
        pos = QComboBox position in the directions list
        """
        super().__init__()
        self.pos = pos
        self.currentIndexChanged[int].connect(self.on_modify)
        self.setFocusPolicy(Qt.StrongFocus)
    
    # ----------------------------------------------------------------------
    def wheelEvent(self, *args, **kwargs):
        if self.hasFocus():
            return QComboBox.wheelEvent(self, *args, **kwargs)
        else:    
            pass

    # ----------------------------------------------------------------------
    @pyqtSlot(int)
    def on_modify(self, index):
        """
        Slot connected to comboBox param value change. Emits the pos and
        new value.
        """
        val = self.itemData(index)
        self.signal_paramChanged[int, object].emit(self.pos, val)


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
        
        nb_val = range(len(chosen_value)) 
        for i in nb_val:
            self.l.addWidget(self.add_To_ComboBox(all_Values, chosen_value[i], i))
            
            # Add a vertical separator
            if i != nb_directions:
                add_v_separator(self.l)            
        
        nb_val = range(len(chosen_value), nb_directions) 
        for i in nb_val:
            self.l.addWidget(self.add_To_ComboBox(all_Values, None, i))
            
            # Add a vertical separator
            if i != nb_val[-1]:
                add_v_separator(self.l)                        


    # ----------------------------------------------------------------------
    def add_To_ComboBox(self, values, chosenValue, pos):
        """
        Add the possibles values found in the structure file to a QComboBox.
        Highlight the subject's specific value.
        values = list of values.
        chosenValue = subject's specific value.
        pos = QComboBox position in the directions list
        """
        templateChoices = QComboBox_Directions(pos)

        # Iterates over the possible choices
        for val in values:
            templateChoices.addItem(str(val), val)
            if val == chosenValue:
                templateChoices.setCurrentText(val)

        templateChoices.signal_paramChanged[int, object].connect(self.on_modify)
        
        return templateChoices

        
    @pyqtSlot(int, object)
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
        
        try:
            self.chosen_value.remove(None)
        except:
            pass
        
        self.signal_paramChanged[str, list].emit(self.paramName, self.chosen_value)

########################################################################
class Connect_Directions_Online(QObject):
    """
    This class is used to connect the directions modifications in case of the online
    modality. It modifies the module according to the newly selected
    parameter value.
    """

    signal_paramChanged = pyqtSignal([str, list])
    
    #----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value, all_Values, nb_directions, chosen_events, events):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter directions.
        all_Values = list of all possible directions
        nb_directions = number of directions to add.
        chosen_events = the subject's specific parameter events linked to the corresponding direction.
        events = events 
        """
        super().__init__()
        
        self.directions = Connect_Directions(paramName, chosen_value, all_Values, nb_directions)
        self.directions.signal_paramChanged[str, list].connect(self.on_modify)
        
        self.events = Connect_Directions('DIR_EVENTS', chosen_events, events, nb_directions)
        self.events.signal_paramChanged[str, list].connect(self.on_modify)
        
        self.l = QVBoxLayout()
        self.l.addLayout(self.directions.l)
        self.l.addLayout(self.events.l)
    
    @pyqtSlot(str, list)
    #----------------------------------------------------------------------
    def on_modify(self, paramName, new_Values):
        """
        Slot connected to the changes in the directions.
        """
        updatedList = []
        
        if paramName == self.directions.paramName:
            for i in range(len(new_Values)):
                updatedList.append((new_Values[i], self.events.chosen_value[i]))
                
        elif paramName == self.events.paramName:
            for i in range(len(new_Values)):
                updatedList.append((self.directions.chosen_value[i], new_Values[i]))
                
        self.signal_paramChanged[str, list].emit(self.directions.paramName, updatedList)
        
########################################################################
class ComboBox(QComboBox):
    """
    Overload of QCombobox to overwrite wheelEvent()
    """

    #----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
    
    # ----------------------------------------------------------------------
    def wheelEvent(self, *args, **kwargs):
        if self.hasFocus():
            return QComboBox.wheelEvent(self, *args, **kwargs)
        else:    
            pass 

########################################################################
class Connect_ComboBox(QObject):
    """
    This class is used to connect the comboBox modifications at the
    GUI level. It modifies the module according to the newly selected
    parameter value.
    """

    signal_paramChanged = pyqtSignal([str, object])
    signal_additionalParamChanged = pyqtSignal([str, dict])

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosenValue, all_values):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosenValue = the subject's specific parameter value.
        all_Values = list of all possible values for a parameter
        """
        super().__init__()

        self.paramName = paramName
        # self.frame = QFrame()
        self.chosen_value = chosenValue
       
        self.add_To_ComboBox(all_values, chosenValue)
        
    # ----------------------------------------------------------------------
    def add_To_ComboBox(self, values, chosenValue):
        """
        Add the possibles values found in the structure file to a QComboBox. \
        Highlight the subject's specific value.
        
        values = list of values.
        chosenValue = subject's specific value.
        """
        self.templateChoices = ComboBox()
        self.additionalParams = list()
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
                     
        # Special case of a dict 
        if type(values) is dict:
            dict_values = values
            values = tuple(values)
                        
        # Iterates over the possible choices
        for val in values:
            
            try:
                key_val = val
                val = dict_values[val]
                pass
            except:
                pass
            
            # additional parameters to modify
            if type(val) is dict:              
                content_dict = val
                chosen_additionalParams = chosenValue[key_val]
                p = Connect_Modifiable_Dict(key_val, chosen_additionalParams, content_dict)
                p.signal_paramChanged[str, dict].connect(self.on_modify)
                self.additionalParams.append(p)
                self.templateChoices.addItem(str(key_val), key_val)

            elif val is list:
                chosen_additionalParams = chosenValue[key_val]
                p = Connect_Modifiable_List(key_val, chosen_additionalParams)
                p.signal_paramChanged[str, list].connect(self.on_modify)
                self.additionalParams.append(p)
                self.templateChoices.addItem(str(key_val), key_val)
                
            elif val is str:
                chosen_additionalParams = chosenValue[key_val]
                p = Connect_LineEdit(key_val, chosen_additionalParams)
                p.signal_paramChanged[str, str].connect(self.on_modify)
                self.additionalParams.append(p)
                self.templateChoices.addItem(str(key_val), key_val)             
                
            else:
                self.templateChoices.addItem(str(key_val), val)
            
        if type(chosenValue) is dict:
            chosenValue = chosenValue['selected']
        
        self.layout.addWidget(self.templateChoices)      
        for p in self.additionalParams:
            self.layout.addWidget(p)
        
        index = self.templateChoices.findText(str(chosenValue))        
        if index != -1:
            self.templateChoices.setCurrentIndex(index)
            self.on_select(index)

        self.templateChoices.currentIndexChanged[int].connect(self.on_select)
    
    @pyqtSlot(int)
    # ----------------------------------------------------------------------
    def on_select(self, index):
        """
        Slot connected to comboBox param change
        """       
        val = self.templateChoices.itemData(index)
        
        # In case of a simple comboBox without additional params for all possible selections
        if not self.additionalParams:
            self.signal_paramChanged[str, object].emit(self.paramName, val)
        
        # In case of additional params for at least one selection
        else:  
            i = 0
            for p in self.additionalParams:
                if p.paramName == val:
                    i = 1
                    p.show()
                    self.signal_additionalParamChanged[str, dict].emit(self.paramName, {'selected': p.paramName, p.paramName: p.chosen_value})
                else:
                    p.hide()
                    pass
            
            #  In case of additional params but not for the selected one (e.g in case of None)
            if i == 0:
                self.signal_additionalParamChanged[str, dict].emit(self.paramName, {'selected': self.templateChoices.itemText(index), self.templateChoices.itemText(index): val})

    @pyqtSlot(str, dict)
    @pyqtSlot(str, list)
    @pyqtSlot(str, str)
    # @pyqtSlot(str, float)
    # @pyqtSlot(str, int)
    #----------------------------------------------------------------------
    def on_modify(self, key, p):
        """
        Slot connected on the additional parameters changes
        """
        self.signal_additionalParamChanged[str, dict].emit(self.paramName, {'selected': key, key: p})
        
        
########################################################################
class Connect_Bias(QObject):
    """
    This class is used for the feedback bias toward one classes.
    It modifies the module according to the newly parameter value.
    """
    
    signal_paramChanged = pyqtSignal([str, object])
    
    #----------------------------------------------------------------------
    def __init__(self, paramName, directions, chosen_value):
        """
        Constructor
        
        directions = directions on which one can apply the bias. Directions
        correspond to the classifier's classes.
        """
        super().__init__()
        
        self.paramName = paramName
        self.selected_direction = chosen_value[0]
        self.l = QHBoxLayout()
        
        if chosen_value[0] is not None:
            self.directions = Connect_ComboBox('direction', chosen_value[0], directions)
            self.spinBox = Connect_DoubleSpinBox('value', chosen_value[1])
        else:
            self.directions = Connect_ComboBox('direction', chosen_value, directions)
            self.spinBox = Connect_DoubleSpinBox('value', 0.0)
            self.spinBox.setDisabled(True)
    
        self.l.addWidget(self.directions.templateChoices)
        self.l.addWidget(self.spinBox)
        
        self.directions.signal_paramChanged[str, object].connect(self.on_modify)        
        self.spinBox.signal_paramChanged[str, float].connect(self.on_modify)

        
    @pyqtSlot(str, str)
    @pyqtSlot(str, float)
    @pyqtSlot(str, type(None))
    #----------------------------------------------------------------------
    def on_modify(self, paramName, new_Value):
        """
        Slot connected to the bias's direction or bias's value change
        """
        if 'direction' in paramName:
            self.selected_direction = new_Value
            if new_Value is None:
                self.setValue(0.0)
                self.spinBox.setDisabled(True)
                self.signal_paramChanged[str, type(None)].emit(self.paramName, new_Value)
            else:
                self.spinBox.setDisabled(False)
        
        elif 'value' in paramName:
            self.signal_paramChanged[str, tuple].emit(self.paramName, (self.selected_direction, new_Value))
    

########################################################################
class Connect_SpinBox(QSpinBox):
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
        self.setMinimum(-1)
        self.setMaximum(10000)
        
        self.setValue(chosen_value)
        self.editingFinished.connect(self.on_modify)
        self.setFocusPolicy(Qt.StrongFocus)
    
    # ----------------------------------------------------------------------
    def wheelEvent(self, *args, **kwargs):
        if self.hasFocus():
            return QComboBox.wheelEvent(self, *args, **kwargs)
        else:    
            pass

    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the SpinBox
        """
        self.signal_paramChanged[str, int].emit(self.paramName, self.value())


########################################################################
class Connect_DoubleSpinBox(QDoubleSpinBox):
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
        self.setSingleStep(0.05)
        self.setValue(chosen_value)
        self.editingFinished.connect(self.on_modify)
        self.setFocusPolicy(Qt.StrongFocus)
    
    # ----------------------------------------------------------------------
    def wheelEvent(self, *args, **kwargs):
        if self.hasFocus():
            return QDoubleSpinBox.wheelEvent(self, *args, **kwargs)
        else:    
            pass
        
    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the DoubleSpinBox
        """
        self.signal_paramChanged[str, float].emit(self.paramName, self.value())


########################################################################
class Connect_LineEdit(QLineEdit):
    """
    This class is used to connect the lineEdit modifications at the
    GUI level. It modifies the module according to the newly 
    parameter value.
    """

    signal_paramChanged = pyqtSignal([str, str], [str, type(None)])

    #----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter value.
        """
        super().__init__(str(chosen_value))
        
        self.paramName = paramName

        self.editingFinished.connect(self.on_modify)
        self.chosen_value = chosen_value
        

    @pyqtSlot()
    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the lineEdit
        """
        text = self.text()
        if text == 'None' or text == '':
            text = None
        self.signal_paramChanged[str, type(text)].emit(self.paramName, text)

    
########################################################################
class Connect_Modifiable_List(QFrame):
    """
    This class is used in case of lists containing modifiable contents.
    It modifies the module according to the newly parameter value.
    """

    signal_paramChanged = pyqtSignal([str, list])
    paramWidgets = []

    # ----------------------------------------------------------------------
    def __init__(self, paramName, chosen_value):
        """Constructor
        
        paramName = Name of the parameter corresponding to the widget to create 
        chosen_value = the subject's specific parameter value.
        """

        super().__init__()
        self.paramName = paramName
        self.chosen_value = chosen_value
        layout = QHBoxLayout()
        self.tempWidgets = []
        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        
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
                    vLayout.addWidget(tempWidget.w)
                
                vLayout.setContentsMargins(0, 0, 0, 0)
                layout.addLayout(vLayout)

                # Add a vertical line to separate parameters' type.
                if chosen_value[k] != chosen_value[-1]:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.VLine)
                    separator.setFrameShadow(QFrame.Sunken)
                    layout.addWidget(separator)

            else:
                tempWidget = tempWidget_for_Modifiable_List(chosen_value[k], k, None)
                tempWidget.signal_paramChanged[list, int].connect(self.on_modify)
                tempWidget.signal_paramChanged[list, float].connect(self.on_modify)
                tempWidget.signal_paramChanged[list, str].connect(self.on_modify)
                self.tempWidgets.append(tempWidget)
                layout.addWidget(tempWidget.w)
        
        self.setLayout(layout)

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
class tempWidget_for_Modifiable_List(QObject):
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
            self.w.signal_paramChanged[str, str].connect(self.on_modify)
            self.w.signal_paramChanged[str, type(None)].connect(self.on_modify)

        elif type(listElements) is int:
            self.w = Connect_SpinBox(str(None), listElements)
            self.w.signal_paramChanged[str, int].connect(self.on_modify)
    
        elif type(listElements) is float:
            self.w = Connect_DoubleSpinBox(str(None), listElements)
            self.w.signal_paramChanged[str, float].connect(self.on_modify)

        elif type(listElements[pos2]) is str:
            self.w = Connect_LineEdit(str(pos2), listElements[pos2])
            self.w.signal_paramChanged[str, str].connect(self.on_modify)
            self.w.signal_paramChanged[str, type(None)].connect(self.on_modify)

        elif type(listElements[pos2]) is int:
            self.w = Connect_SpinBox(str(pos2), listElements[pos2])
            self.w.signal_paramChanged[str, int].connect(self.on_modify)

        elif type(listElements[pos2]) is float:
            self.w = Connect_DoubleSpinBox(str(pos2), listElements[pos2])
            self.w.signal_paramChanged[str, float].connect(self.on_modify)
        

    @pyqtSlot(str, int)
    @pyqtSlot(str, float)
    @pyqtSlot(str, str)
    @pyqtSlot(str, type(None))
    # ----------------------------------------------------------------------
    def on_modify(self, inerPos, value):
        """
        Connect to the inner loop widget. Emit the received info + outer loop
        position
        """
        self.signal_paramChanged[list, type(value)].emit([self.outerPos, inerPos], value)


########################################################################
class Connect_Modifiable_Dict(QFrame):
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
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0);
                
        for key, value in content_dict.items():

            # Add a horizontal line to separate parameters' type.
            add_v_separator(layout)

            if value is int:
                spinBox = Connect_SpinBox(key, chosen_value[key])
                spinBox.signal_paramChanged[str, int].connect(self.on_modify)
                self.paramWidgets.append(spinBox)
                label = QLabel(key)
                # label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                # spinBox.w.setFixedWidth(50)
                layout.addWidget(label)
                layout.addWidget(spinBox)

            elif value is float:
                doublespinBox = Connect_DoubleSpinBox(key, chosen_value[key])
                self.paramWidgets.append(doublespinBox)
                doublespinBox.signal_paramChanged[str, float].connect(self.on_modify)
                label = QLabel(key)
                # label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                # doublespinBox.w.setFixedWidth(50)
                layout.addWidget(label)
                layout.addWidget(doublespinBox)

            elif value is str:
                lineEdit = Connect_LineEdit(key, chosen_value[key])
                lineEdit.signal_paramChanged[str, str].connect(self.on_modify)
                lineEdit.signal_paramChanged[str, type(None)].connect(self.on_modify)
                self.paramWidgets.append(lineEdit)
                label = QLabel(key)
                # label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                # lineEdit.w.setFixedWidth(50)
                layout.addWidget(label)
                layout.addWidget(lineEdit)

            elif value is list:
                p = Connect_Modifiable_List(key, chosen_value[key])
                p.signal_paramChanged[str, list].connect(self.on_modify)
                self.paramWidgets.append(p)
                label = QLabel(key)
                label.setAlignment(Qt.AlignCenter)
                layout.addWidget(label)
                layout.addWidget(p)                

            elif type(value) is tuple:
                comboBox = Connect_ComboBox(key, chosen_value[key], value)
                comboBox.signal_paramChanged[str, object].connect(self.on_modify)
                self.paramWidgets.append(comboBox)
                label = QLabel(key)
                layout.addWidget(label)
                layout.addWidget(comboBox.templateChoices)                
                
            # Add a vertical line
            if key == list(content_dict.items())[-1][0]:
                add_v_separator(layout)

        layout.addStretch(1)
        # self.setLayout(layout)
        # self.layout = layout
        self.setLayout(layout)


    @pyqtSlot(str, int)
    @pyqtSlot(str, float)
    @pyqtSlot(str, str)
    @pyqtSlot(str, tuple)
    @pyqtSlot(str, list)
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
        self.signal_pathChanged[str, str].emit(self.name, path_name)
        
        
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
        self.defaultPath = os.environ['PYCNBI_ROOT']
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
        self.signal_pathChanged[str, str].emit(self.name, path_name[0])

########################################################################
class Connect_NewSubject(QDialog):
    """
    Allow to create a new subject folders in PYCNBI_SCRIPTS and PYCNBI_DATA when the
    pushButton_new is pressed and its name is provided
    """

    signal_error = pyqtSignal(str)
    
    #----------------------------------------------------------------------
    def __init__(self, parent, lineEdit_pathSearch):
        """Constructor"""
        super().__init__(parent)
        
        # Ui lineEdit where to write the selected path.
        self.lineEdit_pathSearch = lineEdit_pathSearch
        
        protocols_path = Path(os.environ['PYCNBI_ROOT']) / 'pycnbi' / 'config_files' 
        protocols = self.find_protocols(os.fspath(protocols_path))
        
        formLayout = QFormLayout()
        lineEdit = QLineEdit()
        formLayout.addRow('Subject ID:', lineEdit)        
        formLayout.addRow('Protocol:', self.create_widgets(protocols))
        
        buttonBox = QDialogButtonBox()
        buttonBox.addButton(QDialogButtonBox.Ok)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        
        l = QVBoxLayout()
        l.addLayout(formLayout)
        l.addWidget(buttonBox)
        
        self.setLayout(l)
        
        buttonBox.accepted.connect(self.create_subject_folders)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
               
        self.show()
            
    #----------------------------------------------------------------------
    def find_protocols(self, path):
        """
        Find the possible protocols, defined in the config_files folder
        """
        folders = next(os.walk(path))[1]
        
        return [f for f in folders if '__' not in f] 
    
    #----------------------------------------------------------------------
    def create_widgets(self, protocols):
        """
        Create an QComboBox containing the protocols
        """
        comboBox = QComboBox()
        comboBox.addItems(protocols)
        
        return comboBox
    
    #----------------------------------------------------------------------
    @pyqtSlot()
    def create_subject_folders(self):
        """
        Create one folder in PYCNBI_SCRIPTS and one in PYCNBI_DATA
        """
        subject_id = self.layout().itemAt(0).itemAt(1).widget().text()
        protocol = self.layout().itemAt(0).itemAt(3).widget().currentText()
        
        try:
            # for PYCNBI_SCRIPTS
            scripts_path = Path(os.environ['PYCNBI_SCRIPTS']) / (subject_id + '-' + protocol)            
            os.mkdir(scripts_path)
              
            # Add path to the lineEdit_pathSearch
            self.lineEdit_pathSearch.setText(os.fspath(scripts_path))        
            
            # for PYCNBI_DATA
            data_path = Path(os.environ['PYCNBI_DATA']) / (subject_id + '-' + protocol)
            os.mkdir(data_path)            
            
            # Copy the protocol config_files
            files_path = Path(os.environ['PYCNBI_ROOT']) / 'pycnbi' / 'config_files' / protocol / 'template_files'
            files = glob(os.fspath(files_path / "*.py") , recursive=False)
            config_files = [f for f in files if 'structure' not in f]
            for f in config_files:
                fileName = os.path.split(f)[1].split('.')[0]
                copy2(f, (os.fspath(scripts_path / fileName) + ('_' + subject_id + '-' + protocol +'.py')))
            
        except Exception as e:
            self.signal_error[str].emit(str(e))
            #error_dialog = QErrorMessage(self)
            #error_dialog.showMessage(str(e))                 

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
