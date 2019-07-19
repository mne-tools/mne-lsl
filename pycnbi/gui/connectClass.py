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


    # ----------------------------------------------------------------------
    @pyqtSlot(int)
    def on_modify(self, index):
        """
        Slot connected to comboBox param value change. Emits the pos and
        new value.
        """
        val = self.itemData(index)
        self.signal_paramChanged.emit(self.pos, val)


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
        Add the possibles values found in the structure file to a QComboBox and
        add it to a QFormLayout. Highlight the subject's specific value.
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

        templateChoices.signal_paramChanged.connect(self.on_modify)
        
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
        
        self.signal_paramChanged.emit(self.paramName, self.chosen_value)

########################################################################
class Connect_Directions_Online(QObject):
    """
    This class is used to connect the directions modifications in case of the online
    modality. It modifies the module according to the newly selected
    parameter value.
    """

    signal_paramChanged = pyqtSignal(str, list)
    
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
        self.directions.signal_paramChanged.connect(self.on_modify)
        
        self.events = Connect_Directions('DIR_EVENTS', chosen_events, events, nb_directions)
        self.events.signal_paramChanged.connect(self.on_modify)
        
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
                
        self.signal_paramChanged.emit(self.directions.paramName, updatedList)
    

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
        self.frame = QFrame()
        self.chosen_value = chosenValue
        
        self.add_To_ComboBox(all_values, chosenValue)
        
    # ----------------------------------------------------------------------
    def add_To_ComboBox(self, values, chosenValue):
        """
        Add the possibles values found in the structure file to a QComboBox and
        add it to a QFormLayout. Highlight the subject's specific value.
        
        values = list of values.
        chosenValue = subject's specific value.
        """
        self.templateChoices = QComboBox()
        self.additionalParams = list()
        self.layout = QHBoxLayout()
                     
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
                p.signal_paramChanged.connect(self.on_modify)
                self.additionalParams.append(p)
                self.templateChoices.addItem(str(key_val), key_val)

            elif val is list:
                chosen_additionalParams = chosenValue[key_val]
                p = Connect_Modifiable_List(key_val, chosen_additionalParams)
                p.signal_paramChanged.connect(self.on_modify)
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
            self.layout.addWidget(p.frame)
        
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
            self.signal_paramChanged.emit(self.paramName, val)
        
        # In case of additional params for at least one selection
        else:  
            i = 0
            for p in self.additionalParams:
                if p.paramName == val:
                    i = 1
                    p.frame.show()
                    self.signal_additionalParamChanged.emit(self.paramName, {'selected': p.paramName, p.paramName: p.chosen_value})
                else:
                    p.frame.hide()
                    pass
            
            #  In case of additional params but not for the selected one (e.g in case of None)
            if i == 0:
                self.signal_additionalParamChanged.emit(self.paramName, {'selected': self.templateChoices.itemText(index), self.templateChoices.itemText(index): val})

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
            self.spinBox.w.setDisabled(True)
    
        self.l.addWidget(self.directions.templateChoices)
        self.l.addWidget(self.spinBox.w)
        
        self.directions.signal_paramChanged.connect(self.on_modify)
        self.directions.signal_paramChanged.connect(self.on_modify)
        
        self.spinBox.signal_paramChanged.connect(self.on_modify)

        
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
                self.spinBox.w.setValue(0.0)
                self.spinBox.w.setDisabled(True)
                self.signal_paramChanged[str, type(None)].emit(self.paramName, new_Value)
            else:
                self.spinBox.w.setDisabled(False)
        
        elif 'value' in paramName:
            self.signal_paramChanged[str, tuple].emit(self.paramName, (self.selected_direction, new_Value))
    

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
        self.w.setSingleStep(0.05)
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

    signal_paramChanged = pyqtSignal([str, str], [str, type(None)])

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
        self.chosen_value = chosen_value
        
        #  To fit the disp_params function of mainWindow. 
        self.frame = self.w

    @pyqtSlot()
    # ----------------------------------------------------------------------
    def on_modify(self):
        """
        Changes the module according to the new value written in the lineEdit
        """
        text = self.w.text()
        if text == 'None':
            text = None
        self.signal_paramChanged[str, type(text)].emit(self.paramName, text)

    
########################################################################
class Connect_Modifiable_List(QObject):
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
        self.frame = QFrame()
        self.frame.setStyleSheet("margin:0; padding:0")
        
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
                layout.addWidget(tempWidget.w.w)
        
        self.frame.setLayout(layout)
        # self.layout = layout

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
        layout = QHBoxLayout()
        self.frame = QFrame()
        # self.frame.setStyleSheet("margin:0; padding:0")
        self.frame.setContentsMargins(0, 0, 0, 0);
                
        for key, value in content_dict.items():

            # Add a horizontal line to separate parameters' type.
            add_v_separator(layout)

            if value is int:
                spinBox = Connect_SpinBox(key, chosen_value[key])
                spinBox.signal_paramChanged.connect(self.on_modify)
                self.paramWidgets.append(spinBox)
                label = QLabel(key)
                # label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                # spinBox.w.setFixedWidth(50)
                layout.addWidget(label)
                layout.addWidget(spinBox.w)

            elif value is float:
                doublespinBox = Connect_DoubleSpinBox(key, chosen_value[key])
                self.paramWidgets.append(doublespinBox)
                doublespinBox.signal_paramChanged.connect(self.on_modify)
                label = QLabel(key)
                # label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                # doublespinBox.w.setFixedWidth(50)
                layout.addWidget(label)
                layout.addWidget(doublespinBox.w)

            elif value is str:
                lineEdit = Connect_LineEdit(key, chosen_value[key])
                lineEdit.signal_paramChanged.connect(self.on_modify)
                self.paramWidgets.append(lineEdit)
                label = QLabel(key)
                # label.setFixedWidth(80)
                label.setAlignment(Qt.AlignCenter)
                # lineEdit.w.setFixedWidth(50)
                layout.addWidget(label)
                layout.addWidget(lineEdit.w)
                
            elif type(value) is tuple:
                comboBox = Connect_ComboBox(key, chosen_value[key], value)
                comboBox.signal_paramChanged.connect(self.on_modify)
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
        self.frame.setLayout(layout)


    @pyqtSlot(str, int)
    @pyqtSlot(str, float)
    @pyqtSlot(str, str)
    @pyqtSlot(str, tuple)
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
