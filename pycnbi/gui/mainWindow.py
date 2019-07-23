#!/usr/bin/env python
#coding:utf-8

"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@gmail.com>
  Purpose: Defines the mainWindow class for the PyCNBI GUI.
  Created: 2/22/2019
"""

import os
import sys
import inspect
from os.path import expanduser
from importlib import import_module, reload
import multiprocessing as mp

from PyQt5.QtGui import QTextCursor, QFont
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QLine
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QAction, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QFormLayout, QWidget, QPushButton, QFrame, QSizePolicy

from ui_mainwindow import Ui_MainWindow
from streams import WriteStream, MyReceiver, redirect_stdout_to_queue
from readWriteTxt import read_params_from_txt
from pickedChannelsDialog import PickChannelsDialog, Channel_Select
from connectClass import PathFolderFinder, PathFileFinder, Connect_Directions, Connect_ComboBox, Connect_LineEdit, Connect_SpinBox, Connect_DoubleSpinBox, Connect_Modifiable_List, Connect_Modifiable_Dict,  Connect_Directions_Online, Connect_Bias

from pycnbi.utils import q_common as qc
from pycnbi.triggers.trigger_def import trigger_def

DEFAULT_PATH = os.environ['PYCNBI_SCRIPTS']


class cfg_class:
    def __init__(self, cfg):
        for key in dir(cfg):
            if key[0] == '_':
                continue
            setattr(self, key, getattr(cfg, key))

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

        self.cfg_struct = None      # loaded module containing all param possible values
        self.cfg_subject = None     # loaded module containing subject specific values
        self.paramsWidgets = {}     # dict of all the created parameters widgets

        self.load_ui_from_file()

        self.redirect_stdout()

        self.connect_signals_to_slots()

        # Terminal
        self.ui.textEdit_terminal.setReadOnly(1)
        font = QFont()
        font.setPointSize(10)
        self.ui.textEdit_terminal.setFont(font)

        # Define in which modality we are
        self.modality = None

    # ----------------------------------------------------------------------
    def redirect_stdout(self):
        """
        Create Queue and redirect sys.stdout to this queue.
        Create thread that will listen on the other end of the queue, and send the text to the textedit_terminal.
        """
        queue = mp.Queue()

        self.thread = QThread()

        self.my_receiver = MyReceiver(queue)
        self.my_receiver.mysignal.connect(self.on_terminal_append)
        self.my_receiver.moveToThread(self.thread)

        self.thread.started.connect(self.my_receiver.run)
        self.thread.start()

        redirect_stdout_to_queue(self.my_receiver.queue)


    #----------------------------------------------------------------------
    def load_ui_from_file(self):
        """
        Loads the UI interface from file.
        """
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


    #----------------------------------------------------------------------
    def clear_params(self):
        """
        Clear all previously loaded params widgets.
        """

        if self.ui.scrollAreaWidgetContents_Basics.layout() != None:
            QWidget().setLayout(self.ui.scrollAreaWidgetContents_Adv.layout())
            QWidget().setLayout(self.ui.scrollAreaWidgetContents_Basics.layout())


    # ----------------------------------------------------------------------
    def extract_value_from_module(self, key, values):
        """
        Extracts the subject's specific value associated with key.
        key = parameter name.
        values = list of all the parameters values.
        """
        for v in values:
            if v[0] == key:
                return v[1]

    # ----------------------------------------------------------------------
    def read_params_from_txt(self, txtFile):
        """
        Loads the parameters from a txt file.
        """
        folderPath = self.ui.lineEdit_pathSearch.text()
        file = open(folderPath + '/' + txtFile)
        params = file.read().splitlines()
        file.close()

        return params

    # ----------------------------------------------------------------------
    def disp_params(self, cfg_template_module, cfg_module):
        """
        Displays the parameters in the corresponding UI scrollArea.
        cfg = config module
        """

        self.clear_params()
        # Extract the parameters and their possible values from the template modules.
        params = inspect.getmembers(cfg_template_module)

        # Extract the chosen values from the subject's specific module.
        all_chosen_values = inspect.getmembers(cfg_module)

        filePath = self.ui.lineEdit_pathSearch.text()

        # Load channels
        if self.modality == 'train':
            subjectDataPath = '%s/%s/fif' % (os.environ['PYCNBI_DATA'], filePath.split('/')[-1])
            self.channels = read_params_from_txt(subjectDataPath, 'channelsList.txt')
        self.directions = ()

        # Iterates over the classes
        for par in range(2):
            param = inspect.getmembers(params[par][1])
            # Create layouts
            layout = QFormLayout()

            # Iterates over the list
            for p in param:
                # Remove useless attributes
                if '__' in p[0]:
                    continue

                # Iterates over the dict
                for key, values in p[1].items():
                    chosen_value = self.extract_value_from_module(key, all_chosen_values)

                    # For the feedback directions [offline and online].
                    if 'DIRECTIONS' in key:
                        self.directions = values

                        if self.modality is 'offline':
                            nb_directions = 4
                            directions = Connect_Directions(key, chosen_value, values, nb_directions)

                        elif self.modality is 'online':
                            cls_path = self.paramsWidgets['DECODER_FILE'].lineEdit_pathSearch.text()
                            cls = qc.load_obj(cls_path)
                            events = cls['cls'].classes_        # Finds the events on which the decoder has been trained on
                            events = list(map(int, events))
                            nb_directions = len(events)
                            chosen_events = [event[1] for event in chosen_value]
                            chosen_value = [val[0] for val in chosen_value]

                            # Need tdef to convert int to str trigger values
                            try:
                                [tdef.by_value(i) for i in events]
                            except:
                                trigger_file = self.extract_value_from_module('TRIGGER_FILE', all_chosen_values)
                                tdef = trigger_def(trigger_file)
                                # self.on_guichanges('tdef', tdef)
                                events = [tdef.by_value[i] for i in events]

                            directions = Connect_Directions_Online(key, chosen_value, values, nb_directions, chosen_events, events)

                        directions.signal_paramChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: directions})
                        layout.addRow(key, directions.l)


                    # For providing a folder path.
                    elif 'PATH' in key:
                        pathfolderfinder = PathFolderFinder(key, DEFAULT_PATH, chosen_value)
                        pathfolderfinder.signal_pathChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: pathfolderfinder})
                        layout.addRow(key, pathfolderfinder.layout)
                        continue

                    # For providing a file path.
                    elif 'FILE' in key:
                        pathfilefinder = PathFileFinder(key, chosen_value)
                        pathfilefinder.signal_pathChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: pathfilefinder})
                        layout.addRow(key, pathfilefinder.layout)
                        continue

                    # For the special case of choosing the trigger classes to train on
                    elif 'TRIGGER_DEF' in key:
                        trigger_file = self.extract_value_from_module('TRIGGER_FILE', all_chosen_values)
                        tdef = trigger_def(trigger_file)
                        # self.on_guichanges('tdef', tdef)
                        nb_directions = 4
                        #  Convert 'None' to real None (real None is removed when selected in the GUI)
                        tdef_values = [ None if i == 'None' else i for i in list(tdef.by_name) ]
                        directions = Connect_Directions(key, chosen_value, tdef_values, nb_directions)
                        directions.signal_paramChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: directions})
                        layout.addRow(key, directions.l)
                        continue

                    # To select specific electrodes
                    elif '_CHANNELS' in key or 'CHANNELS_' in key:
                        ch_select = Channel_Select(key, self.channels, chosen_value)
                        ch_select.signal_paramChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: ch_select})
                        layout.addRow(key, ch_select.layout)

                    elif 'BIAS' in key:
                        #  Add None to the list in case of no bias wanted
                        self.directions = tuple([None] + list(self.directions))
                        bias = Connect_Bias(key, self.directions, chosen_value)
                        bias.signal_paramChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: bias})
                        layout.addRow(key, bias.l)

                    # For all the int values.
                    elif values is int:
                        spinBox = Connect_SpinBox(key, chosen_value)
                        spinBox.signal_paramChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: spinBox})
                        layout.addRow(key, spinBox.w)
                        continue

                    # For all the float values.
                    elif values is float:
                        doublespinBox = Connect_DoubleSpinBox(key, chosen_value)
                        doublespinBox.signal_paramChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: doublespinBox})
                        layout.addRow(key, doublespinBox.w)
                        continue

                    # For parameters with multiple non-fixed values in a list (user can modify them)
                    elif values is list:
                        modifiable_list = Connect_Modifiable_List(key, chosen_value)
                        modifiable_list.signal_paramChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: modifiable_list})
                        layout.addRow(key, modifiable_list.frame)
                        continue

                    #  For parameters containing a string to modify
                    elif values is str:
                        lineEdit = Connect_LineEdit(key, chosen_value)
                        lineEdit.signal_paramChanged[str, str].connect(self.on_guichanges)
                        lineEdit.signal_paramChanged[str, type(None)].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: lineEdit})
                        layout.addRow(key, lineEdit.w)
                        continue

                    # For parameters with multiple fixed values.
                    elif type(values) is tuple:
                        comboParams = Connect_ComboBox(key, chosen_value, values)
                        comboParams.signal_paramChanged.connect(self.on_guichanges)
                        comboParams.signal_additionalParamChanged.connect(self.on_guichanges)
                        self.paramsWidgets.update({key: comboParams})
                        layout.addRow(key, comboParams.layout)
                        continue

                    # For parameters with multiple non-fixed values in a dict (user can modify them)
                    elif type(values) is dict:
                        try:
                            selection = chosen_value['selected']
                            comboParams = Connect_ComboBox(key, chosen_value, values)
                            comboParams.signal_paramChanged.connect(self.on_guichanges)
                            comboParams.signal_additionalParamChanged.connect(self.on_guichanges)
                            self.paramsWidgets.update({key: comboParams})
                            layout.addRow(key, comboParams.layout)

                        except:
                            modifiable_dict = Connect_Modifiable_Dict(key, chosen_value, values)
                            modifiable_dict.signal_paramChanged.connect(self.on_guichanges)
                            self.paramsWidgets.update({key: modifiable_dict})
                            layout.addRow(key, modifiable_dict.frame)
                        continue

                # Add a horizontal line to separate parameters' type.
                if p != param[-1]:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    separator.setFrameShadow(QFrame.Sunken)
                    layout.addRow(separator)

                # Display the parameters according to their types.
                if params[par][0] == 'Basic':
                    self.ui.scrollAreaWidgetContents_Basics.setLayout(layout)
                elif params[par][0] == 'Advanced':
                    self.ui.scrollAreaWidgetContents_Adv.setLayout(layout)


    # ----------------------------------------------------------------------
    def load_config(self, cfg_path, cfg_file, subj_file):
        """
        Dynamic loading of a config file.
        Format the lib to fit the previous developed pycnbi code if subject specific file (not for the templates).
        cfg_path: path to the folder containing the config file.
        cfg_file: config file to load.
        subj_file: true or false, if true it means it is the subject specific file. Format it.
        """
        if self.cfg_subject == None or cfg_file not in self.cfg_subject.__file__:
            # Dynamic loading
            sys.path.append(cfg_path)
            cfg_module = import_module(cfg_file)
        else:
            cfg_module = reload(self.cfg_subject)

        return cfg_module

    #----------------------------------------------------------------------
    def load_all_params(self, cfg_template, cfg_file):
        """
        Loads the params structure and assign the subject/s specific value.
        It also checks the sanity of the loaded params according to the protocol.
        """

        # Loads the template
        if self.cfg_struct == None or cfg_template not in self.cfg_struct.__file__:
            self.cfg_struct = self.load_struct_params(cfg_template)

        # Loads the subject's specific values
        self.cfg_subject = self.load_subject_params(cfg_file)

        # Display parameters on the GUI
        self.disp_params(self.cfg_struct, self.cfg_subject)

        # Check the parameters integrity
        self.cfg_subject = self.m.check_config(self.cfg_subject)



    # ----------------------------------------------------------------------
    def load_struct_params(self, cfg_template):
        """
        Load the parameters' structure from file depending on the choosen protocol.
        """
        cfg_template_path = os.environ['PYCNBI_ROOT']+'\pycnbi\config_files'
        cfg_template_module  = self.load_config(cfg_template_path, cfg_template, False)
        return cfg_template_module


    #----------------------------------------------------------------------
    def load_subject_params(self, cfg_file):
        """
        Loads the subject specific parameters' values from file and displays them.
        cfg_file: config file to load.
        """
        cfg_path = self.ui.lineEdit_pathSearch.text()+'/python'
        cfg_module  = self.load_config(cfg_path, cfg_file, True)
        return cfg_module


    @pyqtSlot(str, str)
    @pyqtSlot(str, bool)
    @pyqtSlot(str, list)
    @pyqtSlot(str, float)
    @pyqtSlot(str, int)
    @pyqtSlot(str, dict)
    @pyqtSlot(str, tuple)
    @pyqtSlot(str, type(None))
    # ----------------------------------------------------------------------
    def on_guichanges(self, name, new_Value):
        """
        Apply the modification to the corresponding param of the cfg module

        name = parameter name
        new_value = new str value to to change in the module
        """

        # In case of a dict containing several option (contains 'selected')
        try:
            tmp = getattr(self.cfg_subject, name)
            tmp['selected'] = new_Value['selected']
            tmp[new_Value['selected']] = new_Value[new_Value['selected']]
            setattr(self.cfg_subject, name, tmp)
        # In case of simple data format
        except:
            setattr(self.cfg_subject, name, new_Value)

        print("The parameter %s is %s" % (name, getattr(self.cfg_subject, name)))
        print("It's type is: %s \n" % type(getattr(self.cfg_subject, name)))


    # ----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_pathSearch(self):
        """
        Opens the File dialog window when the search button is pressed.
        """
        path_name = QFileDialog.getExistingDirectory(caption="Choose the subject's directory", directory=DEFAULT_PATH)
        self.ui.lineEdit_pathSearch.insert(path_name)


    # ----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_offline(self):
        """
        Loads the Offline parameters.
        """
        import pycnbi.protocols.train_mi as m

        self.m = m

        self.modality = 'offline'
        cfg_template = 'config_structure_train_mi'
        cfg_file = 'config_train_mi'

        self.load_all_params(cfg_template, cfg_file)


    # ----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_train(self):
        """
        Loads the Training parameters.
        """
        import pycnbi.decoder.trainer as m

        self.m = m

        self.modality = 'train'
        cfg_template = 'config_structure_trainer_mi'
        cfg_file = 'config_trainer_mi'

        self.load_all_params(cfg_template, cfg_file)


    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_online(self):
        """
        Loads the Online parameters.
        """
        import pycnbi.protocols.test_mi as m

        self.m = m

        self.modality = 'online'
        cfg_template = 'config_structure_test_mi'
        cfg_file = 'config_test_mi'

        self.load_all_params(cfg_template, cfg_file)


    #----------------------------------------------------------------------v
    @pyqtSlot()
    def on_click_start(self):
        """
        Launch the selected protocol. It can be Offline, Train or Online.
        """
        ccfg = cfg_class(self.cfg_subject)
        self.process = mp.Process(target=self.m.run, args=[ccfg, self.my_receiver.queue])
        self.process.start()


    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_stop(self):
        """
        Stop the protocol process
        """
        self.process.terminate()
        self.process.join()


    #----------------------------------------------------------------------
    @pyqtSlot(str)
    def on_terminal_append(self, text):
        """
        Writes to the QtextEdit_terminal the redirected stdout.
        """
        self.ui.textEdit_terminal.moveCursor(QTextCursor.End)
        self.ui.textEdit_terminal.insertPlainText(text)


    #----------------------------------------------------------------------
    def connect_signals_to_slots(self):
        """Connects the signals to the slots"""
        # Search  folder button
        self.ui.pushButton_Search.clicked.connect(self.on_click_pathSearch)
        # Offline button
        self.ui.pushButton_Offline.clicked.connect(self.on_click_offline)
        # Train button
        self.ui.pushButton_Train.clicked.connect(self.on_click_train)
        # Online button
        self.ui.pushButton_Online.clicked.connect(self.on_click_online)
        # Start button
        self.ui.pushButton_Start.clicked.connect(self.on_click_start)
        # Stop button
        self.ui.pushButton_Stop.clicked.connect(self.on_click_stop)

def main():
    #unittest.main()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
