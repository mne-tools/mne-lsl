#!/usr/bin/env python
#coding:utf-8

"""
  Author:  Arnaud Desvachez --<arnaud.desvachez@gmail.com>
  Purpose: Defines the mainWindow class for the NeuroDecode GUI.
  Created: 2/22/2019
"""

import os
import sys
import time
import inspect
import logging
import multiprocessing as mp
from threading import Thread
from datetime import datetime
from glob import glob
from pathlib import Path
from importlib import import_module, reload

from PyQt5.QtGui import QTextCursor, QFont
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QFormLayout, QWidget, \
     QFrame, QErrorMessage

from ui_mainwindow import Ui_MainWindow
from streams import MyReceiver, redirect_stdout_to_queue, GuiTerminal, search_lsl_streams_thread
from readWriteFile import read_params_from_file, save_params_to_file
from pickedChannelsDialog import Channel_Select
from connectClass import PathFolderFinder, PathFileFinder, Connect_Directions, Connect_ComboBox, \
     Connect_LineEdit, Connect_SpinBox, Connect_DoubleSpinBox, Connect_Modifiable_List, \
     Connect_Modifiable_Dict,  Connect_Directions_Online, Connect_Bias, Connect_NewSubject

from neurodecode import logger, init_logger
from neurodecode.utils import q_common as qc
from neurodecode.utils import pycnbi_utils as pu
from neurodecode.triggers.trigger_def import trigger_def
import neurodecode.stream_viewer.stream_viewer as viewer
import neurodecode.stream_recorder.stream_recorder as recorder

class cfg_class:
    def __init__(self, cfg):
        for key in dir(cfg):
            if key[0] == '_':
                continue
            setattr(self, key, getattr(cfg, key))

########################################################################
class MainWindow(QMainWindow):
    """
    Defines the mainWindow class for the neurodecode GUI.
    """
    
    hide_recordTerminal = pyqtSignal(bool)
    signal_error = pyqtSignal(str)
    
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

        # Define in which modality we are
        self.modality = None
        
        # Recording process
        self.record_terminal = None
        self.recordLogger = logging.getLogger('recorder')
        self.recordLogger.propagate = False
        init_logger(self.recordLogger)        
        
        # To display errors
        self.error_dialog = QErrorMessage(self)
        self.error_dialog.setWindowTitle('Warning')
        
        # Mp sharing variables
        self.record_state = mp.Value('i', 0)
        self.protocol_state = mp.Value('i', 0)
        self.lsl_state = mp.Value('i', 0)
        self.viewer_state = mp.Value('i', 0)
        
        # Disable widgets
        self.ui.groupBox_Modality.setEnabled(False)
        self.ui.groupBox_Launch.setEnabled(False)
        

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

        redirect_stdout_to_queue(logger, self.my_receiver.queue, 'INFO')


    #----------------------------------------------------------------------
    def load_ui_from_file(self):
        """
        Loads the UI interface from file.
        """
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Protocol terminal
        self.ui.textEdit_terminal.setReadOnly(1)
        font = QFont()
        font.setPointSize(10)
        self.ui.textEdit_terminal.setFont(font)
        
        # Viewer button
        self.ui.pushButton_Viewer.setEnabled(False)


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
        if self.modality == 'trainer':
            subjectDataPath = Path('%s/%s/%s/fif' % (os.environ['NEUROD_DATA'], filePath.split('/')[-2], filePath.split('/')[-1]))
            self.channels = read_params_from_file(subjectDataPath, 'channelsList.txt')    
                
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
                            chosen_events = [event[1] for event in chosen_value]
                            chosen_value = [val[0] for val in chosen_value]
                            nb_directions = len(chosen_value) 
                            directions = Connect_Directions_Online(key, chosen_value, values, nb_directions, chosen_events, [None])
                            
                        directions.signal_paramChanged[str, list].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: directions})
                        layout.addRow(key, directions.l)                

                    # For the special case of choosing the trigger classes to train on
                    elif 'TRIGGER_DEF' in key:
                        
                        trigger_def = Connect_Directions(key, chosen_value, [None], 4)
                        trigger_def.signal_paramChanged[str, list].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: trigger_def})
                        layout.addRow(key, trigger_def.l)

                    # For providing a folder path.
                    elif 'PATH' in key:
                        pathfolderfinder = PathFolderFinder(key, os.environ['NEUROD_DATA'], chosen_value)
                        pathfolderfinder.signal_pathChanged[str, str].connect(self.on_guichanges)
                        pathfolderfinder.signal_error[str].connect(self.on_error)
                        self.paramsWidgets.update({key: pathfolderfinder})
                        layout.addRow(key, pathfolderfinder.layout)
                        
                        if not chosen_value:
                            self.signal_error[str].emit(key + ' is empty! Provide a path before starting.')
                        continue

                    # For providing a file path.
                    elif 'FILE' in key:
                        pathfilefinder = PathFileFinder(key, chosen_value)
                        pathfilefinder.signal_pathChanged[str, str].connect(self.on_guichanges)
                        pathfilefinder.signal_error[str].connect(self.on_error)
                        self.paramsWidgets.update({key: pathfilefinder})
                        layout.addRow(key, pathfilefinder.layout)
                        
                        if not chosen_value:
                            self.signal_error[str].emit(key + ' is empty! Provide a file before starting.')                        

                    # To select specific electrodes
                    elif '_CHANNELS' in key or 'CHANNELS_' in key:
                        ch_select = Channel_Select(key, self.channels, chosen_value)
                        ch_select.signal_paramChanged[str, list].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: ch_select})
                        layout.addRow(key, ch_select.layout)

                    elif 'BIAS' in key:
                        #  Add None to the list in case of no bias wanted
                        self.directions = tuple([None] + list(self.directions))
                        bias = Connect_Bias(key, self.directions, chosen_value)
                        bias.signal_paramChanged[str, object].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: bias})
                        layout.addRow(key, bias.l)

                    # For all the int values.
                    elif values is int:
                        spinBox = Connect_SpinBox(key, chosen_value)
                        spinBox.signal_paramChanged[str, int].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: spinBox})
                        layout.addRow(key, spinBox)

                    # For all the float values.
                    elif values is float:
                        doublespinBox = Connect_DoubleSpinBox(key, chosen_value)
                        doublespinBox.signal_paramChanged[str, float].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: doublespinBox})
                        layout.addRow(key, doublespinBox)

                    # For parameters with multiple non-fixed values in a list (user can modify them)
                    elif values is list:
                        modifiable_list = Connect_Modifiable_List(key, chosen_value)
                        modifiable_list.signal_paramChanged[str, list].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: modifiable_list})
                        layout.addRow(key, modifiable_list)
                        continue

                    #  For parameters containing a string to modify
                    elif values is str:
                        lineEdit = Connect_LineEdit(key, chosen_value)
                        lineEdit.signal_paramChanged[str, str].connect(self.on_guichanges)
                        lineEdit.signal_paramChanged[str, type(None)].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: lineEdit})
                        layout.addRow(key, lineEdit)

                    # For parameters with multiple fixed values.
                    elif type(values) is tuple:
                        comboParams = Connect_ComboBox(key, chosen_value, values)
                        comboParams.signal_paramChanged[str, object].connect(self.on_guichanges)
                        comboParams.signal_additionalParamChanged[str, dict].connect(self.on_guichanges)
                        self.paramsWidgets.update({key: comboParams})
                        layout.addRow(key, comboParams.layout)
                        continue

                    # For parameters with multiple non-fixed values in a dict (user can modify them)
                    elif type(values) is dict:
                        try:
                            selection = chosen_value['selected']
                            comboParams = Connect_ComboBox(key, chosen_value, values)
                            comboParams.signal_paramChanged[str, object].connect(self.on_guichanges)
                            comboParams.signal_additionalParamChanged[str, dict].connect(self.on_guichanges)
                            self.paramsWidgets.update({key: comboParams})
                            layout.addRow(key, comboParams.layout)
                        except:
                            modifiable_dict = Connect_Modifiable_Dict(key, chosen_value, values)
                            modifiable_dict.signal_paramChanged[str, dict].connect(self.on_guichanges)
                            self.paramsWidgets.update({key: modifiable_dict})
                            layout.addRow(key, modifiable_dict)

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

        
        
        # Connect inter-widgets signals and slots
        if self.modality == 'trainer':
            self.paramsWidgets['TRIGGER_FILE'].signal_pathChanged[str, str].connect(trigger_def.on_new_tdef_file)
            self.paramsWidgets['TRIGGER_FILE'].on_selected()
        
        if self.modality == 'online':
            self.paramsWidgets['TRIGGER_FILE'].signal_pathChanged[str, str].connect(directions.on_new_tdef_file)
            self.paramsWidgets['TRIGGER_FILE'].on_selected()
        
            self.paramsWidgets['DECODER_FILE'].signal_pathChanged[str, str].connect(directions.on_new_decoder_file)
            self.paramsWidgets['DECODER_FILE'].on_selected()
            
    # ----------------------------------------------------------------------
    def load_config(self, cfg_file):
        """
        Dynamic loading of a config file.
        Format the lib to fit the previous developed neurodecode code if subject specific file (not for the templates).
        cfg_file: tuple containing the path and the config file name.
        """
        if self.cfg_subject == None or cfg_file[1] not in self.cfg_subject.__file__:
            # Dynamic loading
            sys.path.append(cfg_file[0])
            cfg_module = import_module(cfg_file[1].split('.')[0])
        else:
            cfg_module = reload(self.cfg_subject)

        return cfg_module

    #----------------------------------------------------------------------
    def load_all_params(self, cfg_template, cfg_file):
        """
        Loads the params structure and assign the subject/s specific value.
        It also checks the sanity of the loaded params according to the protocol.
        """
        try:
            # Loads the subject's specific values
            self.cfg_subject = self.load_config(cfg_file)

            # Loads the template
            if self.cfg_struct == None or cfg_template[1] not in self.cfg_struct.__file__:
                self.cfg_struct = self.load_config(cfg_template)

            # Display parameters on the GUI
            self.disp_params(self.cfg_struct, self.cfg_subject)
            
            # Check the parameters integrity
            self.m.check_config(self.cfg_subject)
            
        except Exception as e:
            self.signal_error[str].emit(str(e))

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

        print("The parameter %s has been changed to %s" % (name, getattr(self.cfg_subject, name)))
        

    # ----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_pathSearch(self):
        """
        Opens the File dialog window when the search button is pressed.
        """
        
        buttonIcon = self.ui.pushButton_Search.text()
        
        if buttonIcon == 'Search':            
            path_name = QFileDialog.getExistingDirectory(caption="Choose the subject's directory", directory=os.environ['NEUROD_SCRIPTS'])
            
            if path_name:
                self.ui.lineEdit_pathSearch.clear()
                self.ui.lineEdit_pathSearch.insert(path_name)
                self.ui.pushButton_Search.setText('Accept')
                self.ui.pushButton_Search.setStyleSheet("color: red;")            
        else:
            self.ui.pushButton_Search.setText('Search')
            self.ui.pushButton_Search.setStyleSheet("color: black;")
            self.on_enable_modality()

    # ----------------------------------------------------------------------
    def look_for_subject_file(self, modality):
        '''
        Look if the subject config file is contained in the subject folder
        
        modality = offline, trainer or online
        '''
        is_found = False
        cfg_file = None
        cfg_path = Path(self.ui.lineEdit_pathSearch.text())
        
        for f in glob(os.fspath(cfg_path / "*.py") , recursive=False):
            fileName =  os.path.split(f)[-1]
            if modality in fileName and 'structure' not in fileName:
                is_found = True
                cfg_file = f
                break
        return is_found, cfg_file    

    #----------------------------------------------------------------------
    def find_structure_file(self, cfg_file, modality):
        """
        Find the structure config file associated with the subject config file
        
        cfg_file: subject specific config file
        modality = offline, trainer or online
        """
        # Find the config template
        tmp = cfg_file.split('.')[0]  # Remove the .py
        self.protocol = tmp.split('-')[-1]    # Extract the protocol name
        template_path = Path(os.environ['NEUROD_ROOT']) / 'neurodecode' / 'config_files' / self.protocol / 'structure_files'
        
        for f in glob(os.fspath(template_path / "*.py") , recursive=False):
            fileName =  os.path.split(f)[-1]
            if modality in fileName and 'structure' in fileName:
                return f            
    
    #----------------------------------------------------------------------
    def prepare_config_files(self, modality):
        """
        Find both the subject config file and the associated structure config
        file paths
        """
        is_found, cfg_file = self.look_for_subject_file(modality)
            
        if is_found is False:
            self.error_dialog.showMessage('Config file missing: copy an ' + modality + ' config file to the subject folder or create a new subjet')
            return None, None
        else:
            cfg_template = self.find_structure_file(cfg_file, modality)
            cfg_file = os.path.split(cfg_file)
            cfg_template = os.path.split(cfg_template)
            
            return cfg_file, cfg_template
    
    # ----------------------------------------------------------------------
    def load_protocol_module(self, module_name):
        """
        Load or reload the protocol's module associated with the modality
        
        module_name = name of the module to load
        """
        if module_name not in sys.modules:
            path2protocol =  os.path.split(self.ui.lineEdit_pathSearch.text())[0]
            sys.path.append(path2protocol)
            self.m = import_module(module_name)
        else:
            self.m = reload(sys.modules[module_name])
        
    # ----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_offline(self):
        """
        Loads the Offline parameters.
        """
        self.modality = 'offline'
                
        cfg_file, cfg_template = self.prepare_config_files(self.modality)
        module_name = 'offline_' + self.protocol
        
        self.load_protocol_module(module_name)
        
        self.ui.checkBox_Record.setChecked(True)
        self.ui.checkBox_Record.setEnabled(False)
        
        if cfg_file and cfg_template:
            self.load_all_params(cfg_template, cfg_file)       
           
        self.ui.groupBox_Launch.setEnabled(True)
        
    # ----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_train(self):
        """
        Loads the Training parameters.
        """
        self.modality = 'trainer'
                
        cfg_file, cfg_template = self.prepare_config_files(self.modality)
        module_name = 'trainer_' + self.protocol
        
        self.load_protocol_module(module_name)   
        
        self.ui.checkBox_Record.setChecked(False)
        self.ui.checkBox_Record.setEnabled(False)
        
        if cfg_file and cfg_template:
            self.load_all_params(cfg_template, cfg_file)
            
        self.ui.groupBox_Launch.setEnabled(True)

    #----------------------------------------------------------------------
    @pyqtSlot()
    def on_click_online(self):
        """
        Loads the Online parameters.
        """
        self.modality = 'online'        
                
        cfg_file, cfg_template = self.prepare_config_files(self.modality)
        module_name = 'online_' + self.protocol 
        
        self.load_protocol_module(module_name)      

        self.ui.checkBox_Record.setChecked(True)
        self.ui.checkBox_Record.setEnabled(True)
        
        if cfg_file and cfg_template:
            self.load_all_params(cfg_template, cfg_file)

        self.ui.groupBox_Launch.setEnabled(True)

        
    #----------------------------------------------------------------------v
    @pyqtSlot()
    def on_click_start(self):
        """
        Launch the selected protocol. It can be Offline, Train or Online.
        """
        self.record_dir = Path(self.cfg_subject.DATA_PATH)
        
        ccfg = cfg_class(self.cfg_subject)  #  because a module is not pickable
        
        with self.record_state.get_lock():
            self.record_state.value = 0

        if not self.protocol_state.value:            
            self.ui.textEdit_terminal.clear()
            
            # Recording shared variable + recording terminal            
            if self.ui.checkBox_Record.isChecked():
                
                amp = self.ui.comboBox_LSL.currentData()
                if not amp:   
                    self.signal_error[str].emit('No LSL amplifier specified.')
                    return                
                
                if not self.record_terminal:                
                    self.record_terminal = GuiTerminal(self.recordLogger, 'INFO', self.width())
                    self.hide_recordTerminal[bool].connect(self.record_terminal.setHidden)
                
                else:
                    self.record_terminal.textEdit.clear()
                    self.record_terminal.textEdit.insertPlainText('Waiting for the recording to start...\n')
                    self.hide_recordTerminal[bool].emit(False)
                
                
                # Protocol shared variable
                with self.protocol_state.get_lock():
                    self.protocol_state.value = 2  #  0=stop, 1=start, 2=wait
                    
                processesToLaunch = [('recording', recorder.run_gui, [self.record_state, self.protocol_state, self.record_dir, self.recordLogger, amp['name'], amp['serial'], False, self.record_terminal.my_receiver.queue]), \
                                     ('protocol', self.m.run, [ccfg, self.protocol_state, self.my_receiver.queue])]

            else:    
                # Protocol shared variable
                with self.protocol_state.get_lock():
                    self.protocol_state.value = 1  #  0=stop, 1=start, 2=wait
                
                processesToLaunch = [('protocol', self.m.run, [ccfg, self.protocol_state, self.my_receiver.queue])]
                      
            launchedProcess = Thread(target=self.launching_subprocesses, args=processesToLaunch)
            launchedProcess.start()
            logger.info(self.modality + ' protocol starting...')
            self.ui.pushButton_Start.setText('Stop')
            
        else:    
            with self.protocol_state.get_lock():
                self.protocol_state.value = 0
            time.sleep(2)
            self.hide_recordTerminal[bool].emit(True)
            self.ui.pushButton_Start.setText('Start')


    #----------------------------------------------------------------------
    @pyqtSlot(str)
    def on_terminal_append(self, text):
        """
        Writes to the QtextEdit_terminal the redirected stdout.
        """
        self.ui.textEdit_terminal.moveCursor(QTextCursor.End)
        self.ui.textEdit_terminal.insertPlainText(text)
    
    @pyqtSlot()
    #----------------------------------------------------------------------
    def on_click_newSubject(self):
        """
        Instance a Connect_NewSubject QDialog class
        """
        buttonIcon = self.ui.pushButton_NewSubject.text()
        
        if buttonIcon == 'New':
            qdialog = Connect_NewSubject(self, self.ui.lineEdit_pathSearch)
            qdialog.signal_error[str].connect(self.on_error)
            self.ui.pushButton_NewSubject.setText('Accept')
            self.ui.pushButton_NewSubject.setStyleSheet("color: red;")
        else:
            self.ui.pushButton_NewSubject.setText('New')
            self.ui.pushButton_NewSubject.setStyleSheet("color: black;")
            self.on_enable_modality()


    #----------------------------------------------------------------------
    def on_error(self, errorMsg):
        """
        Display the error message into a QErrorMessage
        """
        self.error_dialog.showMessage(errorMsg)
        
    #----------------------------------------------------------------------
    def on_click_save_params_to_file(self):
        """
        Save the params to a config_file
        """
        filePath, fileName = os.path.split(self.cfg_subject.__file__)
        fileName = fileName.split('.')[0]       # Remove the .py
        
        file = self.cfg_subject.__file__.split('.')[0] + '_' + datetime.now().strftime('%m.%d.%d.%M') + '.py'
        filePath = QFileDialog.getSaveFileName(self, 'Save config file', file, 'python(*.py)')
        if filePath[0]:
            save_params_to_file(filePath[0], cfg_class(self.cfg_subject))
        
    @pyqtSlot(list)
    #----------------------------------------------------------------------
    def fill_comboBox_lsl(self, amp_list):
        """
        Fill the comboBox with the available lsl streams
        """
        # Clear the comboBox_lsl first
        self.ui.comboBox_LSL.clear()
        
        for amp in amp_list:
            amp_formated = '{} ({})'.format(amp[1], amp[2])
            self.ui.comboBox_LSL.addItem(amp_formated, {'name':amp[1], 'serial':amp[2]})
        self.ui.pushButton_LSL.setText('Search')
        self.ui.pushButton_Viewer.setEnabled(True)
    
    #----------------------------------------------------------------------
    def on_click_lsl_button(self):
        """
        Find the available lsl streams and display them in the comboBox_LSL
        """
        if self.lsl_state.value == 1:
                      
            with self.lsl_state.get_lock():
                self.lsl_state.value = 0
            
            self.lsl_thread.terminate()
            self.lsl_thread.wait()
            self.ui.pushButton_LSL.setText('Search')
        
        else:      
            self.ui.textEdit_terminal.clear()
            
            with self.lsl_state.get_lock():
                self.lsl_state.value = 1
            
            self.lsl_thread = search_lsl_streams_thread(self.lsl_state, logger)
            self.lsl_thread.signal_lsl_found[list].connect(self.fill_comboBox_lsl)
            self.lsl_thread.start()
            
            self.ui.pushButton_LSL.setText('Stop')
        
        
    #----------------------------------------------------------------------
    def on_click_start_viewer(self):
        """
        Launch the viewer to check the signals in a seperate process 
        """
        # Start Viewer
        if not self.viewer_state.value:
            self.ui.textEdit_terminal.clear()
            
            with self.viewer_state.get_lock():
                self.viewer_state.value = 1
            
            amp = self.ui.comboBox_LSL.currentData()
            viewerprocess = mp.Process(target=instantiate_scope, args=[amp, self.viewer_state, logger, self.my_receiver.queue])
            viewerprocess.start()
            
            self.ui.pushButton_Viewer.setText('Stop')
            
        # Stop Viewer
        else:    
            with self.viewer_state.get_lock():
                self.viewer_state.value = 0            
            
            self.ui.pushButton_Viewer.setEnabled(True)
            self.ui.pushButton_Viewer.setText('Viewer')
            
    @pyqtSlot()
    #----------------------------------------------------------------------
    def on_enable_modality(self):
        """
        Enable the modalities groupBox if the provided path exists
        """
        subjectFolder = self.ui.lineEdit_pathSearch.text()

        if subjectFolder:
            exist = os.path.isdir(subjectFolder)
            
            if not exist:
                self.signal_error[str].emit('The provided subject folder does not exists.')
            else:
                self.ui.groupBox_Modality.setEnabled(True)
            
            
    #----------------------------------------------------------------------
    def connect_signals_to_slots(self):
        """Connects the signals to the slots"""
        
        # New subject button
        self.ui.pushButton_NewSubject.clicked.connect(self.on_click_newSubject)
        
        # Search Subject folder Search button
        self.ui.pushButton_Search.clicked.connect(self.on_click_pathSearch)
        
        # Enable modality when subject folder path is given
        self.ui.lineEdit_pathSearch.editingFinished.connect(self.on_enable_modality)
        
        # Offline button
        self.ui.pushButton_Offline.clicked.connect(self.on_click_offline)
        
        # Train button
        self.ui.pushButton_Train.clicked.connect(self.on_click_train)
        
        # Online button
        self.ui.pushButton_Online.clicked.connect(self.on_click_online)
        
        # Start button
        self.ui.pushButton_Start.clicked.connect(self.on_click_start)
        
        # Save conf file
        self.ui.actionSave_config_file.triggered.connect(self.on_click_save_params_to_file)
        
        # Error dialog
        self.signal_error[str].connect(self.on_error)
        
        # Start viewer button
        self.ui.pushButton_Viewer.clicked.connect(self.on_click_start_viewer)
        
        # LSL button
        self.ui.pushButton_LSL.clicked.connect(self.on_click_lsl_button)

    #----------------------------------------------------------------------
    def launching_subprocesses(*args):
        """
        Launch subprocesses
        
        processesToLaunch = list of tuple containing the functions to launch
        and their args
        """
        launchedProcesses = dict()
        
        for p in args[1:]:
            launchedProcesses[p[0]] = mp.Process(target=p[1], args=p[2])
            launchedProcesses[p[0]].start()
        
        # Wait that the protocol is finished to stop recording
        launchedProcesses['protocol'].join()
        
        recordState = args[1][2][1]     #  Sharing variable
        try:        
            with recordState.get_lock():
                recordState.value = 0
        except:
            pass

#----------------------------------------------------------------------
def instantiate_scope(amp, state, logger=logger, queue=None):
    logger.info('Connecting to a %s (Serial %s).' % (amp['name'], amp['serial']))
    app = QApplication(sys.argv)
    ex = viewer.Scope(amp['name'], amp['serial'], state, queue)
    sys.exit(app.exec_())
    
#----------------------------------------------------------------------    
def main():
    #unittest.main()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
