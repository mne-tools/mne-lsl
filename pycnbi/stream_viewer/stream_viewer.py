# -*- coding: utf-8 -*-
from __future__ import print_function, division, unicode_literals

"""
 EEG Scope
 Iñaki Iturrate, Kyuhwa Lee
 2017

 V1.0
 TODO
	- Should move to VisPY: http://vispy.org/plot.html#module-vispy.plot but still under development
	- The scope should be a class itself
	- Scope of EXG signals
	- Events should stored in a class.

"""

DEBUG_TRIGGER = False # TODO: parameterize
NUM_X_CHANNELS = 16 # TODO: parameterize


import os
import sys
import pdb
import time
import math
import struct
import traceback
import subprocess
import numpy as np
import pyqtgraph as pg
import pycnbi
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from scipy.signal import butter, lfilter, lfiltic, buttord
from pycnbi.stream_receiver.stream_receiver import StreamReceiver
from pycnbi import logger
from pyqtgraph.Qt import QtCore, QtGui, uic
from configparser import RawConfigParser
from builtins import input

# Load GUI. Designed with QT Creator, feel free to change stuff
form_class = uic.loadUiType("./mainwindow.ui")[0]


class Scope(QtGui.QMainWindow, form_class):
    def __init__(self, amp_name, amp_serial):
        super(Scope, self).__init__()
        self.amp_name = amp_name
        self.amp_serial = amp_serial
        self.init_scope()

    #
    # 	Main init function
    #
    def init_scope(self):

        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')

        self.init_config_file()
        self.init_loop()
        self.init_panel_GUI()
        self.init_scope_GUI()
        self.init_timer()

    #
    #	Initializes config file
    #
    def init_config_file(self):
        self.scope_settings = RawConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
        if (len(sys.argv) == 1):
            self.show_channel_names = 0
            self.device_name = ""
        else:
            if (sys.argv[1].find("gtec") > -1):
                self.device_name = "gtec"
                self.show_channel_names = 1
            elif (sys.argv[1].find("biosemi") > -1):
                self.device_name = "biosemi"
                self.show_channel_names = 1
            elif (sys.argv[1].find("hiamp") > -1):
                self.device_name = "hiamp"
                self.show_channel_names = 1
            else:
                self.device_name = ""
                self.show_channel_names = 0
        # self.scope_settings.read(os.getenv("HOME") + "/.scope_settings.ini")
        self.scope_settings.read(".scope_settings.ini")

    #
    # 	Initialize control panel parameter
    #
    def init_panel_GUI(self):
        self.setupUi(self)

        self.show_TID_events = False
        self.show_LPT_events = False
        self.show_Key_events = False

        # Event handler
        self.comboBox_scale.activated.connect(self.onActivated_combobox_scale)
        self.spinBox_time.valueChanged.connect(self.onValueChanged_spinbox_time)
        self.checkBox_car.stateChanged.connect(self.onActivated_checkbox_car)
        self.checkBox_bandpass.stateChanged.connect(
            self.onActivated_checkbox_bandpass)
        self.checkBox_showTID.stateChanged.connect(
            self.onActivated_checkbox_TID)
        self.checkBox_showLPT.stateChanged.connect(
            self.onActivated_checkbox_LPT)
        self.checkBox_showKey.stateChanged.connect(
            self.onActivated_checkbox_Key)
        self.pushButton_bp.clicked.connect(self.onClicked_button_bp)
        self.pushButton_rec.clicked.connect(self.onClicked_button_rec)
        self.pushButton_stoprec.clicked.connect(self.onClicked_button_stoprec)

        self.pushButton_stoprec.setEnabled(False)
        self.comboBox_scale.setCurrentIndex(4)
        self.checkBox_car.setChecked(
            int(self.scope_settings.get("filtering", "apply_car_filter")))
        self.checkBox_bandpass.setChecked(
            int(self.scope_settings.get("filtering", "apply_bandpass_filter")))
        self.checkBox_showTID.setChecked(
            int(self.scope_settings.get("plot", "show_TID_events")))
        self.checkBox_showLPT.setChecked(
            int(self.scope_settings.get("plot", "show_LPT_events")))
        self.checkBox_showKey.setChecked(
            int(self.scope_settings.get("plot", "show_KEY_events")))
        self.statusBar.showMessage("[Not recording]")

        self.channels_to_show_idx = []
        idx = 0
        for y in range(0, 4):
            for x in range(0, NUM_X_CHANNELS):
                if (idx < self.config['eeg_channels']):
                    # self.table_channels.item(x,y).setTextAlignment(QtCore.Qt.AlignCenter)
                    QtGui.QTableWidgetItem.setSelected(self.table_channels.item(x, y), True) # Qt5
                    #self.table_channels.setItemSelected(self.table_channels.item(x, y), True) # Qt4 only
                    self.channels_to_show_idx.append(idx)
                else:
                    self.table_channels.setItem(x, y,
                        QtGui.QTableWidgetItem("N/A"))
                    self.table_channels.item(x, y).setFlags(
                        QtCore.Qt.NoItemFlags)
                    self.table_channels.item(x, y).setTextAlignment(
                        QtCore.Qt.AlignCenter)
                idx += 1

        self.table_channels.verticalHeader().setStretchLastSection(True)
        self.table_channels.horizontalHeader().setStretchLastSection(True)
        self.table_channels.itemSelectionChanged.connect(
            self.onSelectionChanged_table)

        self.screen_width = 522
        self.screen_height = 160
        # self.setGeometry(100,100, self.screen_width, self.screen_height)
        # self.setFixedSize(self.screen_width, self.screen_height)
        self.setWindowTitle('EEG Scope Panel')
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()
        self.show()

    #
    #	Initialize scope parameters
    #
    def init_scope_GUI(self):

        self.bool_parser = {True:'1', False:'0'}

        # PyQTGraph plot initialization
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('EEG Scope')
        self.win.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.win.keyPressEvent = self.keyPressEvent
        self.win.show()
        self.main_plot_handler = self.win.addPlot()
        self.win.resize(1280, 800)

        # Scales available in the GUI. If you change the options in the GUI
        # you should change them here as well
        self.scales_range = [1, 10, 25, 50, 100, 250, 500, 1000, 2500, 100000]

        # Scale in uV
        self.scale = int(self.scope_settings.get("plot", "scale_plot"))
        # Time window to show in seconds
        self.seconds_to_show = int(self.scope_settings.get("plot", "time_plot"))

        # Y Tick labels. Use values from the config file.
        self.channel_labels = []
        values = []

        ''' For non-LSL systems having no channel names
        for x in range(0, self.config['eeg_channels']):
            if (self.show_channel_names):
                self.channel_labels.append("(" + str(x + 1) + ") " +
                    self.scope_settings.get("internal",
                    "channel_names_" + self.device_name + str(
                    self.config['eeg_channels'])).split(', ')[x])
            else:
                self.channel_labels.append('CH ' + str(x + 1))
        '''
        ch_names = np.array( self.sr.get_channel_names() )
        self.channel_labels = ch_names[ self.sr.get_eeg_channels() ]
        for x in range(0, len(self.channels_to_show_idx)):
            values.append((-x * self.scale,
                self.channel_labels[self.channels_to_show_idx[x]]))

        values_axis = []
        values_axis.append(values)
        values_axis.append([])

        # Update table labels with current names
        idx = 0
        for y in range(0, 4):
            for x in range(0, NUM_X_CHANNELS):
                if (idx < self.config['eeg_channels']):
                    self.table_channels.item(x, y).setText(
                        self.channel_labels[idx])
                idx += 1

        # Plot initialization
        self.main_plot_handler.getAxis('left').setTicks(values_axis)
        self.main_plot_handler.setRange(xRange=[0, self.seconds_to_show],
            yRange=[+1.5 * self.scale,
                -0.5 * self.scale - self.scale * self.config['eeg_channels']])
        self.main_plot_handler.disableAutoRange()
        self.main_plot_handler.showGrid(y=True)
        self.main_plot_handler.setLabel(axis='left',
            text='Scale (uV): ' + str(self.scale))
        self.main_plot_handler.setLabel(axis='bottom', text='Time (s)')

        # X axis
        self.x_ticks = np.zeros(self.config['sf'] * self.seconds_to_show);
        for x in range(0, self.config['sf'] * self.seconds_to_show):
            self.x_ticks[x] = (x * 1) / float(self.config['sf'])

        # Plotting colors. If channels > 16, colors will roll back to the beginning
        self.colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [0, 255, 255], [255, 0, 255], [128, 100, 100], [0, 128, 0],
            [0, 128, 128], [128, 128, 0], [255, 128, 128], [128, 0, 128],
            [128, 255, 0], [255, 128, 0], [0, 255, 128], [128, 0, 255]])

        # We want a lightweight scope, so we downsample the plotting to 64 Hz
        self.subsampling_value = self.config['sf'] / 64

        # EEG data for plotting
        self.data_plot = np.zeros((self.config['sf'] * self.seconds_to_show,
        self.config['eeg_channels']))
        self.curve_eeg = []
        for x in range(0, len(self.channels_to_show_idx)):
            self.curve_eeg.append(self.main_plot_handler.plot(x=self.x_ticks,
                y=self.data_plot[:, self.channels_to_show_idx[x]],
                pen=pg.mkColor(
                    self.colors[self.channels_to_show_idx[x] % 16, :])))
        # self.curve_eeg[-1].setDownsampling(ds=self.subsampling_value, auto=False, method="mean")


        # Events data
        self.events_detected = []
        self.events_curves = []
        self.events_text = []

        # CAR initialization
        self.apply_car = int(
            self.scope_settings.get("filtering", "apply_car_filter"))
        self.matrix_car = np.zeros(
            (self.config['eeg_channels'], self.config['eeg_channels']),
            dtype=float)
        self.matrix_car[:, :] = -1 / float(self.config['eeg_channels'])
        np.fill_diagonal(self.matrix_car,
            1 - (1 / float(self.config['eeg_channels'])))

        # Laplacian initalization. TO BE DONE
        self.matrix_lap = np.zeros(
            (self.config['eeg_channels'], self.config['eeg_channels']),
            dtype=float)
        np.fill_diagonal(self.matrix_lap, 1)
        self.matrix_lap[2, 0] = -1
        self.matrix_lap[0, 2] = -0.25
        self.matrix_lap[0, 2] = -0.25

        # BP initialization
        self.apply_bandpass = int(
            self.scope_settings.get("filtering", "apply_bandpass_filter"))
        if (self.apply_bandpass):
            self.doubleSpinBox_hp.setValue(float(
                self.scope_settings.get("filtering",
                    "bandpass_cutoff_frequency").split(' ')[0]))
            self.doubleSpinBox_lp.setValue(float(
                self.scope_settings.get("filtering",
                    "bandpass_cutoff_frequency").split(' ')[1]))
            self.pushButton_bp.click()

        self.checkBox_bandpass.setChecked(self.apply_car)
        self.checkBox_bandpass.setChecked(self.apply_bandpass)

        self.update_title_scope()

        # Help variables
        self.show_help = 0
        self.help = pg.TextItem(
            "CNBI EEG Scope v0.3 \n" + "----------------------------------------------------------------------------------\n" + "C: De/activate CAR Filter\n" + "B: De/activate Bandpass Filter (with current settings)\n" + "T: Show/hide TiD events\n" + "L: Show/hide LPT events\n" + "K: Show/hide Key events. If not shown, they are NOT recorded!\n" + "0-9: Add a user-specific Key event. Do not forget to write down why you marked it.\n" + "Up, down arrow keys: Increase/decrease the scale, steps of 10 uV\n" + "Left, right arrow keys: Increase/decrease the time to show, steps of 1 s\n" + "Spacebar: Stop the scope plotting, whereas data acquisition keeps running (EXPERIMENTAL)\n" + "Esc: Exits the scope",
            anchor=(0, 0), border=(70, 70, 70),
            fill=pg.mkColor(20, 20, 20, 200), color=(255, 255, 255))

        # Stop plot functionality
        self.stop_plot = 0

        # Force repaint even when we shouldn't repaint.
        self.force_repaint = 0

    # For some strange reason when the width is > 1 px the scope runs slow.
    # self.pen_plot = []
    # for x in range(0, self.config['eeg_channels']):
    # 	self.pen_plot.append(pg.mkPen(self.colors[x%16,:], width=3))

    #
    # 	Initializes the BCI loop parameters
    #
    def init_loop_cnbiloop(self):

        self.fin = open(self.scope_settings.get("internal", "path_pipe"), 'r')

        # 12 unsigned ints (4 bytes)
        data = struct.unpack("<12I", self.fin.read(4 * 12))

        self.config = {'id':data[0], 'sf':data[1], 'labels':data[2],
            'samples':data[3], 'eeg_channels':data[4], 'exg_channels':data[5],
            'tri_channels':data[6], 'eeg_type':data[8], 'exg_type':data[9],
            'tri_type':data[10], 'lbl_type':data[11], 'tim_size':1,
            'idx_size':1}

        self.tri = np.zeros(self.config['samples'])
        self.eeg = np.zeros(
            (self.config['samples'], self.config['eeg_channels']),
            dtype=np.float)
        self.exg = np.zeros(
            (self.config['samples'], self.config['exg_channels']),
            dtype=np.float)

        # TID initialization
        self.bci = BCI.BciInterface()

    #
    # 	Initializes the BCI loop parameters
    #
    def init_loop(self):

        self.updating = False

        self.sr = StreamReceiver(window_size=1, buffer_size=10,
            amp_serial=self.amp_serial, amp_name=self.amp_name)
        srate = int(self.sr.sample_rate)
        # n_channels= self.sr.channels

        # 12 unsigned ints (4 bytes)
        ########## TODO: assumkng 32 samples chunk => make it read from LSL header
        data = ['EEG', srate, ['L', 'R'], 32, len(self.sr.get_eeg_channels()),
            0, self.sr.get_trigger_channel(), None, None, None, None, None]
        logger.info('Trigger channel is %d' % self.sr.get_trigger_channel())

        self.config = {'id':data[0], 'sf':data[1], 'labels':data[2],
            'samples':data[3], 'eeg_channels':data[4], 'exg_channels':data[5],
            'tri_channels':data[6], 'eeg_type':data[8], 'exg_type':data[9],
            'tri_type':data[10], 'lbl_type':data[11], 'tim_size':1,
            'idx_size':1}

        self.tri = np.zeros(self.config['samples'])
        self.last_tri = 0
        self.eeg = np.zeros(
            (self.config['samples'], self.config['eeg_channels']),
            dtype=np.float)
        self.exg = np.zeros(
            (self.config['samples'], self.config['exg_channels']),
            dtype=np.float)
        self.ts_list = []
        self.ts_list_tri = []

    #
    # 	Initializes the QT timer, which will call the update function every 20 ms
    #
    def init_timer(self):

        self.tm = qc.Timer()  # leeq

        QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.flush()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(20);

    # QtCore.QTimer.singleShot( 20, self.update_loop )


    #
    #	Main update function (connected to the timer)
    #
    def update_loop(self):
        try:
            # assert self.updating==False, 'thread destroyed?'
            # self.updating= True

            # self.handle_tobiid_input()	# Read TiDs
            self.read_eeg()  # Read new chunk
            if len(self.ts_list) > 0:
                self.filter_signal()  # Filter acquired data
                self.update_ringbuffers()  # Update the plotting infor
                if (not self.stop_plot):
                    self.repaint()  # Call paint event
        except:
            logger.exception('Exception. Dropping into a shell.')
            pdb.set_trace()
        finally:
            # self.updating= False
            # using singleShot instead
            # QtCore.QTimer.singleShot( 20, self.update_loop )
            pass

    #
    #	Read EEG
    #
    def read_eeg(self):

        # if self.updating==True: print( '##### ERROR: thread destroyed ? ######' )
        # self.updating= True

        try:
            # data, self.ts_list= self.sr.inlets[0].pull_chunk(max_samples=self.config['sf']) # [frames][channels]
            data, self.ts_list = self.sr.acquire(blocking=False)
            
            # TODO: check and change to these two lines
            #self.sr.acquire(blocking=False, decim=DECIM)
            #data, self.ts_list = self.sr.get_window()

            if len(self.ts_list) == 0:
                # self.eeg= None
                # self.tri= None
                return

            n = self.config['eeg_channels']
            '''
            x= np.array( data )
            trg_ch= self.config['tri_channels']
            if trg_ch is not None:
                self.tri= np.reshape( x[:,trg_ch], (-1,1) ) # samples x 1
            self.eeg= np.reshape( x[:,self.sr.eeg_channels], (-1,n) ) # samples x channels
            '''
            trg_ch = self.config['tri_channels']
            if trg_ch is not None:
                self.tri = np.reshape(data[:, trg_ch], (-1, 1))  # samples x 1
            self.eeg = np.reshape(data[:, self.sr.eeg_channels],
                (-1, n))  # samples x channels

            if DEBUG_TRIGGER:
                # show trigger value
                try:
                    trg_value = max(self.tri)
                    if trg_value > 0:
                        logger.info('Received trigger %s' % trg_value)
                except:
                    logger.exception('Error! self.tri = %s' % self.tri)

                    # Read exg. self.config.samples*self.config.exg_ch, type float
                    # bexg = np.random.rand( 1, self.config['samples'] * self.config['exg_channels'] )
                    # self.exg = np.reshape(list(bexg), (self.config['samples'],self.config['exg_channels']))
        except WindowsError:
            # print('**** Access violation in read_eeg():\n%s\n%s'% (sys.exc_info()[0], sys.exc_info()[1]))
            pass
        except:
            logger.exception()
            pdb.set_trace()

    #
    #	Read EEG
    #
    def read_eeg_cnbiloop(self):

        # Reading in python is blocking, so it will wait until having the amount of data needed
        # Read timestamp. 1 value, type double
        timestamp = struct.unpack("<d", self.fin.read(8 * 1))
        # Read index. 1 value, type uint64
        index = struct.unpack("<Q", self.fin.read(8 * 1))
        # Read labels. self.config.labels, type double
        labels = struct.unpack("<" + str(self.config['labels']) + "I",
            self.fin.read(4 * self.config['labels']))
        # Read eeg. self.config.samples*self.config.eeg_ch, type float
        beeg = struct.unpack("<" + str(
            self.config['samples'] * self.config['eeg_channels']) + "f",
            self.fin.read(
                4 * self.config['samples'] * self.config['eeg_channels']))
        self.eeg = np.reshape(list(beeg),
            (self.config['samples'], self.config['eeg_channels']))
        # Read exg. self.config.samples*self.config.exg_ch, type float
        bexg = struct.unpack("<" + str(
            self.config['samples'] * self.config['exg_channels']) + "f",
            self.fin.read(
                4 * self.config['samples'] * self.config['exg_channels']))
        self.exg = np.reshape(list(bexg),
            (self.config['samples'], self.config['exg_channels']))
        # Read tri. self.config.samples*self.config.tri_ch, type float
        self.tri = struct.unpack("<" + str(
            self.config['samples'] * self.config['tri_channels']) + "i",
            self.fin.read(
                4 * self.config['samples'] * self.config['tri_channels']))

    #
    #	Bandpas + CAR filtering
    #
    def filter_signal(self):

        if (self.apply_bandpass):
            for x in range(0, self.eeg.shape[1]):
                self.eeg[:, x], self.zi[:, x] = lfilter(self.b, self.a,
                    self.eeg[:, x], -1, self.zi[:, x])

        # We only apply CAR if selected AND there are at least 2 channels. Otherwise it makes no sense
        if (self.apply_car) and (len(self.channels_to_show_idx) > 1):
            self.eeg = np.dot(self.matrix_car, np.transpose(self.eeg))
            self.eeg = np.transpose(self.eeg)

    #
    #	Update ringbuffers and events for plotting
    #
    def update_ringbuffers(self):
        # leeq
        self.data_plot = np.roll(self.data_plot, -len(self.ts_list), 0)
        self.data_plot[-len(self.ts_list):, :] = self.eeg

        # We have to remove those indexes that reached time = 0
        delete_indices_e = []
        delete_indices_c = []
        for x in range(0, len(self.events_detected), 2):
            xh = int(x / 2)
            self.events_detected[x] -= len(self.ts_list)  # leeq
            if (self.events_detected[x] < 0) and (not self.stop_plot):
                delete_indices_e.append(x)
                delete_indices_e.append(x + 1)
                delete_indices_c.append(xh)
                self.events_curves[xh].clear()
                self.main_plot_handler.removeItem(self.events_text[xh])

        self.events_detected = [i for j, i in enumerate(self.events_detected) if
            j not in delete_indices_e]
        self.events_curves = [i for j, i in enumerate(self.events_curves) if
            j not in delete_indices_c]
        self.events_text = [i for j, i in enumerate(self.events_text) if
            j not in delete_indices_c]

        # Find LPT events and add them
        if (self.show_LPT_events) and (not self.stop_plot):
            for x in range(len(self.tri)):
                tri = int(self.tri[x])
                if tri != 0 and (tri > self.last_tri):
                    self.addEventPlot("LPT", tri)
                    logger.info('Trigger %d received' % tri)
                self.last_tri = tri

    #
    #	Called by repaint()
    #
    def paintEvent(self, e):
        # Distinguish between paint events from timer and event QT widget resizing, clicking etc (sender is None)
        # We should only paint when the timer triggered the event.
        # Just in case, there's a flag to force a repaint even when we shouldn't repaint
        sender = self.sender()
        if 'force_repaint' not in self.__dict__.keys():
            logger.warning('force_repaint is not set! Is it a Qt bug?')
            self.force_repaint = 0
        if (sender is None) and (not self.force_repaint):
            pass
        else:
            self.force_repaint = 0
            qp = QtGui.QPainter()
            qp.begin(self)
            # Update the interface
            self.paintInterface(qp)
            qp.end()

    #
    #	Update stuff on the interface. Only graphical updates should be added here
    #
    def paintInterface(self, qp):

        # Update EEG channels
        for x in range(0, len(self.channels_to_show_idx)):
            self.curve_eeg[x].setData(x=self.x_ticks, y=self.data_plot[:,self.channels_to_show_idx[x]] - x * self.scale)

        # Update events
        for x in range(0, len(self.events_detected), 2):
            xh = int(x / 2)
            self.events_curves[xh].setData(x=np.array(
                [self.x_ticks[self.events_detected[x]],
                    self.x_ticks[self.events_detected[x]]]), y=np.array(
                [+1.5 * self.scale,
                    -0.5 * self.scale - self.scale * self.config[
                        'eeg_channels']]))
            self.events_text[xh].setPos(self.x_ticks[self.events_detected[x]],
                self.scale)

    #
    #	Do necessary stuff when scale has changed
    #
    def update_plot_scale(self, new_scale):

        if (new_scale < 1):
            new_scale = 1
        # commented out by dbdq.
        # else:
        #	new_scale = new_scale - new_scale%10

        self.scale = new_scale

        # Y Tick labels
        values = []
        for x in range(0, len(self.channels_to_show_idx)):
            values.append((-x * self.scale,
            self.channel_labels[self.channels_to_show_idx[x]]))

        values_axis = []
        values_axis.append(values)
        values_axis.append([])

        self.main_plot_handler.getAxis('left').setTicks(values_axis)
        self.main_plot_handler.setRange(
            yRange=[+self.scale, -self.scale * len(self.channels_to_show_idx)])
        self.main_plot_handler.setLabel(axis='left',
            text='Scale (uV): ' + str(self.scale))
        self.trigger_help()

        # We force an immediate repaint to avoid "shakiness".
        if (not self.stop_plot):
            self.force_repaint = 1
            self.repaint()

    #
    #	Do necessary stuff when seconds to show have changed
    #
    def update_plot_seconds(self, new_seconds):

        # Do nothing unless...
        if (new_seconds != self.seconds_to_show) and (new_seconds > 0) and (
                new_seconds < 100):
            self.spinBox_time.setValue(new_seconds)
            self.main_plot_handler.setRange(xRange=[0, new_seconds])
            self.x_ticks = np.zeros(self.config['sf'] * new_seconds);
            for x in range(0, self.config['sf'] * new_seconds):
                self.x_ticks[x] = (x * 1) / float(self.config['sf'])

            if (new_seconds > self.seconds_to_show):
                padded_signal = np.zeros((self.config['sf'] * new_seconds,
                self.config['eeg_channels']))
                padded_signal[padded_signal.shape[0] - self.data_plot.shape[0]:,
                :] = self.data_plot
                for x in range(0, len(self.events_detected), 2):
                    self.events_detected[x] += padded_signal.shape[0] - \
                                               self.data_plot.shape[0]
                self.data_plot = padded_signal

            else:
                for x in range(0, len(self.events_detected), 2):
                    self.events_detected[x] -= self.data_plot.shape[0] - \
                                               self.config['sf'] * new_seconds
                self.data_plot = self.data_plot[
                self.data_plot.shape[0] - self.config['sf'] * new_seconds:, :]

            self.seconds_to_show = new_seconds
            self.trigger_help()

            # We force an immediate repaint to avoid "shakiness".
            if (not self.stop_plot):
                self.force_repaint = 1
                self.repaint()

    #
    # Handle TOBI iD events
    #
    def handle_tobiid_input(self):

        data = None
        try:
            data = self.bci.iDsock_bus.recv(512)
            self.bci.idStreamer_bus.Append(data)
        except:
            self.nS = False
            self.dec = 0
            pass

        # deserialize ID message
        if data:
            if self.bci.idStreamer_bus.Has("<tobiid", "/>"):
                msg = self.bci.idStreamer_bus.Extract("<tobiid", "/>")
                self.bci.id_serializer_bus.Deserialize(msg)
                self.bci.idStreamer_bus.Clear()
                tmpmsg = int(self.bci.id_msg_bus.GetEvent())
                if (self.show_TID_events) and (not self.stop_plot):
                    self.addEventPlot("TID", tmpmsg)

            elif self.bci.idStreamer_bus.Has("<tcstatus", "/>"):
                MsgNum = self.bci.idStreamer_bus.Count("<tcstatus")
                for i in range(1, MsgNum - 1):
                    # Extract most of these messages and trash them
                    msg_useless = self.bci.idStreamer_bus.Extract("<tcstatus",
                        "/>")

    #
    # 	Add an event to the scope
    #
    def addEventPlot(self, event_name, event_id):
        if (event_name == "TID"):
            color = pg.mkColor(0, 0, 255)
        elif (event_name == "KEY"):
            color = pg.mkColor(255, 0, 0)
        elif (event_name == "LPT"):
            color = pg.mkColor(0, 255, 0)
        else:
            color = pg.mkColor(255, 255, 255)

        self.events_detected.append(self.data_plot.shape[0] - 1)
        self.events_detected.append(event_id)
        self.events_curves.append(self.main_plot_handler.plot(pen=color,
            x=np.array([self.x_ticks[-1], self.x_ticks[-1]]), y=np.array(
                [+1.5 * self.scale,
                    -1.5 * self.scale * self.config['eeg_channels']])))
        # text = pg.TextItem(event_name + "(" + str(self.events_detected[-1]) + ")", anchor=(1.1,0), fill=(0,0,0), color=color)
        text = pg.TextItem(str(self.events_detected[-1]), anchor=(1.1, 0),
            fill=(0, 0, 0), color=color)
        text.setPos(self.x_ticks[-1], self.scale)
        self.events_text.append(text)
        self.main_plot_handler.addItem(self.events_text[-1])

    #
    #	Calculation of bandpass coefficients.
    #	Order is computed automatically.
    #	Note that if filter is unstable this function crashes (TODO handle problems)
    #
    def butter_bandpass(self, highcut, lowcut, fs, num_ch):
        low = lowcut / (0.5 * fs)
        high = highcut / (0.5 * fs)
        # get the order. TO BE DONE: Sometimes it fails
        ord = buttord(high, low, 2, 40)
        b, a = butter(ord[0], [high, low], btype='band')
        zi = np.zeros([a.shape[0] - 1, num_ch])
        return b, a, zi

    #
    #	Updates the title shown in the scope
    #
    def update_title_scope(self):
        if (hasattr(self, 'main_plot_handler')):
            self.main_plot_handler.setTitle(
                title='TLK: ' + self.bool_parser[self.show_TID_events] +
                      self.bool_parser[self.show_LPT_events] + self.bool_parser[
                          self.show_Key_events] + ', CAR: ' + self.bool_parser[
                          self.apply_car] + ', BP: ' + self.bool_parser[
                          self.apply_bandpass] + ' [' + str(
                    self.doubleSpinBox_hp.value()) + '-' + str(
                    self.doubleSpinBox_lp.value()) + '] Hz')
            # ', BP: ' + self.bool_parser[self.apply_bandpass] + (' [' + str(self.doubleSpinBox_hp.value()) + '-' + str(self.doubleSpinBox_lp.value()) + '] Hz' if self.apply_bandpass else ''))

    #
    #	Shows / hide help in the scope window
    #
    def trigger_help(self):
        if self.show_help:
            self.help.setPos(0, self.scale)
            self.main_plot_handler.addItem(self.help)
            self.help.setZValue(1)
        else:
            self.main_plot_handler.removeItem(self.help)

    # ----------------------------------------------------------------------------------------------------
    # 			EVENT HANDLERS
    # ----------------------------------------------------------------------------------------------------
    def onClicked_button_rec(self):
        # Simply call cl_rpc for this.
        if (len(self.lineEdit_recFilename.text()) > 0):
            if ".gdf" in self.lineEdit_recFilename.text():
                self.pushButton_stoprec.setEnabled(True)
                self.pushButton_rec.setEnabled(False)
                # Popen is more efficient than os.open, since it is non-blocking
                subprocess.Popen(
                    ["cl_rpc", "openxdf", str(self.lineEdit_recFilename.text()),
                        "dummy_log", "dummy_log"], close_fds=True)
                self.statusBar.showMessage(
                    "Recording file " + str(self.lineEdit_recFilename.text()))
            elif ".bdf" in self.lineEdit_recFilename.text():
                self.pushButton_stoprec.setEnabled(True)
                self.pushButton_rec.setEnabled(False)
                subprocess.Popen(
                    ["cl_rpc", "openxdf", str(self.lineEdit_recFilename.text()),
                        "dummy_log", "dummy_log"], close_fds=True)
                self.statusBar.showMessage(
                    "Recording file " + str(self.lineEdit_recFilename.text()))
            else:
                pass

    def onClicked_button_stoprec(self):
        subprocess.Popen(["cl_rpc", "closexdf"], close_fds=True)
        self.pushButton_rec.setEnabled(True)
        self.pushButton_stoprec.setEnabled(False)
        self.statusBar.showMessage("Not recording")

    def onActivated_checkbox_bandpass(self):
        self.apply_bandpass = False
        self.pushButton_bp.setEnabled(self.checkBox_bandpass.isChecked())
        self.doubleSpinBox_hp.setEnabled(self.checkBox_bandpass.isChecked())
        self.doubleSpinBox_lp.setEnabled(self.checkBox_bandpass.isChecked())
        self.update_title_scope()

    def onActivated_checkbox_car(self):
        self.apply_car = self.checkBox_car.isChecked()
        self.update_title_scope()

    def onActivated_checkbox_TID(self):
        self.show_TID_events = self.checkBox_showTID.isChecked()
        self.update_title_scope()

    def onActivated_checkbox_LPT(self):
        self.show_LPT_events = self.checkBox_showLPT.isChecked()
        self.update_title_scope()

    def onActivated_checkbox_Key(self):
        self.show_Key_events = self.checkBox_showKey.isChecked()
        self.update_title_scope()

    def onValueChanged_spinbox_time(self):
        self.update_plot_seconds(self.spinBox_time.value())

    def onActivated_combobox_scale(self):
        self.update_plot_scale(
            self.scales_range[self.comboBox_scale.currentIndex()])

    def onClicked_button_bp(self):
        if (self.doubleSpinBox_lp.value() > self.doubleSpinBox_hp.value()):
            self.apply_bandpass = True
            self.b, self.a, self.zi = self.butter_bandpass(
                self.doubleSpinBox_hp.value(), self.doubleSpinBox_lp.value(),
                self.config['sf'], self.config['eeg_channels'])
        self.update_title_scope()

    def onSelectionChanged_table(self):

        # Remove current plot
        for x in range(0, len(self.channels_to_show_idx)):
            self.main_plot_handler.removeItem(self.curve_eeg[x])

        # Which channels should I plot?
        self.channels_to_show_idx = []
        self.channels_to_hide_idx = []
        idx = 0
        for y in range(0, 4):
            for x in range(0, NUM_X_CHANNELS):
                if (idx < self.config['eeg_channels']):
                    #if (self.table_channels.isItemSelected( # Qt4 only
                    if (QtGui.QTableWidgetItem.isSelected( # Qt5
                        self.table_channels.item(x, y))):
                        self.channels_to_show_idx.append(idx)
                    else:
                        self.channels_to_hide_idx.append(idx)
                    idx += 1

        # Add new plots
        self.curve_eeg = []
        for x in range(0, len(self.channels_to_show_idx)):
            self.curve_eeg.append(self.main_plot_handler.plot(x=self.x_ticks,
                y=self.data_plot[:, self.channels_to_show_idx[x]],
                pen=self.colors[self.channels_to_show_idx[x] % NUM_X_CHANNELS, :]))
            self.curve_eeg[-1].setDownsampling(ds=self.subsampling_value,
                auto=False, method="mean")

        # Update CAR so it's computed based only on the shown channels
        if (len(self.channels_to_show_idx) > 1):
            self.matrix_car = np.zeros(
                (self.config['eeg_channels'], self.config['eeg_channels']),
                dtype=float)
            self.matrix_car[:, :] = -1 / float(len(self.channels_to_show_idx))
            np.fill_diagonal(self.matrix_car,
                1 - (1 / float(len(self.channels_to_show_idx))))
            for x in range(0, len(self.channels_to_hide_idx)):
                self.matrix_car[self.channels_to_hide_idx[x], :] = 0
                self.matrix_car[:, self.channels_to_hide_idx[x]] = 0

        # Refresh the plot
        self.update_plot_scale(self.scale)

    def keyPressEvent(self, event):
        key = event.key()
        if (key == QtCore.Qt.Key_Escape):
            self.closeEvent(None)
        if (key == QtCore.Qt.Key_H):
            self.show_help = not self.show_help
            self.trigger_help()
        if (key == QtCore.Qt.Key_Up):
            # Python's log(x, 10) has a rounding bug. Use log10(x) instead.
            new_scale = self.scale + max(1, 10 ** int(math.log10(self.scale)))
            self.update_plot_scale(new_scale)
        if (key == QtCore.Qt.Key_Space):
            self.stop_plot = not self.stop_plot
        if (key == QtCore.Qt.Key_Down):
            if self.scale >= 2:
                # Python's log(x, 10) has a rounding bug. Use log10(x) instead.
                new_scale = self.scale - max(1,
                    10 ** int(math.log10(self.scale - 1)))
                self.update_plot_scale(new_scale)
        if (key == QtCore.Qt.Key_Left):
            self.update_plot_seconds(self.seconds_to_show - 1)
        if (key == QtCore.Qt.Key_Right):
            self.update_plot_seconds(self.seconds_to_show + 1)
        if (key == QtCore.Qt.Key_L):
            self.checkBox_showLPT.setChecked(
                not self.checkBox_showLPT.isChecked())
        if (key == QtCore.Qt.Key_T):
            self.checkBox_showTID.setChecked(
                not self.checkBox_showTID.isChecked())
        if (key == QtCore.Qt.Key_K):
            self.checkBox_showKey.setChecked(
                not self.checkBox_showKey.isChecked())
        if (key == QtCore.Qt.Key_C):
            self.checkBox_car.setChecked(not self.checkBox_car.isChecked())
        if (key == QtCore.Qt.Key_B):
            self.checkBox_bandpass.setChecked(
                not self.checkBox_bandpass.isChecked())
            if self.checkBox_bandpass.isChecked():
                self.pushButton_bp.click()
        if ((key >= QtCore.Qt.Key_0) and (key <= QtCore.Qt.Key_9)):
            if (self.show_Key_events) and (not self.stop_plot):
                self.addEventPlot("KEY", 990 + key - QtCore.Qt.Key_0)
                # self.bci.id_msg_bus.SetEvent(990 + key - QtCore.Qt.Key_0)
                # self.bci.iDsock_bus.sendall(self.bci.id_serializer_bus.Serialize());
                # 666

    #
    #	Function called when a closing event was triggered.
    #
    def closeEvent(self, event):
        '''
        reply = QtGui.QMessageBox.question(self, "Quit", "Are you sure you want to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
        if (reply == QtGui.QMessageBox.Yes):
            if (self.pushButton_stoprec.isEnabled()):
                subprocess.Popen(["cl_rpc", "closexdf"], close_fds=True)
            self.fin.close()
            exit()
        '''

        # leeq
        if (self.pushButton_stoprec.isEnabled()):
            subprocess.Popen(["cl_rpc", "closexdf"], close_fds=True)
        # quit()
        app.quit()

        # ----------------------------------------------------------------------------------------------------
        # 		END OF EVENT HANDLERS
        # ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    if len(sys.argv) == 2:
        amp_name = sys.argv[1]
        amp_serial = None
    elif len(sys.argv) == 3:
        amp_name, amp_serial = sys.argv[1:3]
    else:
        amp_name, amp_serial = pu.search_lsl()
    if amp_name == 'None':
        amp_name = None
    logger.info('Connecting to a server %s (Serial %s).' % (amp_name, amp_serial))

    app = QtGui.QApplication(sys.argv)
    ex = Scope(amp_name, amp_serial)
    sys.exit(app.exec_())
