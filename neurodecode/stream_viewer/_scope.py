"""
 EEG Scope
 Iñaki Iturrate, Kyuhwa Lee, Arnaud Desvachez

 TODO
	- Should move to VisPY: http://vispy.org/plot.html#module-vispy.plot but still under development
	- Events should be stored in a class.
"""

from ..gui.streams import redirect_stdout_to_queue
from ..stream_recorder import StreamRecorder
from ..stream_receiver import StreamReceiver
from .. import logger
from .ui_mainwindow_Viewer import Ui_MainWindow
from configparser import RawConfigParser
from PyQt5.QtWidgets import (QMainWindow, QTableWidgetItem,
                             QHeaderView, QFileDialog)
from scipy.signal import butter, sosfilt
from PyQt5.QtGui import QPainter
from pathlib import Path
from PyQt5 import QtCore
import multiprocessing as mp
import pyqtgraph as pg
import numpy as np
import math
import sys
import os

# import time

DEBUG_TRIGGER = False  # TODO: parameterize

class _Scope(QMainWindow):
    """
    Internal class.

    Load UI, data acquisition and ploting.
    """

    def __init__(self, stream_name, state=mp.Value('i', 1), queue=None):
        """
        Constructor.
        """
        super(_Scope, self).__init__()

        self.stream_name = stream_name
        self.state = state
        self.recordState = mp.Value('i', 0)

        redirect_stdout_to_queue(logger, queue, 'INFO')
        logger.info('Viewer launched')

        self.load_ui()
        self.init_scope()

    def load_ui(self):
        """
        Load the GUI from .ui file created by QtCreator.
        """
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.setGeometry(100, 100,
                         self.geometry().width(),
                         self.geometry().height())
        self.setFixedSize(self.geometry().width(),
                          self.geometry().height())

    # ------------------------- INIT_SCOPE -------------------------
    def init_scope(self):
        """
        Main init function.
        """
        self.load_config_file()
        self.init_loop()
        self.init_panel_GUI()
        self.init_scope_GUI()
        self.init_timer()

    def load_config_file(self):
        """
        Load predefined parameters from the config file.
        """
        path2_viewerFolder = Path(__file__).parent
        self.scope_settings = RawConfigParser(
            allow_no_value=True, inline_comment_prefixes=('#', ';'))
        self.scope_settings.read(str(path2_viewerFolder/'.scope_settings.ini'))

    # ------------------------- INIT_LOOP -------------------------
    def init_loop(self, bufsize=0.2, winsize=0.2):
        """
        Instance a StreamReceiver and extract info from the stream.
        """
        self.sr = StreamReceiver(bufsize=bufsize, winsize=winsize,
                                 stream_name=self.stream_name)
        self.sr.streams[self.stream_name].blocking = False
        # time.sleep(bufsize) # Delay to fill the LSL buffer.

        self.config = {
            'sf': int(
                self.sr.streams[self.stream_name].sample_rate),
            'samples': round(
                self.sr.streams[self.stream_name].sample_rate * winsize),
            'eeg_channels': len(
                self.sr.streams[self.stream_name].ch_list[1:]),
            'exg_channels': 0,
            'tri_channels': 1,
        }

        # For now, not a fixed number of samples per chunk --> TO FIX
        self.tri = np.zeros(self.config['samples'])
        self.eeg = np.zeros(
            (self.config['samples'], self.config['eeg_channels']),
            dtype=np.float)
        self.exg = np.zeros(
            (self.config['samples'], self.config['exg_channels']),
            dtype=np.float)

        self._last_tri = 0
        self._ts_list = []  # timestamps list

    # ----------------------- INIT_PANEL_GUI -----------------------
    def init_panel_GUI(self):
        """
        Initialize control panel parameters.
        """
        self.show_events()
        self.connect_signals_to_slots()
        self.set_checked_widgets()

        self._ui.pushButton_stoprec.setEnabled(False)
        self._ui.comboBox_scale.setCurrentIndex(2)

        # self._ui.pushButton_bp.setDisabled(True)

        self.fill_table_channels()
        self.set_window_size_policy()
        self.show()

    def show_events(self, tid=False, lpt=False, key=False):
        """
        Display or not events.
        """
        self._show_TID_events = tid
        self._show_LPT_events = lpt
        self._show_Key_events = key

    def connect_signals_to_slots(self):
        """
        Event handler. Connect QT signals to slots.
        """
        self._ui.comboBox_scale.activated.connect(
            self.onActivated_combobox_scale)
        self._ui.spinBox_time.valueChanged.connect(
            self.onValueChanged_spinbox_time)
        self._ui.checkBox_car.stateChanged.connect(
            self.onActivated_checkbox_car)
        self._ui.checkBox_bandpass.stateChanged.connect(
            self.onActivated_checkbox_bandpass)
        self._ui.checkBox_showTID.stateChanged.connect(
            self.onActivated_checkbox_TID)
        self._ui.checkBox_showLPT.stateChanged.connect(
            self.onActivated_checkbox_LPT)
        self._ui.checkBox_showKey.stateChanged.connect(
            self.onActivated_checkbox_Key)
        self._ui.pushButton_bp.clicked.connect(
            self.onClicked_button_bp)
        self._ui.pushButton_recdir.clicked.connect(
            self.on_click_button_recdir)
        self._ui.pushButton_rec.clicked.connect(
            self.onClicked_button_rec)
        self._ui.pushButton_stoprec.clicked.connect(
            self.onClicked_button_stoprec)

    def set_checked_widgets(self):
        """
        Set checkBox widgets to checked state.
        """
        self._ui.checkBox_car.setChecked(
            int(self.scope_settings.get("filtering", "apply_car_filter")))
        self._ui.checkBox_bandpass.setChecked(
            int(self.scope_settings.get("filtering", "apply_bandpass_filter")))
        self._ui.checkBox_showTID.setChecked(
            int(self.scope_settings.get("plot", "show_TID_events")))
        self._ui.checkBox_showLPT.setChecked(
            int(self.scope_settings.get("plot", "show_LPT_events")))
        self._ui.checkBox_showKey.setChecked(
            int(self.scope_settings.get("plot", "show_Key_events")))
        self._ui.statusBar.showMessage("[Not recording]")

    def fill_table_channels(self):
        """
        Fill the channels table with the available names.
        """
        idx = 0
        self.channels_to_show_idx = []

        nb_channels = self.config['eeg_channels'] + self.config['exg_channels']
        self.set_table_size(nb_channels)

        for x in range(0, self._nb_table_rows):
            for y in range(0, self._nb_table_columns):
                if (idx < self.config['eeg_channels']):
                    self._ui.table_channels.setItem(
                        x, y, QTableWidgetItem(idx))
                    self._ui.table_channels.item(x, y).setTextAlignment(
                        QtCore.Qt.AlignCenter)
                    self._ui.table_channels.item(x, y).setSelected(True)  # Qt5
                    self.channels_to_show_idx.append(idx)
                else:
                    self._ui.table_channels.setItem(x, y,
                                                    QTableWidgetItem("N/A"))
                    self._ui.table_channels.item(x, y).setFlags(
                        QtCore.Qt.NoItemFlags)
                    self._ui.table_channels.item(x, y).setTextAlignment(
                        QtCore.Qt.AlignCenter)
                idx += 1

        self._ui.table_channels.verticalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._ui.table_channels.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        self._ui.table_channels.itemSelectionChanged.connect(
            self.onSelectionChanged_table)

    def set_table_size(self, nb_channels):
        """
        Compute the numbers of raws and columns of the channel table.
        """
        if nb_channels > 64:
            self._nb_table_columns = 8
        else:
            self._nb_table_columns = 4

        self._nb_table_rows = math.ceil(nb_channels/self._nb_table_columns)

        self._ui.table_channels.setRowCount(self._nb_table_rows)
        self._ui.table_channels.setColumnCount(self._nb_table_columns)

    def set_window_size_policy(self):
        """
        Set window's size and policy.
        """
        self.screen_width = 522
        self.screen_height = 160
        self.setWindowTitle('EEG Scope Panel')
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()

    # ----------------------- INIT_SCOPE_GUI -----------------------
    def init_scope_GUI(self):
        """
        Initialize scope parameters.
        """
        self.bool_parser = {True: '1', False: '0'}

        # Scales available in the GUI.
        self.scales_range = [1, 10, 25, 50, 100, 250, 500, 1000, 2500, 100000]
        # Scale in uV
        self.scale = float(self.scope_settings.get("plot", "scale_plot"))
        # Time window to show in seconds
        self.seconds_to_show = int(
            self.scope_settings.get("plot", "time_plot"))

        self.init_graph()

        # Plotting colors. If channels > 16, colors will cycle to the beginning
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
            self.curve_eeg.append(
                self._main_plot_handler.plot(
                    x=self.x_ticks,
                    y=self.data_plot[:, self.channels_to_show_idx[x]],
                    pen=pg.mkColor(
                        self.colors[self.channels_to_show_idx[x] % 16, :])))

        # Events data
        self.events_detected = []
        self.events_curves = []
        self.events_text = []

        # Filters
        self.init_car()
        self.init_bandpass()
        self.update_title_scope()

        # Help variables
        self.show_help = 0
        self.help = pg.TextItem(
            "Stream Viewer \n" +
            "-----------------------------------------"
            "-----------------------------------------\n" +
            "C: De/activate CAR Filter\n" +
            "B: De/activate Bandpass Filter (with current settings)\n" +
            "T: Show/hide TiD events\n" +
            "L: Show/hide LPT events\n" +
            "K: Show/hide Key events. " +
            "If not shown, they are NOT recorded!\n" +
            "0-9: Add a user-specific Key event. " +
            "Do not forget to write down why you marked it.\n" +
            "Up, down arrow keys: " +
            "Increase/decrease the scale, steps of 10 uV\n" +
            "Left, right arrow keys: " +
            "Increase/decrease the time to show, steps of 1 s\n" +
            "Spacebar: Stop the scope plotting, " +
            "whereas data acquisition keeps running (EXPERIMENTAL)\n" +
            "Esc: Exits the scope",
            anchor=(0, 0), border=(70, 70, 70),
            fill=pg.mkColor(20, 20, 20, 200), color=(255, 255, 255))

        # Stop plot functionality
        self.stop_plot = 0

        # Force repaint even when we shouldn't repaint.
        self.force_repaint = 0

    def init_graph(self):
        """
        Init the PyQTGraph plot.
        """
        self._win = pg.GraphicsWindow()
        self.set_win_geometry_title()
        self._win.keyPressEvent = self.keyPressEvent
        self._win.show()

        self._main_plot_handler = self._win.addPlot()

        # Y Tick labels. Use values from the config file.
        self.channel_labels = []
        values = []
        ch_names = np.array(next(iter(self.sr.streams.values())).ch_list)
        self.channel_labels = ch_names[1:]
        for x in range(0, len(self.channels_to_show_idx)):
            values.append(
                (-x * self.scale,
                 self.channel_labels[self.channels_to_show_idx[x]]))

        values_axis = []
        values_axis.append(values)
        values_axis.append([])

        # Update table labels with current names
        idx = 0
        for x in range(0, self._nb_table_rows):
            for y in range(0, self._nb_table_columns):
                if (idx < self.config['eeg_channels']):
                    self._ui.table_channels.item(x, y).setText(
                        self.channel_labels[idx])
                idx += 1

        # Plot initialization
        self._main_plot_handler.getAxis('left').setTicks(values_axis)
        self._main_plot_handler.setRange(
            xRange=[0, self.seconds_to_show],
            yRange=[+1.5*self.scale,
                    -0.5*self.scale - self.scale*self.config['eeg_channels']])
        self._main_plot_handler.disableAutoRange()
        self._main_plot_handler.showGrid(y=True)
        self._main_plot_handler.setLabel(
            axis='left', text='Scale (uV): ' + str(self.scale))
        self._main_plot_handler.setLabel(axis='bottom', text='Time (s)')

        # X axis
        self.x_ticks = np.zeros(self.config['sf'] * self.seconds_to_show)
        for x in range(0, self.config['sf'] * self.seconds_to_show):
            self.x_ticks[x] = (x * 1) / float(self.config['sf'])

    def set_win_geometry_title(self):
        """
        Set the title and the geometry of the PyQTGraph window based on
        MainWindow size.
        """
        self._win.setWindowTitle('EEG Scope')
        self._win.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self._win.setWindowFlags(QtCore.Qt.WindowMaximizeButtonHint)
        # Position next to the panel window
        self._win.setGeometry(self.geometry().x() + self.width(),
                              self.geometry().y(),
                              self.width() * 2,
                              self.height())

    def init_car(self):
        """
        Init the Common Average Reference.
        """
        self.apply_car = int(
            self.scope_settings.get("filtering", "apply_car_filter"))

        self._ui.checkBox_bandpass.setChecked(self.apply_car)

    def init_bandpass(self):
        """
        Init the bandpass filtering parameters high and low cutoff.
        """
        self.apply_bandpass = int(
            self.scope_settings.get("filtering", "apply_bandpass_filter"))

        self._ui.checkBox_bandpass.setChecked(self.apply_bandpass)

        if (self.apply_bandpass):
            self._ui.doubleSpinBox_hp.setValue(float(
                self.scope_settings.get(
                    "filtering", "bandpass_cutoff_frequency").split(' ')[0]))
            self._ui.doubleSpinBox_lp.setValue(float(
                self.scope_settings.get(
                    "filtering", "bandpass_cutoff_frequency").split(' ')[1]))
            self._ui.pushButton_bp.click()

    # ------------------------- INIT_TIMER -------------------------
    def init_timer(self):
        """
        Initializes the QT timer, which will call the update function every
        20 ms.
        """
        QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.flush()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(20)

    # ------------------------- ACQUISITION -------------------------
    def update_loop(self):
        """
        Main update function (connected to the timer).
        """
        #  Sharing variable to stop at the GUI level
        if not self.state.value:
            logger.info('Viewer stopped.')
            sys.exit()

        try:
            self.read_lsl_stream()
            if len(self._ts_list) > 0:
                self.filter_signal()        # Filter acquired data
                self.update_ringbuffers()   # Update the plotting info
                if not self.stop_plot:
                    self.repaint()          # Call paint event
        except:
            logger.exception('Exception. Dropping into a shell.')

    def read_lsl_stream(self):
        """
        Read chunk of signal from LSL stream.
        """
        self.sr.acquire()
        data, self._ts_list = self.sr.get_buffer()
        self.sr.reset_all_buffers()

        if len(self._ts_list) == 0:
            return

        n = self.config['eeg_channels']

        self.tri = np.reshape(data[:, 0], (-1, 1))      # samples x 1
        self.eeg = np.reshape(data[:, 1:], (-1, n))     # samples x channels

        if DEBUG_TRIGGER:
            # show trigger value
            try:
                trg_value = max(self.tri)
                if trg_value > 0:
                    logger.info(f'Received trigger {trg_value}')
            except:
                logger.exception(f'Error! self.tri = {self.tri}')

    def filter_signal(self):
        """
        Bandpass + CAR filtering.
        """
        if self.apply_bandpass:
            self.eeg, self.zi = sosfilt(self.sos, self.eeg, 0, self.zi)

        # We only apply CAR if selected AND there are at least 2 channels.
        # Otherwise it makes no sense.
        if self.apply_car and len(self.channels_to_show_idx) > 1:
            # Compute CAR virtual channel
            car_ch = np.mean(
                self.eeg[:, self.channels_to_show_idx], axis=1)

            # Apply CAR
            self.eeg -= car_ch.reshape((-1, 1))

    def update_ringbuffers(self):
        """
        Update ringbuffers and events for plotting.
        """
        # leeq
        self.data_plot = np.roll(self.data_plot, - len(self._ts_list), 0)
        self.data_plot[-len(self._ts_list):, :] = self.eeg

        # We have to remove those indexes that reached time = 0
        delete_indices_e = []
        delete_indices_c = []
        for x in range(0, len(self.events_detected), 2):
            xh = int(x / 2)
            self.events_detected[x] -= len(self._ts_list)  # leeq
            if self.events_detected[x] < 0 and not self.stop_plot:
                delete_indices_e.append(x)
                delete_indices_e.append(x + 1)
                delete_indices_c.append(xh)
                self.events_curves[xh].clear()
                self._main_plot_handler.removeItem(self.events_text[xh])

        self.events_detected = [i for j, i in enumerate(self.events_detected)
                                if j not in delete_indices_e]
        self.events_curves = [i for j, i in enumerate(self.events_curves)
                              if j not in delete_indices_c]
        self.events_text = [i for j, i in enumerate(self.events_text)
                            if j not in delete_indices_c]

        # Find LPT events and add them.
        if self._show_LPT_events and not self.stop_plot:
            for x in range(len(self.tri)):
                tri = int(self.tri[x])
                if tri != 0 and (tri > self._last_tri):
                    self.addEventPlot("LPT", tri)
                    logger.info(f'Trigger {tri} received.')
                self._last_tri = tri

    # --------------------------- PAINT ---------------------------
    def paintEvent(self, e):
        """
        Called by repaint().
        """
        # Distinguish between paint events from timer and event QT widget resizing, clicking etc (sender is None)
        # We should only paint when the timer triggered the event.
        # Just in case, there's a flag to force a repaint even when we shouldn't repaint
        sender = self.sender()
        if 'force_repaint' not in self.__dict__.keys():
            logger.warning('force_repaint is not set! Is it a Qt bug?')
            self.force_repaint = 0
        if sender is None and not self.force_repaint:
            pass
        else:
            self.force_repaint = 0
            qp = QPainter()
            qp.begin(self)
            # Update the interface
            self.paintInterface(qp)
            qp.end()

    def paintInterface(self, qp):
        """
        Update stuff on the interface.
        Only graphical updates should be added here.
        """
        # Update EEG channels
        for x in range(0, len(self.channels_to_show_idx)):
            self.curve_eeg[x].setData(
                x=self.x_ticks,
                y=self.data_plot[:, self.channels_to_show_idx[x]] - x*self.scale)

        # Update events
        for x in range(0, len(self.events_detected), 2):
            xh = int(x / 2)
            self.events_curves[xh].setData(
                x=np.array(
                    [self.x_ticks[self.events_detected[x]],
                     self.x_ticks[self.events_detected[x]]]),
                y=np.array(
                    [+1.5*self.scale,
                     -0.5*self.scale - self.scale*self.config['eeg_channels']]))
            self.events_text[xh].setPos(
                self.x_ticks[self.events_detected[x]],
                self.scale)

    # --------------------------- UPDATE ---------------------------
    def update_plot_scale(self, new_scale):
        """
        Do necessary stuff when scale has changed.
        """
        self.scale = new_scale

        # Y Tick labels
        values = []
        for x in range(0, len(self.channels_to_show_idx)):
            values.append((-x * self.scale,
                           self.channel_labels[self.channels_to_show_idx[x]]))

        values_axis = []
        values_axis.append(values)
        values_axis.append([])

        self._main_plot_handler.getAxis('left').setTicks(values_axis)
        self._main_plot_handler.setRange(
            yRange=[+self.scale, -self.scale*len(self.channels_to_show_idx)])
        self._main_plot_handler.setLabel(
            axis='left', text='Scale (uV): ' + str(self.scale))
        self.trigger_help()

        # We force an immediate repaint to avoid "shakiness".
        if not self.stop_plot:
            self.force_repaint = 1
            self.repaint()

    def update_plot_seconds(self, new_seconds):
        """
        Do necessary stuff when seconds to show have changed.
        """
        # Do nothing unless...
        if (new_seconds != self.seconds_to_show) and (0 < new_seconds < 100):
            self._ui.spinBox_time.setValue(new_seconds)
            self._main_plot_handler.setRange(xRange=[0, new_seconds])
            self.x_ticks = np.zeros(self.config['sf'] * new_seconds)
            for x in range(0, self.config['sf']*new_seconds):
                self.x_ticks[x] = x / float(self.config['sf'])

            if new_seconds > self.seconds_to_show:
                padded_signal = np.zeros(
                    (self.config['sf'] * new_seconds,
                     self.config['eeg_channels']))
                padded_signal[padded_signal.shape[0]-self.data_plot.shape[0]:,
                              :] = self.data_plot
                for x in range(0, len(self.events_detected), 2):
                    self.events_detected[x] += padded_signal.shape[0] - \
                        self.data_plot.shape[0]
                self.data_plot = padded_signal

            else:
                for x in range(0, len(self.events_detected), 2):
                    self.events_detected[x] -= self.data_plot.shape[0] - \
                        self.config['sf']*new_seconds
                self.data_plot = self.data_plot[
                    self.data_plot.shape[0]-self.config['sf']*new_seconds:, :]

            self.seconds_to_show = new_seconds
            self.trigger_help()

            # We force an immediate repaint to avoid "shakiness".
            if not self.stop_plot:
                self.force_repaint = 1
                self.repaint()

    def addEventPlot(self, event_name, event_id):
        """
        Add an event to the scope
        """
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
        self.events_curves.append(
            self._main_plot_handler.plot(
                pen=color,
                x=np.array([self.x_ticks[-1], self.x_ticks[-1]]),
                y=np.array([+1.5 * self.scale, -1.5 * self.scale * self.config['eeg_channels']])))
        # text = pg.TextItem(event_name + "(" + str(self.events_detected[-1]) + ")", anchor=(1.1,0), fill=(0,0,0), color=color)
        text = pg.TextItem(str(self.events_detected[-1]), anchor=(1.1, 0),
                           fill=(0, 0, 0), color=color)
        text.setPos(self.x_ticks[-1], self.scale)
        self.events_text.append(text)
        self._main_plot_handler.addItem(self.events_text[-1])

    def update_title_scope(self):
        """
        Updates the title shown in the scope
        """
        if hasattr(self, 'main_plot_handler'):
            self._main_plot_handler.setTitle(
                title='TLK: ' +
                self.bool_parser[self._show_TID_events] +
                self.bool_parser[self._show_LPT_events] +
                self.bool_parser[self._show_Key_events] +
                ', CAR: ' +
                self.bool_parser[self.apply_car] +
                ', BP: ' +
                self.bool_parser[self.apply_bandpass] +
                ' [' + str(self._ui.doubleSpinBox_hp.value()) +
                '-' + str(self._ui.doubleSpinBox_lp.value()) + '] Hz')

    # --------------------------- UTILS ---------------------------
    def handle_tobiid_input(self):
        """
        Handle TOBI iD events.
        TODO: What is this?? self.bci is not even defined.
        """
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
                if (self._show_TID_events) and (not self.stop_plot):
                    self.addEventPlot("TID", tmpmsg)

            elif self.bci.idStreamer_bus.Has("<tcstatus", "/>"):
                MsgNum = self.bci.idStreamer_bus.Count("<tcstatus")
                for i in range(1, MsgNum - 1):
                    # Extract most of these messages and trash them
                    self.bci.idStreamer_bus.Extract("<tcstatus", "/>")

    def butter_bandpass(self, lowcut, highcut, fs, num_ch):
        """
        Calculation of bandpass coefficients.

        TODO: AUTOMATIC ORDER COMPUTATION
        (If filter is unstable this function crashes (TODO handle problems))
        """
        low = lowcut / (0.5 * fs)
        high = highcut / (0.5 * fs)
        sos = butter(2, [low, high], btype='band', output='sos', fs=fs)
        zi = np.zeros((sos.shape[0], 2, num_ch))
        return sos, zi

    def trigger_help(self):
        """
        Shows / hide help in the scope window.
        """
        if self.show_help:
            self.help.setPos(0, self.scale)
            self._main_plot_handler.addItem(self.help)
            self.help.setZValue(1)
        else:
            self._main_plot_handler.removeItem(self.help)

    # -------------------------------------------------------------------
    # --------------------------- EVENT HANDLERS ------------------------
    # -------------------------------------------------------------------
    def on_click_button_recdir(self):
        """
        Open a QFileDialog to select the recording directory.
        """
        defaultPath = os.environ["NEUROD_DATA"]  # TODO: Change with a non PATH variable path
        path_name = QFileDialog.getExistingDirectory(
            caption="Choose the recording directory", directory=defaultPath)

        if path_name:
            self._ui.lineEdit_recdir.setText(path_name)
            self._ui.pushButton_rec.setEnabled(True)

    # -------------------------------------------------------------------
    def onClicked_button_rec(self):
        self._ui.pushButton_stoprec.setEnabled(True)
        self._ui.pushButton_rec.setEnabled(False)

        record_dir = self._ui.lineEdit_recdir.text()
        with self.recordState.get_lock():
            self.recordState.value = 1

        recorder = StreamRecorder(record_dir, logger, self.recordState)
        recorder.start(stream_name=self.stream_name, verbose=False)
        self._ui.statusBar.showMessage("Recording to" + record_dir)

    # -------------------------------------------------------------------
    def onClicked_button_stoprec(self):
        with self.recordState.get_lock():
            self.recordState.value = 0
        self._ui.pushButton_rec.setEnabled(True)
        self._ui.pushButton_stoprec.setEnabled(False)
        self._ui.statusBar.showMessage("Not recording.")

    # -------------------------------------------------------------------
    def onActivated_checkbox_bandpass(self):
        self.apply_bandpass = False
        self._ui.pushButton_bp.setEnabled(
            self._ui.checkBox_bandpass.isChecked())
        self._ui.doubleSpinBox_hp.setEnabled(
            self._ui.checkBox_bandpass.isChecked())
        self._ui.doubleSpinBox_lp.setEnabled(
            self._ui.checkBox_bandpass.isChecked())
        self.update_title_scope()

    # -------------------------------------------------------------------
    def onActivated_checkbox_car(self):
        self.apply_car = self._ui.checkBox_car.isChecked()
        self.update_title_scope()

    # -------------------------------------------------------------------
    def onActivated_checkbox_TID(self):
        self._show_TID_events = self._ui.checkBox_showTID.isChecked()
        self.update_title_scope()

    # -------------------------------------------------------------------
    def onActivated_checkbox_LPT(self):
        self._show_LPT_events = self._ui.checkBox_showLPT.isChecked()
        self.update_title_scope()

    # -------------------------------------------------------------------
    def onActivated_checkbox_Key(self):
        self._show_Key_events = self._ui.checkBox_showKey.isChecked()
        self.update_title_scope()

    # -------------------------------------------------------------------
    def onValueChanged_spinbox_time(self):
        self.update_plot_seconds(self._ui.spinBox_time.value())

    # -------------------------------------------------------------------
    def onActivated_combobox_scale(self):
        self.update_plot_scale(
            self.scales_range[self._ui.comboBox_scale.currentIndex()])

    # -------------------------------------------------------------------
    def onClicked_button_bp(self):
        if self._ui.doubleSpinBox_lp.value() > self._ui.doubleSpinBox_hp.value():
            self.apply_bandpass = True
            self.sos, self.zi = self.butter_bandpass(
                self._ui.doubleSpinBox_lp.value(),
                self._ui.doubleSpinBox_hp.value(),
                self.config['sf'],
                self.config['eeg_channels'])
        self.update_title_scope()

    # -------------------------------------------------------------------
    def onSelectionChanged_table(self):
        # Remove current plot
        for x in range(0, len(self.channels_to_show_idx)):
            self._main_plot_handler.removeItem(self.curve_eeg[x])

        # Which channels should I plot?
        self.channels_to_show_idx = []
        self.channels_to_hide_idx = []
        idx = 0
        for x in range(0, self._nb_table_rows):
            for y in range(0, self._nb_table_columns):
                if (idx < self.config['eeg_channels']):
                    if (QTableWidgetItem.isSelected(
                            self._ui.table_channels.item(x, y))):
                        self.channels_to_show_idx.append(idx)
                    else:
                        self.channels_to_hide_idx.append(idx)
                    idx += 1

        # Add new plots
        self.curve_eeg = []
        for x in range(0, len(self.channels_to_show_idx)):
            self.curve_eeg.append(
                self._main_plot_handler.plot(
                    x=self.x_ticks,
                    y=self.data_plot[:, self.channels_to_show_idx[x]],
                    pen=self.colors[
                        self.channels_to_show_idx[x] % self._nb_table_rows, :]
                )
            )
            self.curve_eeg[-1].setDownsampling(
                ds=self.subsampling_value, auto=False, method="mean")

        # Refresh the plot
        self.update_plot_scale(self.scale)

    # -------------------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.closeEvent(None)
        if key == QtCore.Qt.Key_H:
            self.show_help = not self.show_help
            self.trigger_help()
        if key == QtCore.Qt.Key_Up:
            # Python's log(x, 10) has a rounding bug. Use log10(x) instead.
            new_scale = self.scale + max(1, 10 ** int(math.log10(self.scale)))
            self.update_plot_scale(new_scale)
        if key == QtCore.Qt.Key_Space:
            self.stop_plot = not self.stop_plot
        if key == QtCore.Qt.Key_Down:
            if self.scale >= 2:
                # Python's log(x, 10) has a rounding bug. Use log10(x) instead.
                new_scale = self.scale - max(
                    1, 10 ** int(math.log10(self.scale - 1)))
                self.update_plot_scale(new_scale)
        if key == QtCore.Qt.Key_Left:
            self.update_plot_seconds(self.seconds_to_show - 1)
        if key == QtCore.Qt.Key_Right:
            self.update_plot_seconds(self.seconds_to_show + 1)
        if key == QtCore.Qt.Key_L:
            self._ui.checkBox_showLPT.setChecked(
                not self._ui.checkBox_showLPT.isChecked())
        if key == QtCore.Qt.Key_T:
            self._ui.checkBox_showTID.setChecked(
                not self._ui.checkBox_showTID.isChecked())
        if key == QtCore.Qt.Key_K:
            self._ui.checkBox_showKey.setChecked(
                not self._ui.checkBox_showKey.isChecked())
        if key == QtCore.Qt.Key_C:
            self._ui.checkBox_car.setChecked(
                not self._ui.checkBox_car.isChecked())
        if key == QtCore.Qt.Key_B:
            self._ui.checkBox_bandpass.setChecked(
                not self._ui.checkBox_bandpass.isChecked())
            if self._ui.checkBox_bandpass.isChecked():
                self._ui.pushButton_bp.click()
        if key >= QtCore.Qt.Key_0 and key <= QtCore.Qt.Key_9:
            if self._show_Key_events and not self.stop_plot:
                self.addEventPlot("KEY", 990 + key - QtCore.Qt.Key_0)
                # self.bci.id_msg_bus.SetEvent(990 + key - QtCore.Qt.Key_0)
                # self.bci.iDsock_bus.sendall(self.bci.id_serializer_bus.Serialize());
                # 666

    # -------------------------------------------------------------------
    def closeEvent(self, event):
        """
        Function called when a closing event was triggered.
        """
        if self._ui.pushButton_stoprec.isEnabled():
            # Stop Recording
            with self.recordState.get_lock():
                self.recordState.value = 0
        # Stop viewer
        with self.state.get_lock():
            self.state.value = 0
