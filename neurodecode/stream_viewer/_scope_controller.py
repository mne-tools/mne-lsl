import math
from pathlib import Path
from configparser import RawConfigParser

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QHeaderView,
                             QTableWidgetItem, QFileDialog)
try:
    from .backends import _BackendVispy, _BackendPyQt5
except:
    from .backends import _BackendPyQt5
from .ui_ScopeSettings import UI_MainWindow
from ..stream_recorder import StreamRecorder
from .. import logger


class _ScopeControllerUI(QMainWindow):
    """
    GUI controlling the scope and the backend.

    Parameters
    ----------
    scope : neurodecode.stream_viewer._scope._Scope
        The scope connected to a stream receiver acquiring the data and applying
        filtering. The scope has a buffer of _scope._BUFFER_DURATION (default: 30s).
    backend : str
        One of the supported backend's name. Supported 'vispy', 'pyqt5'.
        If None, load the first backend in the order: Vispy, PyQt5.
    """
    def __init__(self, scope, backend=None):
        super().__init__()

        # Select the backend
        if backend == 'vispy' or backend is None:
            try:
                self.backend = _BackendVispy(scope)
            except:
                logger.warning("Could not use the backend 'Vispy'. "
                               "Resorting to 'PyQt5'.")
                self.backend = _BackendPyQt5(scope)
        elif backend == 'pyqt5':
            self.backend = _BackendPyQt5(scope)
        else:
            logger.error(f"StreamViewer backend '{backend}' not supported."
                         "Supported: 'pyqt5', 'vispy'.")

        # Load UI
        self.scope = scope
        self.load_ui()

        # Load and set default configuration
        self.set_init_configuration(
            str(Path(__file__).parent/'.scope_settings_eeg.ini'))

        # Init backend
        self.init_backend()

    def load_ui(self):
        """
        Loads the UI created with QtCreator.
        """
        # Load
        self.ui = UI_MainWindow(self)
        self.connect_signals_to_slots()

        # Fill additional informations
        self._fill_list_signal_y_scale(self.scope.signal_y_scales.keys())
        self._fill_table_channels()

        # Set position on the screen
        self.setGeometry(100, 100,
                         self.geometry().width(),
                         self.geometry().height())
        self.setFixedSize(self.geometry().width(),
                          self.geometry().height())

        # Display
        self.show()

    def _fill_list_signal_y_scale(self, y_scales, init_idx=0):
        """
        Fill the drop-down menu to select the signal range.

        Parameters
        ----------
        y_scales : list of str
            The list of items to place in the drop-down menu.
        init_idx : int
            The default selected item in the drop-down menu.
        """
        for y_scale in y_scales:
            self.ui.comboBox_signal_y_scale.addItem(str(y_scale))
            self.ui.comboBox_signal_y_scale.setCurrentIndex(init_idx)

    def _fill_table_channels(self):
        """
        Fill the table widget with the channel names.
        """
        self._nb_table_columns = 8 if self.scope.n_channels > 64 else 4
        self._nb_table_rows = math.ceil(
            self.scope.n_channels / self._nb_table_columns)
        self.ui.table_channels.setRowCount(self._nb_table_rows)
        self.ui.table_channels.setColumnCount(self._nb_table_columns)

        for idx, ch in enumerate(range(self.scope.n_channels)):
            row = idx // self._nb_table_columns
            col = idx % self._nb_table_columns

            self.ui.table_channels.setItem(
                row, col, QTableWidgetItem(idx))
            self.ui.table_channels.item(row, col).setTextAlignment(
                QtCore.Qt.AlignCenter)
            self.ui.table_channels.item(row, col).setText(
                self.scope.channels_labels[idx])
            self.ui.table_channels.item(row, col).setSelected(True)

        self.ui.table_channels.verticalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.ui.table_channels.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

    def set_init_configuration(self, file):
        """
        Load and set a default configuration for the GUI.
        """
        path2_viewerFolder = Path(__file__).parent
        self.scope_settings = RawConfigParser(
            allow_no_value=True, inline_comment_prefixes=('#', ';'))
        self.scope_settings.read(str(path2_viewerFolder/file))

        # Y-Scale (amplitude)
        try:
            scale = float(self.scope_settings.get("plot", "signal_y_scale"))
            init_idx = list(self.scope.signal_y_scales.values()).index(scale)
            self.ui.comboBox_signal_y_scale.setCurrentIndex(init_idx)
        except:
            self.ui.comboBox_signal_y_scale.setCurrentIndex(0)

        # X-Scale (time)
        try:
            self.ui.spinBox_signal_x_scale.setValue(int(
                self.scope_settings.get("plot", "time_plot")))
        except:
            self.ui.spinBox_signal_x_scale.setValue(10)  # 10s by default

        # BP/CAR Filters
        try:
            self.ui.checkBox_car.setChecked(bool(
                self.scope_settings.get("filtering", "apply_car_filter")))
        except:
            self.ui.checkBox_car.setChecked(False)

        try:
            self.ui.checkBox_bandpass.setChecked(bool(
                self.scope_settings.get("filtering", "apply_bandpass_filter")))
        except:
            self.ui.checkBox_bandpass.setChecked(False)

        try:
            self.ui.doubleSpinBox_bandpass_low.setValue(float(
                self.scope_settings.get(
                    "filtering", "bandpass_cutoff_frequency").split(' ')[0]))
            self.ui.doubleSpinBox_bandpass_high.setValue(float(
                self.scope_settings.get(
                    "filtering", "bandpass_cutoff_frequency").split(' ')[1]))
        except:
            # Default to [1, 40] Hz.
            self.ui.doubleSpinBox_bandpass_low.setValue(1.)
            self.ui.doubleSpinBox_bandpass_high.setValue(40.)

        self.ui.doubleSpinBox_bandpass_high.setMinimum(
            self.ui.doubleSpinBox_bandpass_low.value()+1)
        self.ui.doubleSpinBox_bandpass_low.setMaximum(
            self.ui.doubleSpinBox_bandpass_high.value()-1)
        self.scope.init_bandpass_filter(
            low=self.ui.doubleSpinBox_bandpass_low.value(),
            high=self.ui.doubleSpinBox_bandpass_high.value())

        # Events
        try:
            self.ui.checkBox_show_LPT_events.setChecked(
                bool(self.scope_settings.get("plot", "show_LPT_events")))
        except:
            self.ui.checkBox_show_LPT_events.setChecked(False)

        self.ui.statusBar.showMessage("[Not recording]")

    def init_backend(self):
        """
        Initialize the backend.

        When calling .load_ui(), .connect_signals_to_slots() is called which
        required both a scope and a backend to be defined.
        However, the backend requires the _ScopeControllerUI to be fully
        initialized and setup. Thus the backend is initialized in 2 steps:
            - Creation of the instance (with all the associated methods)
            - Initialization (with all the associated variables)
        """
        x_scale = self.ui.spinBox_signal_x_scale.value()
        y_scale_idx = self.ui.comboBox_signal_y_scale.currentIndex()
        y_scale = list(self.scope.signal_y_scales.values())[y_scale_idx]
        geometry = (self.geometry().x() + self.width(), self.geometry().y(),
                    self.width() * 2, self.height())
        self.backend.init_backend(geometry, x_scale, y_scale, self.channels_to_show_idx)
        self.backend.start_timer()


    # -------------------------------------------------------------------
    # --------------------------- EVENT HANDLERS ------------------------
    # -------------------------------------------------------------------
    def connect_signals_to_slots(self):
        """
        Event handler. Connect QT signals to slots.
        """
        # Scales
        self.ui.comboBox_signal_y_scale.activated.connect(
            self.onActivated_comboBox_signal_y_scale)
        self.ui.spinBox_signal_x_scale.valueChanged.connect(
            self.onValueChanged_spinBox_signal_x_scale)

        # CAR / Filters
        self.ui.checkBox_car.stateChanged.connect(
            self.onClicked_checkBox_car)
        self.ui.checkBox_bandpass.stateChanged.connect(
            self.onClicked_checkBox_bandpass)
        self.ui.doubleSpinBox_bandpass_low.valueChanged.connect(
            self.onValueChanged_doubleSpinBox_bandpass_low)
        self.ui.doubleSpinBox_bandpass_high.valueChanged.connect(
            self.onValueChanged_doubleSpinBox_bandpass_high)

        # Events
        self.ui.checkBox_show_LPT_events.stateChanged.connect(
            self.onClicked_checkBox_show_LPT_events)

        # Recording
        self.ui.pushButton_start_recording.clicked.connect(
            self.onClicked_pushButton_start_recording)
        self.ui.pushButton_stop_recording.clicked.connect(
            self.onClicked_pushButton_stop_recording)
        self.ui.pushButton_set_recording_dir.clicked.connect(
            self.onClicked_pushButton_set_recording_dir)

        # Channel table
        self.ui.table_channels.itemSelectionChanged.connect(
            self.onSelectionChanged_table_channels)

    @QtCore.pyqtSlot()
    def onActivated_comboBox_signal_y_scale(self):
        y_scale = list(self.scope.signal_y_scales.values())[
            self.ui.comboBox_signal_y_scale.currentIndex()]
        self.backend.update_y_scale(float(y_scale))

    @QtCore.pyqtSlot()
    def onValueChanged_spinBox_signal_x_scale(self):
        self.backend.update_x_scale(self.ui.spinBox_signal_x_scale.value())

    @QtCore.pyqtSlot()
    def onClicked_checkBox_car(self):
        self.scope._apply_car = bool(
            self.ui.checkBox_car.isChecked())

    @QtCore.pyqtSlot()
    def onClicked_checkBox_bandpass(self):
        self.scope._apply_bandpass = bool(
            self.ui.checkBox_bandpass.isChecked())

    @QtCore.pyqtSlot()
    def onValueChanged_doubleSpinBox_bandpass_low(self):
        self.ui.doubleSpinBox_bandpass_high.setMinimum(
            self.ui.doubleSpinBox_bandpass_low.value()+1)
        self.scope.init_bandpass_filter(
            low=self.ui.doubleSpinBox_bandpass_low.value(),
            high=self.ui.doubleSpinBox_bandpass_high.value())

    @QtCore.pyqtSlot()
    def onValueChanged_doubleSpinBox_bandpass_high(self):
        self.ui.doubleSpinBox_bandpass_low.setMaximum(
            self.ui.doubleSpinBox_bandpass_high.value()-1)
        self.scope.init_bandpass_filter(
            low=self.ui.doubleSpinBox_bandpass_low.value(),
            high=self.ui.doubleSpinBox_bandpass_high.value())

    @QtCore.pyqtSlot()
    def onClicked_checkBox_show_LPT_events(self):
        self.backend._show_LPT_events = bool(
            self.ui.checkBox_show_LPT_events.isChecked())
        self.backend.update_show_LPT_events()

    @QtCore.pyqtSlot()
    def onClicked_pushButton_start_recording(self):
        record_dir = self.ui.lineEdit_recording_dir.text()
        self.recorder = StreamRecorder(record_dir, logger)
        self.recorder.start(stream_name=self.scope.stream_name, verbose=False)
        self.ui.pushButton_stop_recording.setEnabled(True)
        self.ui.pushButton_start_recording.setEnabled(False)
        self.ui.statusBar.showMessage(f"[Recording to '{record_dir}']")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_stop_recording(self):
        self.recorder.stop()
        self.ui.pushButton_start_recording.setEnabled(True)
        self.ui.pushButton_stop_recording.setEnabled(False)
        self.ui.statusBar.showMessage("[Not recording]")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_set_recording_dir(self):
        defaultPath = str(Path.home())
        path_name = QFileDialog.getExistingDirectory(
            caption="Choose the recording directory", directory=defaultPath)

        if path_name:
            self.ui.lineEdit_recording_dir.setText(path_name)
            self.ui.pushButton_start_recording.setEnabled(True)

    @QtCore.pyqtSlot()
    def onSelectionChanged_table_channels(self):
        selected = self.ui.table_channels.selectedItems()
        self.channels_to_show_idx = [
            item.row()*self._nb_table_columns + item.column()
            for item in selected]

        self.scope.channels_to_show_idx = self.channels_to_show_idx
        self.backend.update_channels_to_show_idx(self.channels_to_show_idx)

    def closeEvent(self, event):
        """
        Event called when closing the _ScopeControllerUI window.
        """
        if self.ui.pushButton_stop_recording.isEnabled():
            self.onClicked_pushButton_stop_recording()
        self.backend.close()
        event.accept()
