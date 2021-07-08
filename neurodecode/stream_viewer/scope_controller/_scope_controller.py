import math
from pathlib import Path
from configparser import RawConfigParser

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QMainWindow, QHeaderView,
                             QTableWidgetItem, QFileDialog)

try:
    from ..backends import _BackendVispy
except ImportError:
    pass
from ..backends import _BackendPyQt5
from ._ui_scope_controller import UI_MainWindow
from ...stream_recorder import StreamRecorder
from ... import logger


class _ScopeControllerUI(QMainWindow):
    """
    GUI controlling the scope and the backend.

    Parameters
    ----------
    scope : neurodecode.stream_viewer._scope._Scope
        The scope connected to a stream receiver acquiring the data and
        applying filtering. The scope has a buffer of _scope._BUFFER_DURATION
        (default: 30s).
    backend : str
        One of the supported backend's name. Supported 'vispy', 'pyqt5'.
        If None, load the first backend in the order: Vispy, PyQt5.
    """

    def __init__(self, scope, backend='pyqt5'):
        super().__init__()
        backend = _ScopeControllerUI._check_backend(backend)
        self._backend = backend(scope)

        # Load UI
        self._scope = scope
        self._load_ui()

        # Load and set default configuration
        self._set_init_configuration('settings_scope_eeg.ini')

        # Init backend
        self._init_backend()

    def _load_ui(self):
        """
        Loads the UI created with QtCreator.
        """
        # Load
        self._ui = UI_MainWindow(self)
        self._connect_signals_to_slots()

        # Fill additional informations
        self._fill_list_signal_y_scale(self._scope.signal_y_scales)
        self._fill_table_channels()

        # Set position on the screen
        self.setGeometry(100, 100,
                         self.geometry().width(),
                         self.geometry().height())
        self.setFixedSize(self.geometry().width(),
                          self.geometry().height())

        # Display
        self.show()

    def _fill_list_signal_y_scale(self, y_scales):
        """
        Fill the drop-down menu to select the signal range.

        Parameters
        ----------
        y_scales : list of str
            The list of items to place in the drop-down menu.
        """
        for y_scale in y_scales:
            self._ui.comboBox_signal_y_scale.addItem(str(y_scale))

    def _fill_table_channels(self):
        """
        Fill the table widget with the channel names.
        """
        self._nb_table_columns = 8 if self._scope.n_channels > 64 else 4
        self._nb_table_rows = math.ceil(
            self._scope.n_channels / self._nb_table_columns)
        self._ui.table_channels.setRowCount(self._nb_table_rows)
        self._ui.table_channels.setColumnCount(self._nb_table_columns)

        for idx in range(self._scope.n_channels):
            row = idx // self._nb_table_columns
            col = idx % self._nb_table_columns

            self._ui.table_channels.setItem(
                row, col, QTableWidgetItem(idx))
            self._ui.table_channels.item(row, col).setTextAlignment(
                QtCore.Qt.AlignCenter)
            self._ui.table_channels.item(row, col).setText(
                self._scope.channels_labels[idx])
            self._ui.table_channels.item(row, col).setSelected(True)

        self._ui.table_channels.verticalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._ui.table_channels.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

    def _set_init_configuration(self, file):
        """
        Load and set a default configuration for the GUI.
        """
        path2settings_folder = Path(__file__).parent / 'settings'
        scope_settings = RawConfigParser(
            allow_no_value=True, inline_comment_prefixes=('#', ';'))
        scope_settings.read(str(path2settings_folder/file))

        # Y-Scale (amplitude)
        try:
            scale = float(scope_settings.get("plot", "signal_y_scale"))
            init_idx = list(self._scope.signal_y_scales.values()).index(scale)
            self._ui.comboBox_signal_y_scale.setCurrentIndex(init_idx)
        except Exception:
            self._ui.comboBox_signal_y_scale.setCurrentIndex(0)

        # X-Scale (time)
        try:
            self._ui.spinBox_signal_x_scale.setValue(int(
                scope_settings.get("plot", "time_plot")))
        except Exception:
            self._ui.spinBox_signal_x_scale.setValue(10)  # 10s by default

        # BP/CAR Filters
        try:
            self._ui.checkBox_car.setChecked(bool(
                scope_settings.get("filtering", "apply_car_filter")))
        except Exception:
            self._ui.checkBox_car.setChecked(False)

        try:
            self._ui.checkBox_bandpass.setChecked(bool(
                scope_settings.get("filtering", "apply_bandpass_filter")))
        except Exception:
            self._ui.checkBox_bandpass.setChecked(False)

        try:
            self._ui.doubleSpinBox_bandpass_low.setValue(float(
                scope_settings.get(
                    "filtering", "bandpass_cutoff_frequency").split(' ')[0]))
            self._ui.doubleSpinBox_bandpass_high.setValue(float(
                scope_settings.get(
                    "filtering", "bandpass_cutoff_frequency").split(' ')[1]))
        except Exception:
            # Default to [1, 40] Hz.
            self._ui.doubleSpinBox_bandpass_low.setValue(1.)
            self._ui.doubleSpinBox_bandpass_high.setValue(40.)

        self._ui.doubleSpinBox_bandpass_high.setMinimum(
            self._ui.doubleSpinBox_bandpass_low.value()+1)
        self._ui.doubleSpinBox_bandpass_low.setMaximum(
            self._ui.doubleSpinBox_bandpass_high.value()-1)
        self._scope.init_bandpass_filter(
            low=self._ui.doubleSpinBox_bandpass_low.value(),
            high=self._ui.doubleSpinBox_bandpass_high.value())

        # Events
        try:
            self._ui.checkBox_show_LPT_events.setChecked(
                bool(scope_settings.get("plot", "show_LPT_events")))
        except Exception:
            self._ui.checkBox_show_LPT_events.setChecked(False)

        self._ui.statusBar.showMessage("[Not recording]")

    def _init_backend(self):
        """
        Initialize the backend.

        When calling .load_ui(), .connect_signals_to_slots() is called which
        required both a scope and a backend to be defined.
        However, the backend requires the _ScopeControllerUI to be fully
        initialized and setup. Thus the backend is initialized in 2 steps:
            - Creation of the instance (with all the associated methods)
            - Initialization (with all the associated variables)
        """
        x_scale = self._ui.spinBox_signal_x_scale.value()
        y_scale_idx = self._ui.comboBox_signal_y_scale.currentIndex()
        y_scale = list(self._scope.signal_y_scales.values())[y_scale_idx]
        geometry = (self.geometry().x() + self.width(), self.geometry().y(),
                    self.width() * 2, self.height())
        self._backend.init_backend(
            geometry, x_scale, y_scale, self.channels_to_show_idx)
        self._backend.start_timer()

    # -------------------------------------------------------------------
    # --------------------------- EVENT HANDLERS ------------------------
    # -------------------------------------------------------------------
    def _connect_signals_to_slots(self):
        """
        Event handler. Connect QT signals to slots.
        """
        # Scales
        self._ui.comboBox_signal_y_scale.activated.connect(
            self.onActivated_comboBox_signal_y_scale)
        self._ui.spinBox_signal_x_scale.valueChanged.connect(
            self.onValueChanged_spinBox_signal_x_scale)

        # CAR / Filters
        self._ui.checkBox_car.stateChanged.connect(
            self.onClicked_checkBox_car)
        self._ui.checkBox_bandpass.stateChanged.connect(
            self.onClicked_checkBox_bandpass)
        self._ui.doubleSpinBox_bandpass_low.valueChanged.connect(
            self.onValueChanged_doubleSpinBox_bandpass_low)
        self._ui.doubleSpinBox_bandpass_high.valueChanged.connect(
            self.onValueChanged_doubleSpinBox_bandpass_high)

        # Events
        self._ui.checkBox_show_LPT_events.stateChanged.connect(
            self.onClicked_checkBox_show_LPT_events)

        # Recording
        self._ui.pushButton_start_recording.clicked.connect(
            self.onClicked_pushButton_start_recording)
        self._ui.pushButton_stop_recording.clicked.connect(
            self.onClicked_pushButton_stop_recording)
        self._ui.pushButton_set_recording_dir.clicked.connect(
            self.onClicked_pushButton_set_recording_dir)

        # Channel table
        self._ui.table_channels.itemSelectionChanged.connect(
            self.onSelectionChanged_table_channels)

    @QtCore.pyqtSlot()
    def onActivated_comboBox_signal_y_scale(self):
        y_scale = list(self._scope.signal_y_scales.values())[
            self._ui.comboBox_signal_y_scale.currentIndex()]
        self._backend.y_scale = float(y_scale)

    @QtCore.pyqtSlot()
    def onValueChanged_spinBox_signal_x_scale(self):
        self._backend.x_scale = self._ui.spinBox_signal_x_scale.value()

    @QtCore.pyqtSlot()
    def onClicked_checkBox_car(self):
        self._scope.apply_car = self._ui.checkBox_car.isChecked()

    @QtCore.pyqtSlot()
    def onClicked_checkBox_bandpass(self):
        self._scope.apply_bandpass = self._ui.checkBox_bandpass.isChecked()

    @QtCore.pyqtSlot()
    def onValueChanged_doubleSpinBox_bandpass_low(self):
        self._ui.doubleSpinBox_bandpass_high.setMinimum(
            self._ui.doubleSpinBox_bandpass_low.value()+1)
        self._scope.init_bandpass_filter(
            low=self._ui.doubleSpinBox_bandpass_low.value(),
            high=self._ui.doubleSpinBox_bandpass_high.value())

    @QtCore.pyqtSlot()
    def onValueChanged_doubleSpinBox_bandpass_high(self):
        self._ui.doubleSpinBox_bandpass_low.setMaximum(
            self._ui.doubleSpinBox_bandpass_high.value()-1)
        self._scope.init_bandpass_filter(
            low=self._ui.doubleSpinBox_bandpass_low.value(),
            high=self._ui.doubleSpinBox_bandpass_high.value())

    @QtCore.pyqtSlot()
    def onClicked_checkBox_show_LPT_events(self):
        self._backend.show_LPT_events = bool(
            self._ui.checkBox_show_LPT_events.isChecked())

    @QtCore.pyqtSlot()
    def onClicked_pushButton_start_recording(self):
        record_dir = self._ui.lineEdit_recording_dir.text()
        self._recorder = StreamRecorder(
            record_dir, stream_name=self._scope.stream_name)
        self._recorder.start(fif_subdir=True, verbose=False)
        self._ui.pushButton_stop_recording.setEnabled(True)
        self._ui.pushButton_start_recording.setEnabled(False)
        self._ui.statusBar.showMessage(f"[Recording to '{record_dir}']")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_stop_recording(self):
        self._recorder.stop()
        self._ui.pushButton_start_recording.setEnabled(True)
        self._ui.pushButton_stop_recording.setEnabled(False)
        self._ui.statusBar.showMessage("[Not recording]")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_set_recording_dir(self):
        defaultPath = str(Path.home())
        path_name = QFileDialog.getExistingDirectory(
            caption="Choose the recording directory", directory=defaultPath)

        if path_name:
            self._ui.lineEdit_recording_dir.setText(path_name)
            self._ui.pushButton_start_recording.setEnabled(True)

    @QtCore.pyqtSlot()
    def onSelectionChanged_table_channels(self):
        selected = self._ui.table_channels.selectedItems()
        self.channels_to_show_idx = [
            item.row()*self._nb_table_columns + item.column()
            for item in selected]

        self._scope.channels_to_show_idx = self.channels_to_show_idx
        self._backend.channels_to_show_idx = self.channels_to_show_idx

    def closeEvent(self, event):
        """
        Event called when closing the _ScopeControllerUI window.
        """
        if self._ui.pushButton_stop_recording.isEnabled():
            self.onClicked_pushButton_stop_recording()
        self._backend.close()
        event.accept()

    # --------------------------------------------------------------------
    @staticmethod
    def _check_backend(backend):
        """
        Checks that the requested backend is supported.
        """
        SUPPORTED_BACKENDS = dict()
        try:
            SUPPORTED_BACKEND['vispy'] = _BackendVispy
        except NameError:
            pass
        try:
            SUPPORTED_BACKENDS['pyqt5'] = _BackendPyQt5
        except NameError:
            pass

        if len(SUPPORTED_BACKENDS) == 0:
            logger.error('The StreamViewer did not find an installed backend.')
            raise RuntimeError

        if isinstance(backend, str):
            backend = backend.lower().strip()
            if backend in SUPPORTED_BACKENDS:
                return SUPPORTED_BACKENDS[backend]
            else:
                logger.warning(
                    f"Selected backend '{backend}' is not supported. "
                    "Default to 'pyqt5'.")
                return SUPPORTED_BACKENDS['pyqt5']

        else:
            logger.warning(
                "Selected backend is not a string. Default to 'pyqt5'.")
            return SUPPORTED_BACKENDS['pyqt5']

    # --------------------------------------------------------------------
    @property
    def scope(self):
        """
        The measuring scope.
        """
        return self._scope

    @property
    def backend(self):
        """
        The display backend.
        """
        return self._backend

    @property
    def ui(self):
        """
        The control UI.
        """
        return self._ui
