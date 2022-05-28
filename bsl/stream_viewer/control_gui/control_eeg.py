import math
from configparser import RawConfigParser
from pathlib import Path

from PyQt5 import QtCore
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem

from ...utils._docs import copy_doc, fill_doc
from ...utils._logs import logger
from ..backends.pyqtgraph import _BackendPyQtGraph
from ._control import _ControlGUI
from ._ui_control import UI_MainWindow


@fill_doc
class ControlGUI_EEG(_ControlGUI):
    """Controller GUI for EEG LSL Stream.

    Parameters
    ----------
    %(viewer_scope)s
    """

    def __init__(self, scope):
        super().__init__(scope)
        config_file = "settings_scope_eeg.ini"

        self._load_configuration(config_file)
        self._load_gui()
        self._init_backend()
        self._connect_signals_to_slots()
        self._set_configuration(config_file)

        self._backend.start_timer()

    def _load_gui(self):
        """Load the UI created with QtCreator."""
        logger.debug("Loading GUI..")

        self._ui = UI_MainWindow(self)

        for yRange in self._yRanges:
            self._ui.comboBox_signal_yRange.addItem(str(yRange))
            logger.debug("y-scale option %s added.", yRange)

        # Set table channels row/col
        self._nb_table_columns = 8 if self._scope.nb_channels > 64 else 4
        self._nb_table_rows = math.ceil(
            self._scope.nb_channels / self._nb_table_columns
        )
        self._ui.table_channels.setRowCount(self._nb_table_rows)
        self._ui.table_channels.setColumnCount(self._nb_table_columns)

        logger.debug(
            "Channel table set to %d row and %d col.",
            self._nb_table_rows,
            self._nb_table_columns,
        )

        # Set table channels elements
        for idx in range(self._scope.nb_channels):
            row = idx // self._nb_table_columns
            col = idx % self._nb_table_columns
            self._ui.table_channels.setItem(row, col, QTableWidgetItem(idx))
            self._ui.table_channels.item(row, col).setTextAlignment(
                QtCore.Qt.AlignCenter
            )
            self._ui.table_channels.item(row, col).setText(
                self._scope.channels_labels[idx]
            )
            logger.debug("Added channel %s", self._scope.channels_labels[idx])

        # Table channels header
        self._ui.table_channels.verticalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._ui.table_channels.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        # Set position on the screen
        self.setGeometry(
            100, 100, self.geometry().width(), self.geometry().height()
        )
        self.setFixedSize(self.geometry().width(), self.geometry().height())
        self.show()  # Display

        logger.debug("Loading GUI complete.")

    def _load_configuration(self, file):
        """Load default configuration for the ranges."""
        logger.debug("Loading configuration..")

        path2settings_folder = Path(__file__).parent / "settings"
        logger.debug("Configuration folder is '%s'.", path2settings_folder)
        logger.debug("Configuration file is '%s'.", file)
        scope_settings = RawConfigParser(
            allow_no_value=True, inline_comment_prefixes=("#", ";")
        )
        scope_settings.read(str(path2settings_folder / file))

        # yRange (amplitude)
        self._yRanges = {
            "1uV": 1,
            "10uV": 10,
            "25uV": 25,
            "50uV": 50,
            "100uV": 100,
            "250uV": 250,
            "500uV": 500,
            "1mV": 1000,
            "2.5mV": 2500,
            "10mV": 10000,
        }
        try:
            self._yRange = float(scope_settings.get("plot", "yRange"))
            if self._yRange not in self._yRanges.values:
                logger.debug("yRange %s is not in valid ranges.", self._yRange)
                self._yRange = 25.0
        except Exception:  # Default to 25 uV
            self._yRange = 25.0

        # xRange (time)
        try:
            self._xRange = int(scope_settings.get("plot", "xRange"))
        except Exception:  # Default to 10s
            logger.debug("xRange value could not be converted to integer.")
            self._xRange = 10

        logger.debug("Loading configuration complete.")

    def _set_configuration(self, file):
        """Load and set a default configuration for the GUI."""
        path2settings_folder = Path(__file__).parent / "settings"
        scope_settings = RawConfigParser(
            allow_no_value=True, inline_comment_prefixes=("#", ";")
        )
        scope_settings.read(str(path2settings_folder / file))

        # yRange
        init_idx = list(self._yRanges.values()).index(self._yRange)
        self._ui.comboBox_signal_yRange.setCurrentIndex(init_idx)

        # xRange
        try:
            self._ui.spinBox_signal_xRange.setValue(self._xRange)
        except Exception:  # 10s by default
            self._ui.spinBox_signal_xRange.setValue(10)

        # CAR
        try:
            self._ui.checkBox_car.setChecked(
                bool(scope_settings.get("filtering", "apply_car_filter"))
            )
        except Exception:
            self._ui.checkBox_car.setChecked(False)

        # BP Filters
        self._ui.checkBox_bandpass.setChecked(True)
        self._ui.checkBox_bandpass.setEnabled(False)
        try:
            self._ui.doubleSpinBox_bandpass_low.setValue(
                float(
                    scope_settings.get(
                        "filtering", "bandpass_cutoff_frequency"
                    ).split(" ")[0]
                )
            )
            self._ui.doubleSpinBox_bandpass_high.setValue(
                float(
                    scope_settings.get(
                        "filtering", "bandpass_cutoff_frequency"
                    ).split(" ")[1]
                )
            )
        except Exception:
            # Default to [1, 40] Hz.
            self._ui.doubleSpinBox_bandpass_low.setValue(1.0)
            self._ui.doubleSpinBox_bandpass_high.setValue(40.0)

        self._ui.doubleSpinBox_bandpass_high.setMinimum(
            self._ui.doubleSpinBox_bandpass_low.value() + 1
        )
        self._ui.doubleSpinBox_bandpass_low.setMaximum(
            self._ui.doubleSpinBox_bandpass_high.value() - 1
        )
        self._scope.init_bandpass_filter(
            low=self._ui.doubleSpinBox_bandpass_low.value(),
            high=self._ui.doubleSpinBox_bandpass_high.value(),
        )

        # Trigger events
        try:
            self._ui.checkBox_show_LPT_trigger_events.setChecked(
                bool(scope_settings.get("plot", "show_LPT_events"))
            )
        except Exception:
            self._ui.checkBox_show_LPT_trigger_events.setChecked(False)

        # Table channels
        for idx in range(self._scope.nb_channels):
            row = idx // self._nb_table_columns
            col = idx % self._nb_table_columns
            self._ui.table_channels.item(row, col).setSelected(True)

        # Status bar
        self._ui.statusBar.showMessage("[Not recording]")

    @copy_doc(_ControlGUI._init_backend)
    def _init_backend(self):
        geometry = (
            self.geometry().x() + self.width(),
            self.geometry().y(),
            self.width() * 2,
            self.height(),
        )
        self._backend = _BackendPyQtGraph(
            self._scope, geometry, self._xRange, self._yRange
        )

    # --------------------------------------------------------------------
    @copy_doc(_ControlGUI._connect_signals_to_slots)
    def _connect_signals_to_slots(self):
        super()._connect_signals_to_slots()
        # Scales
        self._ui.comboBox_signal_yRange.activated.connect(
            self.onActivated_comboBox_signal_yRange
        )
        self._ui.spinBox_signal_xRange.valueChanged.connect(
            self.onValueChanged_spinBox_signal_xRange
        )

        # CAR / Filters
        self._ui.checkBox_car.stateChanged.connect(self.onClicked_checkBox_car)
        self._ui.checkBox_bandpass.stateChanged.connect(
            self.onClicked_checkBox_bandpass
        )
        self._ui.doubleSpinBox_bandpass_low.valueChanged.connect(
            self.onValueChanged_doubleSpinBox_bandpass_low
        )
        self._ui.doubleSpinBox_bandpass_high.valueChanged.connect(
            self.onValueChanged_doubleSpinBox_bandpass_high
        )

        # Trigger events
        self._ui.checkBox_show_LPT_trigger_events.stateChanged.connect(
            self.onClicked_checkBox_show_LPT_trigger_events
        )

        # Channel table
        self._ui.table_channels.itemSelectionChanged.connect(
            self.onSelectionChanged_table_channels
        )

    @QtCore.pyqtSlot()
    def onActivated_comboBox_signal_yRange(self):
        logger.debug("yRange event received.")
        self._yRange = float(
            list(self._yRanges.values())[
                self._ui.comboBox_signal_yRange.currentIndex()
            ]
        )
        self._backend.yRange = self._yRange
        logger.debug("y-range set to %d", self._yRange)

    @QtCore.pyqtSlot()
    def onValueChanged_spinBox_signal_xRange(self):
        logger.debug("xRange event received.")
        self._xRange = int(self._ui.spinBox_signal_xRange.value())
        self._backend.xRange = self._xRange
        logger.debug("x-range set to %d", self._xRange)

    @QtCore.pyqtSlot()
    def onClicked_checkBox_car(self):
        logger.debug("Checkbox for CAR event received.")
        self._scope.apply_car = self._ui.checkBox_car.isChecked()
        logger.debug("CAR checkbox: %s", self._ui.checkBox_car.isChecked())

    @QtCore.pyqtSlot()
    def onClicked_checkBox_bandpass(self):
        logger.debug("Checkbox for BP event received.")
        self._scope.apply_bandpass = self._ui.checkBox_bandpass.isChecked()
        logger.debug("BP checkbox: %s", self._ui.checkBox_bandpass.isChecked())

    @QtCore.pyqtSlot()
    def onValueChanged_doubleSpinBox_bandpass_low(self):
        logger.debug("BP-low event received.")
        self._ui.doubleSpinBox_bandpass_high.setMinimum(
            self._ui.doubleSpinBox_bandpass_low.value() + 1
        )
        self._scope.init_bandpass_filter(
            low=self._ui.doubleSpinBox_bandpass_low.value(),
            high=self._ui.doubleSpinBox_bandpass_high.value(),
        )
        logger.debug(
            "BP set to [%d, %d]",
            self._ui.doubleSpinBox_bandpass_low.value(),
            self._ui.doubleSpinBox_bandpass_high.value(),
        )

    @QtCore.pyqtSlot()
    def onValueChanged_doubleSpinBox_bandpass_high(self):
        logger.debug("BP-high event received.")
        self._ui.doubleSpinBox_bandpass_low.setMaximum(
            self._ui.doubleSpinBox_bandpass_high.value() - 1
        )
        self._scope.init_bandpass_filter(
            low=self._ui.doubleSpinBox_bandpass_low.value(),
            high=self._ui.doubleSpinBox_bandpass_high.value(),
        )
        logger.debug(
            "BP set to [%d, %d]",
            self._ui.doubleSpinBox_bandpass_low.value(),
            self._ui.doubleSpinBox_bandpass_high.value(),
        )

    @QtCore.pyqtSlot()
    def onClicked_checkBox_show_LPT_trigger_events(self):
        logger.debug("Checkbox for LPT event received.")
        self._backend.show_LPT_trigger_events = bool(
            self._ui.checkBox_show_LPT_trigger_events.isChecked()
        )
        logger.debug(
            "LPT trigger checkbox: %s",
            self._ui.checkBox_show_LPT_trigger_events.isChecked(),
        )

    @QtCore.pyqtSlot()
    def onSelectionChanged_table_channels(self):
        logger.debug("Channel selection event received.")
        selected = self._ui.table_channels.selectedItems()
        self._scope.selected_channels = sorted(
            [
                item.row() * self._nb_table_columns + item.column()
                for item in selected
            ]
        )
        self._backend.selected_channels = self._scope.selected_channels

    def closeEvent(self, event):
        """Event called when closing the _ScopeControllerUI window."""
        logger.debug("Closing event received.")
        if self._ui.pushButton_stop_recording.isEnabled():
            self.onClicked_pushButton_stop_recording()
        self._backend.close()
        super().closeEvent(event)

    # --------------------------------------------------------------------
    @property
    @copy_doc(_ControlGUI.backend)
    def backend(self):
        return self._backend

    @property
    @copy_doc(_ControlGUI.ui)
    def ui(self):
        return self._ui

    @property
    def xRange(self):
        """Selected X range (time) [seconds]."""
        return self._xRange

    @property
    def yRange(self):
        """Selected Y range (amplitude) [uV]."""
        return self._yRange
