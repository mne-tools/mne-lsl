from qtpy import QtCore, QtWidgets
from qtpy.QtCore import QRect, QSize
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QLabel,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QWidget,
)

from ..scope._scope import _BUFFER_DURATION


class UI_MainWindow(object):  # noqa
    def __init__(self, MainWindow):
        self.load_ui(MainWindow)

    def load_ui(self, MainWindow):  # noqa
        # Set Main window and main widget
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(531, 693)
        MainWindow.setWindowTitle("Scope settings")
        self.MainWidget = QWidget(MainWindow)
        self.MainWidget.setObjectName("MainWidget")
        MainWindow.setCentralWidget(self.MainWidget)

        # Y-Scale of the signal
        self.comboBox_signal_yRange = QComboBox(self.MainWidget)
        self.comboBox_signal_yRange.setGeometry(QRect(10, 20, 85, 27))
        self.comboBox_signal_yRange.setObjectName("comboBox_signal_yRange")

        self.label_signal_yRange = QLabel(self.MainWidget)
        self.label_signal_yRange.setGeometry(QRect(10, 0, 67, 17))
        self.label_signal_yRange.setObjectName("label_signal_yRange")
        self.label_signal_yRange.setText("Scale")

        self.label_signal_yRange_tip = QLabel(self.MainWidget)
        self.label_signal_yRange_tip.setGeometry(QRect(350, 70, 151, 40))
        self.label_signal_yRange_tip.setObjectName("label_signal_yRange_tip")
        self.label_signal_yRange_tip.setText(
            "Scroll on the main plot window\nto change the Y-scale!"
        )

        # X-Scale of the signal
        self.spinBox_signal_xRange = QSpinBox(self.MainWidget)  # Integers
        self.spinBox_signal_xRange.setGeometry(QRect(10, 80, 85, 27))
        self.spinBox_signal_xRange.setMinimum(1)
        self.spinBox_signal_xRange.setMaximum(int(_BUFFER_DURATION))
        self.spinBox_signal_xRange.setProperty("value", 10)  # Default 10s
        self.spinBox_signal_xRange.setObjectName("spinBox_signal_xRange")

        self.label_signal_xRange = QLabel(self.MainWidget)
        self.label_signal_xRange.setGeometry(QRect(10, 60, 67, 17))
        self.label_signal_xRange.setObjectName("label_signal_xRange")
        self.label_signal_xRange.setText("Time (s)")

        # BP Filter
        self.checkBox_bandpass = QCheckBox(self.MainWidget)
        self.checkBox_bandpass.setGeometry(QRect(120, 20, 141, 22))
        self.checkBox_bandpass.setObjectName("checkBox_bandpass")
        self.checkBox_bandpass.setText("Bandpass Filter")

        # Set bandpass 'LOW'
        self.doubleSpinBox_bandpass_low = QDoubleSpinBox(self.MainWidget)  # Floats
        self.doubleSpinBox_bandpass_low.setGeometry(QRect(253, 17, 69, 27))
        self.doubleSpinBox_bandpass_low.setMinimum(0.1)
        self.doubleSpinBox_bandpass_low.setMaximum(1000.0)
        self.doubleSpinBox_bandpass_low.setSingleStep(1.0)
        self.doubleSpinBox_bandpass_low.setProperty("value", 1.0)  # Default 1Hz
        self.doubleSpinBox_bandpass_low.setObjectName("doubleSpinBox_bandpass_low")

        # Set bandpass 'HIGH'
        self.doubleSpinBox_bandpass_high = QDoubleSpinBox(self.MainWidget)
        self.doubleSpinBox_bandpass_high.setGeometry(QRect(333, 17, 69, 27))
        self.doubleSpinBox_bandpass_high.setMinimum(1.0)
        self.doubleSpinBox_bandpass_high.setMaximum(1000.0)
        self.doubleSpinBox_bandpass_high.setSingleStep(1.0)
        self.doubleSpinBox_bandpass_high.setProperty("value", 40.0)  # Default 40Hz
        self.doubleSpinBox_bandpass_high.setObjectName("doubleSpinBox_bandpass_high")

        # CAR
        self.checkBox_car = QCheckBox(self.MainWidget)
        self.checkBox_car.setGeometry(QRect(120, 0, 60, 22))
        self.checkBox_car.setObjectName("checkBox_car")
        self.checkBox_car.setText("CAR")

        # Detrend
        self.checkBox_detrend = QCheckBox(self.MainWidget)
        self.checkBox_detrend.setGeometry(QRect(175, 0, 80, 22))
        self.checkBox_detrend.setObjectName("checkBox_detrend")
        self.checkBox_detrend.setText("Detrend")

        # Show LPT events
        self.checkBox_show_LPT_trigger_events = QCheckBox(self.MainWidget)
        self.checkBox_show_LPT_trigger_events.setGeometry(QRect(120, 80, 151, 22))
        self.checkBox_show_LPT_trigger_events.setObjectName(
            "checkBox_show_LPT_trigger_events"
        )
        self.checkBox_show_LPT_trigger_events.setText("Show LPT events")

        # Lines
        self.line1 = QFrame(self.MainWidget)
        self.line1.setGeometry(QRect(100, -1, 21, 133))
        self.line1.setFrameShape(QFrame.VLine)
        self.line1.setFrameShadow(QFrame.Sunken)
        self.line1.setObjectName("line1")

        self.line2 = QFrame(self.MainWidget)
        self.line2.setGeometry(QRect(110, 45, 421, 21))
        self.line2.setFrameShape(QFrame.HLine)
        self.line2.setFrameShadow(QFrame.Sunken)
        self.line2.setObjectName("line2")

        self.line3 = QFrame(self.MainWidget)
        self.line3.setGeometry(QRect(290, 57, 21, 75))
        self.line3.setFrameShape(QFrame.VLine)
        self.line3.setFrameShadow(QFrame.Sunken)
        self.line3.setObjectName("line3")

        # Table of channels
        self.table_channels = QTableWidget(self.MainWidget)
        self.table_channels.setGeometry(QRect(4, 131, 525, 503))
        self.table_channels.setMaximumSize(QSize(529, 16777215))
        self.table_channels.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_channels.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.table_channels.setShowGrid(False)
        self.table_channels.setObjectName("table_channels")
        self.table_channels.setColumnCount(0)
        self.table_channels.setRowCount(0)
        self.table_channels.horizontalHeader().setVisible(False)
        self.table_channels.horizontalHeader().setHighlightSections(False)
        self.table_channels.verticalHeader().setVisible(False)
        self.table_channels.verticalHeader().setHighlightSections(False)

        # Bottom status Bar
        self.statusBar = QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        # Connect
        # QtCore.QMetaObject.connectSlotsByName(MainWindow)
