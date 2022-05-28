from abc import ABC, abstractmethod
from pathlib import Path

from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from ...stream_recorder import StreamRecorder
from ...utils._docs import fill_doc
from ...utils._logs import logger


class _metaclass_ControlGUI(type(QMainWindow), type(ABC)):
    pass


@fill_doc
class _ControlGUI(QMainWindow, ABC, metaclass=_metaclass_ControlGUI):
    """Class representing a base controller GUI.

    Parameters
    ----------
    %(viewer_scope)s
    """

    @abstractmethod
    def __init__(self, scope):
        super().__init__()
        self._scope = scope

    @abstractmethod
    def _load_gui(self):
        """Load the GUI."""
        pass

    @abstractmethod
    def _init_backend(self):
        """Initialize the backend."""
        pass

    def closeEvent(self, event):
        """Event called when closing the GUI."""
        event.accept()

    # --------------------------------------------------------------------
    @abstractmethod
    def _connect_signals_to_slots(self):
        """Event handler. Connect QT signals to slots."""
        # Recording
        self._ui.pushButton_start_recording.clicked.connect(
            self.onClicked_pushButton_start_recording
        )
        self._ui.pushButton_stop_recording.clicked.connect(
            self.onClicked_pushButton_stop_recording
        )
        self._ui.pushButton_set_recording_dir.clicked.connect(
            self.onClicked_pushButton_set_recording_dir
        )

    @QtCore.pyqtSlot()
    def onClicked_pushButton_start_recording(self):
        logger.debug("Start recording event received.")
        record_dir = self._ui.lineEdit_recording_dir.text()
        self._recorder = StreamRecorder(
            record_dir,
            stream_name=self._scope.stream_name,
            fif_subdir=True,
            verbose=False,
        )
        self._recorder.start(blocking=False)
        self._ui.pushButton_stop_recording.setEnabled(True)
        self._ui.pushButton_start_recording.setEnabled(False)
        self._ui.statusBar.showMessage(f"[Recording to '{record_dir}']")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_stop_recording(self):
        logger.debug("Stop recording event received.")
        if self._recorder.state.value == 1:
            self._recorder.stop()
            self._ui.pushButton_start_recording.setEnabled(True)
            self._ui.pushButton_stop_recording.setEnabled(False)
            self._ui.statusBar.showMessage("[Not recording]")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_set_recording_dir(self):
        logger.debug("Set recording directory event received.")
        defaultPath = str(Path.home())
        path_name = QFileDialog.getExistingDirectory(
            caption="Choose the recording directory", directory=defaultPath
        )

        if path_name:
            self._ui.lineEdit_recording_dir.setText(path_name)
            self._ui.pushButton_start_recording.setEnabled(True)

    # --------------------------------------------------------------------
    @property
    def scope(self):
        """Measuring scope."""
        return self._scope

    @property
    @abstractmethod
    def backend(self):
        """Display backend."""
        pass

    @property
    @abstractmethod
    def ui(self):
        """Control UI."""
        pass
