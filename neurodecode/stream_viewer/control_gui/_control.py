from pathlib import Path
from abc import ABC, abstractmethod

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog

try:
    from ..backends import _BackendVispy
except ImportError:
    pass
from ..backends import _BackendPyQt5

from ... import logger
from ...stream_recorder import StreamRecorder


class _metaclass_ControlGUI(type(QMainWindow), type(ABC)):
    pass


class _ControlGUI(QMainWindow, ABC, metaclass=_metaclass_ControlGUI):
    """
    Class representing a base controller GUI.

    Parameters
    ----------
    scope : stream_viewer.scope._scope._Scope
        Scope connected to a stream receiver acquiring the data and applying
        filtering. The scope has a buffer of _scope._BUFFER_DURATION.
    backend : str
        One of the supported backend's name. Supported 'vispy', 'pyqt5'.
    """

    @abstractmethod
    def __init__(self, scope, backend):
        super().__init__()
        self._scope = scope

    @abstractmethod
    def _load_gui(self):
        """
        Load the GUI.
        """
        pass

    @abstractmethod
    def _init_backend(self, backend):
        """
        Initialize the backend.
        """
        pass

    def closeEvent(self, event):
        """
        Event called when closing the GUI.
        """
        event.accept()

    # --------------------------------------------------------------------
    @abstractmethod
    def _connect_signals_to_slots(self):
        """
        Event handler. Connect QT signals to slots.
        """
        # Recording
        self._ui.pushButton_start_recording.clicked.connect(
            self.onClicked_pushButton_start_recording)
        self._ui.pushButton_stop_recording.clicked.connect(
            self.onClicked_pushButton_stop_recording)
        self._ui.pushButton_set_recording_dir.clicked.connect(
            self.onClicked_pushButton_set_recording_dir)

    @QtCore.pyqtSlot()
    def onClicked_pushButton_start_recording(self):
        record_dir = self._ui.lineEdit_recording_dir.text()
        self._recorder = StreamRecorder(
            record_dir, stream_name=self._scope.stream_name)
        self._recorder.start(fif_subdir=True, blocking=False, verbose=False)
        self._ui.pushButton_stop_recording.setEnabled(True)
        self._ui.pushButton_start_recording.setEnabled(False)
        self._ui.statusBar.showMessage(f"[Recording to '{record_dir}']")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_stop_recording(self):
        if self._recorder.state.value == 1:
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
        Measuring scope.
        """
        return self._scope

    @property
    @abstractmethod
    def backend(self):
        """
        Display backend.
        """
        pass

    @property
    @abstractmethod
    def ui(self):
        """
        Control UI.
        """
        pass
