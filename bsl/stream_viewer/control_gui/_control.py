from pathlib import Path
from abc import ABC, abstractmethod

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog


try:
    from ..backends.vispy import _BackendVispy
except ImportError:
    pass
from ..backends.pyqtgraph import _BackendPyQtGraph

from ...utils._logs import logger
from ...utils._docs import fill_doc
from ...stream_recorder import StreamRecorder


class _metaclass_ControlGUI(type(QMainWindow), type(ABC)):
    pass


@fill_doc
class _ControlGUI(QMainWindow, ABC, metaclass=_metaclass_ControlGUI):
    """
    Class representing a base controller GUI.

    Parameters
    ----------
    %(viewer_scope)s
    %(viewer_backend)s
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
            record_dir, stream_name=self._scope.stream_name, fif_subdir=True,
            verbose=False)
        self._recorder.start(blocking=False)
        while self._recorder._annotation_file is None:
            pass  # less blocking than blocking=True

        self._backend._recorder_annotation_file = \
            open(self._recorder._annotation_file, 'a')

        self._ui.pushButton_stop_recording.setEnabled(True)
        self._ui.pushButton_start_recording.setEnabled(False)
        self._ui.statusBar.showMessage(f"[Recording to '{record_dir}']")

    @QtCore.pyqtSlot()
    def onClicked_pushButton_stop_recording(self):
        if self._recorder.state.value == 1:
            self._backend._AnnotationQueue.join(0.5)
            self._backend._AnnotationThread.join(0.2)
            self._backend._recorder_annotation_file.close()
            self._backend._recorder_annotation_file = None
            self._recorder.stop()
            self._ui.pushButton_start_recording.setEnabled(True)
            self._ui.pushButton_stop_recording.setEnabled(False)
            self._ui.statusBar.showMessage("[Not recording]")
        self._ui.comboBox_label.setEnabled(False)

    @QtCore.pyqtSlot()
    def onClicked_pushButton_set_recording_dir(self):
        defaultPath = str(Path.home())
        path_name = QFileDialog.getExistingDirectory(
            caption="Choose the recording directory", directory=defaultPath)

        if path_name:
            self._ui.lineEdit_recording_dir.setText(path_name)
            self._ui.pushButton_start_recording.setEnabled(True)
            self._ui.comboBox_label.setEnabled(True)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_backend(backend):
        """
        Checks that the requested backend is supported.
        """
        DEFAULT = ['pyqtgraph', 'vispy']  # Default order
        SUPPORTED_BACKENDS = dict()
        try:
            SUPPORTED_BACKENDS['vispy'] = _BackendVispy
        except NameError:
            SUPPORTED_BACKENDS['vispy'] = None
        try:
            SUPPORTED_BACKENDS['pyqtgraph'] = _BackendPyQtGraph
        except NameError:
            SUPPORTED_BACKENDS['pyqtgraph'] = None
        assert set(DEFAULT) == set(SUPPORTED_BACKENDS)

        if all(backend is None for backend in SUPPORTED_BACKENDS.values()):
            logger.error('The StreamViewer did not find an installed backend.')
            raise RuntimeError

        if isinstance(backend, str):
            backend = backend.lower().strip()
            if backend in SUPPORTED_BACKENDS:
                if SUPPORTED_BACKENDS[backend] is None:
                    logger.warning(
                        f"Selected backend '{backend}' is not installed. "
                        f"Default to first backend in the order {DEFAULT}.")
                    for default_backend in DEFAULT:
                        if default_backend is not None:
                            return SUPPORTED_BACKENDS[default_backend]
                return SUPPORTED_BACKENDS[backend]
            else:
                logger.warning(
                    f"Selected backend '{backend}' is not supported. "
                    f"Default to first backend in the order {DEFAULT}.")
                for default_backend in DEFAULT:
                    if default_backend is not None:
                        return SUPPORTED_BACKENDS[default_backend]
        else:
            logger.warning(
                "Selected backend is not a string. "
                f"Default to first backend in the order {DEFAULT}.")
            for default_backend in DEFAULT:
                if default_backend is not None:
                    return SUPPORTED_BACKENDS[default_backend]

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
