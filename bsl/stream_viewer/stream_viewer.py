import sys
import time

from PyQt5.QtWidgets import QApplication

from .scope.scope_eeg import ScopeEEG
from .control_gui.control_eeg import ControlGUI_EEG
from ..stream_receiver import StreamReceiver, StreamEEG
from ..utils._logs import logger
from ..utils.lsl import search_lsl
from ..utils._checks import _check_type, _check_value


class StreamViewer:
    """
    Class for visualizing the signals coming from an LSL stream. The stream
    viewer will connect to only one LSL stream. If ``stream_name`` is set to
    ``None``, an automatic search is performed followed by a prompt if multiple
    non-markers streams are found.

    Supports 2 backends:
        - ``'pyqtgraph'``: fully functional.
        - ``'vispy'``: in progress.

    Parameters
    ----------
    stream_name : str | None
        Servers' name to connect to. ``None`` will prompt the user.
    """

    def __init__(self, stream_name=None):
        self._stream_name = StreamViewer._check_stream_name(stream_name)

    def start(self, bufsize=0.2, backend='pyqtgraph'):
        """
        Connect to the selected amplifier and plot the streamed data.

        If ``stream_name`` is not provided, look for available streams on the
        network.

        Parameters
        ----------
        bufsize : int | float
            Buffer/window size of the attached StreamReceiver.
            The default ``0.2`` should work in most cases.
        backend : str
            Selected backend for plotting. Supports:
                - ``'pyqtgraph'``: fully functional.
                - ``'vispy'``: work in progress.
        """
        backend = StreamViewer._check_backend(backend)

        logger.info(f'Connecting to the stream: {self.stream_name}')
        self._sr = StreamReceiver(bufsize=bufsize, winsize=bufsize,
                                  stream_name=self._stream_name)
        self._sr.streams[self._stream_name].blocking = False
        time.sleep(bufsize)  # Delay to fill the LSL buffer.

        if isinstance(self._sr.streams[self._stream_name], StreamEEG):
            self._scope = ScopeEEG(self._sr, self._stream_name)
            app = QApplication(sys.argv)
            self._ui = ControlGUI_EEG(self._scope, backend)
            sys.exit(app.exec_())
        else:
            logger.error(
                'Unsupported stream type '
                f'{type(self._sr.streams[self._stream_name])}')

    # --------------------------------------------------------------------
    @staticmethod
    def _check_stream_name(stream_name):
        """
        Checks that the stream_name is valid or search for a valid stream on
        the network.
        """
        _check_type(stream_name, (None, str), item_name='stream_name')
        if stream_name is None:
            stream_name = search_lsl(ignore_markers=True)
            if stream_name is None:
                raise RuntimeError('No LSL stream found.')
        return stream_name

    @staticmethod
    def _check_backend(backend):
        """
        Checks that the backend is a string.
        """
        _check_type(backend, (str, ), item_name='backend')
        backend = backend.lower().strip()
        _check_value(backend, ('pyqtgraph', 'vispy'))
        return backend

    # --------------------------------------------------------------------
    @property
    def stream_name(self):
        """
        Connected stream's name.

        :type: str
        """
        return self._stream_name

    @property
    def sr(self):
        """
        Connected StreamReceiver.

        :type: StreamReceiver
        """
        return self._sr
