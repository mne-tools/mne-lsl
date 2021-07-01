import sys
import time
from PyQt5.QtWidgets import QApplication

from ._scope import ScopeEEG
from ._scope_controller import _ScopeControllerUI
from .. import logger
from ..stream_receiver import StreamReceiver, StreamEEG
from ..utils.lsl import search_lsl


class StreamViewer:
    """
    StreamViewer instance. The stream viewer will connect to only one LSL
    stream. If stream_name is set to None, an automatic search is performed
    followed by a prompt if multiple non-markers streams are found.

    Supports 2 backends:
        'pyqt5': fully functional.
        'vispy': signal displayed with limited control and information.

    Parameters
    ----------
    stream_name : str | None
        Servers' name to connect to.
        None: prompts the user.
    """

    def __init__(self, stream_name=None):
        self._stream_name = StreamViewer._check_stream_name(stream_name)

    def start(self, bufsize=0.2, backend='pyqt5'):
        """
        Connect to the selected amplifier and plot the streamed data.

        If stream infos are not provided, look for available streams on the
        network.
        """
        if self._stream_name is None:
            self._stream_name = search_lsl(ignore_markers=True)

        if isinstance(backend, str):
            backend = backend.lower().strip()

        logger.info(f'Connecting to the stream: {self.stream_name}')
        self.sr = StreamReceiver(bufsize=bufsize, winsize=bufsize,
                                 stream_name=self._stream_name)
        self.sr.streams[self._stream_name].blocking = False
        time.sleep(bufsize)  # Delay to fill the LSL buffer.

        if isinstance(self.sr.streams[self._stream_name], StreamEEG):
            self._scope = ScopeEEG(self.sr, self._stream_name)
        else:
            logger.error(
                'Unsupported stream type '
                f'{type(self.sr.streams[self._stream_name])}')

        app = QApplication(sys.argv)
        self._ui = _ScopeControllerUI(self._scope, backend)
        sys.exit(app.exec_())

    # --------------------------------------------------------------------
    @staticmethod
    def _check_stream_name(stream_name):
        if stream_name is not None and not isinstance(stream_name, str):
            logger.error(
                'The stream name must be either None to prompt the user or a '
                'string.')
            raise ValueError

        return stream_name

    # --------------------------------------------------------------------
    @property
    def stream_name(self):
        """
        The connected stream's name.
        """
        return self._stream_name

    @stream_name.setter
    def stream_name(self, stream_name):
        logger.warning("The connected streams cannot be changed.")
