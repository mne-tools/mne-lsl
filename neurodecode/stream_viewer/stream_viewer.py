import sys
import time
from PyQt5.QtWidgets import QApplication

from ._scope import _ScopeEEG
from ._scope_controller import _ScopeControllerUI
from .. import logger
from ..stream_receiver import StreamReceiver, StreamEEG
from ..utils.lsl import search_lsl


class StreamViewer:
    def __init__(self, stream_name=None):
        self.stream_name = stream_name

    def start(self, bufsize=0.2, backend='pyqt5'):
        """
        Connect to the selected amplifier and plot the streamed data.

        If stream infos are not provided, look for available streams on the
        network.
        """
        if self.stream_name is None:
            self.stream_name = search_lsl(ignore_markers=True)

        if isinstance(backend, str):
            backend = backend.lower().strip()

        logger.info(f'Connecting to the stream: {self.stream_name}')
        self.sr = StreamReceiver(bufsize=bufsize, winsize=bufsize,
                                 stream_name=self.stream_name)
        self.sr.streams[self.stream_name].blocking = False
        time.sleep(bufsize)  # Delay to fill the LSL buffer.

        if isinstance(self.sr.streams[self.stream_name], StreamEEG):
            self._scope = _ScopeEEG(self.sr, self.stream_name)
        else:
            logger.error(
                'Unsupported stream type '
                f'{type(self.sr.streams[self.stream_name])}')

        app = QApplication(sys.argv)
        self._ui = _ScopeControllerUI(self._scope, backend)
        sys.exit(app.exec_())
