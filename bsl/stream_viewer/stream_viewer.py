import sys
import time

from PyQt5.QtWidgets import QApplication

from .scope.scope_eeg import ScopeEEG
from .control_gui.control_eeg import ControlGUI_EEG
from ..stream_receiver import StreamReceiver, StreamEEG
from ..utils._logs import logger
from ..utils.lsl import search_lsl
from ..utils._checks import _check_type


class StreamViewer:
    """
    Class for visualizing the signals coming from an LSL stream. The stream
    viewer will connect to only one LSL stream. If ``stream_name`` is set to
    ``None``, an automatic search is performed followed by a prompt if multiple
    non-markers streams are found.

    Parameters
    ----------
    stream_name : str | None
        Servers' name to connect to. ``None`` will prompt the user.
    """

    def __init__(self, stream_name=None):
        self._stream_name = StreamViewer._check_stream_name(stream_name)

    def start(self, bufsize=0.2):
        """
        Connect to the selected amplifier and plot the streamed data.

        If ``stream_name`` is not provided, look for available streams on the
        network.

        Parameters
        ----------
        bufsize : int | float
            Buffer/window size of the attached StreamReceiver.
            The default ``0.2`` should work in most cases since data is fetched
            every 20 ms.
        """
        logger.info('Connecting to the stream: %s', self.stream_name)
        self._sr = StreamReceiver(bufsize=bufsize, winsize=bufsize,
                                  stream_name=self._stream_name)
        self._sr.streams[self._stream_name].blocking = False
        time.sleep(bufsize)  # Delay to fill the LSL buffer.

        if isinstance(self._sr.streams[self._stream_name], StreamEEG):
            self._scope = ScopeEEG(self._sr, self._stream_name)
            app = QApplication(sys.argv)
            self._ui = ControlGUI_EEG(self._scope)
            sys.exit(app.exec_())
        else:
            logger.error(
                'Unsupported stream type %s',
                type(self._sr.streams[self._stream_name]))

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
