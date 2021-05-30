import sys
from PyQt5.QtWidgets import QApplication

from .. import logger
from ._scope import _Scope
from ..utils.lsl import search_lsl


class StreamViewer:
    """
    Class for displaying in real time the signals coming from a LSL stream.

    Parameters
    ----------
    stream_name : str
        The stream's name to connect to.
    """

    def __init__(self, stream_name=None):
        self.stream_name = stream_name

    def start(self):
        """
        Connect to the selected amplifier and plot the streamed data.

        If stream infos are not provided, look for available streams on the network.
        """
        if (self.stream_name is None):
            self.search_stream(ignore_markers=True)

        logger.info(f'Connecting to the stream: {self.amp_name}')

        app = QApplication(sys.argv)
        _Scope(self.stream_name)
        sys.exit(app.exec_())

    def search_stream(self):
        """
        Select an available stream on the LSL server to connect to.

        Assign the found stream and serial number to the internal attributes.
        """
        self.stream_name = search_lsl()
