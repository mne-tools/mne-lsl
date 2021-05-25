#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QApplication

from neurodecode import logger
from neurodecode.stream_viewer._scope import _Scope
from neurodecode.utils.lsl import search_lsl


class StreamViewer:
    """
    Class for displaying in real time the signals coming from a LSL stream.

    Parameters
    ----------
    amp_name : str
        The amplifier's name to connect to.
    """

    def __init__(self, amp_name=None):
        self.amp_name = amp_name

    def start(self):
        """
        Connect to the selected amplifier and plot the streamed data.

        If amp infos are not provided, look for available streams on the network.
        """
        if (self.amp_name is None):
            self.search_stream()

        logger.info(f'Connecting to the stream: {self.amp_name}')

        app = QApplication(sys.argv)
        _Scope(self.amp_name)
        sys.exit(app.exec_())

    def search_stream(self):
        """
        Select an available stream on the LSL server to connect to.

        Assign the found amp name and serial number to the internal attributes.
        """
        self.amp_name = search_lsl()


if __name__ == '__main__':

    amp_name = None

    if len(sys.argv) > 2:
        raise RuntimeError("Too many arguments provided, maximum is 1.")

    if len(sys.argv) > 1:
        amp_name = sys.argv[1]

    stream_viewer = StreamViewer(amp_name)
    stream_viewer.start()
