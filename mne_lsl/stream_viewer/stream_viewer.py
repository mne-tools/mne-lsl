import sys
import time

from ..lsl import StreamInlet, resolve_streams
from ..utils._checks import check_type
from ..utils.logs import _use_log_level, logger


class StreamViewer:
    """Class for visualizing the signals coming from an LSL stream.

    The stream viewer will connect to only one LSL stream.

    Parameters
    ----------
    stream_name : str | None
        Servers' name to connect to.
    """

    def __init__(self, stream_name=None):
        self._sinfo = StreamViewer._check_stream_name(stream_name)

    def start(self, bufsize=0.2):
        """Connect to the selected amplifier and plot the streamed data.

        Parameters
        ----------
        bufsize : int | float
            Buffer/window size of the attached Viewer.
            The default ``0.2`` should work in most cases since data is fetched
            every 20 ms.
        """
        from qtpy.QtWidgets import QApplication

        from .control_gui.control_eeg import ControlGUI_EEG
        from .scope.scope_eeg import ScopeEEG

        self._inlet = StreamInlet(self._sinfo)
        self._inlet.open_stream()
        time.sleep(bufsize)  # Delay to fill the LSL buffer.
        self._scope = ScopeEEG(self._inlet)
        app = QApplication(sys.argv)
        self._ui = ControlGUI_EEG(self._scope)
        sys.exit(app.exec_())

    # --------------------------------------------------------------------
    @staticmethod
    def _check_stream_name(stream_name):  # noqa
        """Check that the name is valid or search for a valid stream on the network."""
        check_type(stream_name, (None, str), item_name="stream_name")
        streams = resolve_streams(name=stream_name)
        if len(streams) == 0:
            raise RuntimeError("No LSL stream found.")
        elif len(streams) == 1:
            return streams[0]
        else:
            with _use_log_level("INFO"):
                logger.info("-- List of servers --")
                for k, stream in enumerate(streams):
                    logger.info("%i: %s", k, stream.name)
            index = input(
                "Stream index? Hit enter without index to select the first server.\n>> "
            )
            if index.strip() == "":
                index = 0
            else:
                index = int(index.strip())
            return streams[index]
