from threading import Thread

import mne
import numpy as np

from ..externals import pylsl
from ..utils import Timer
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils._logs import logger
from ._stream import StreamEEG, StreamMarker


@fill_doc
class StreamReceiver:
    """
    Class for data acquisition from LSL streams.

    It supports the streams of:
        - EEG
        - Markers

    Parameters
    ----------
    %(receiver_bufsize)s
    %(receiver_winsize)s
    %(stream_name)s
    """

    def __init__(self, bufsize=1, winsize=1, stream_name=None):
        self._connected = False
        self._streams = dict()
        self._acquisition_threads = dict()
        self._mne_infos = dict()

        self._winsize = StreamReceiver._check_winsize(winsize)
        self._bufsize = StreamReceiver._check_bufsize(bufsize, self._winsize)
        self.connect(stream_name)

    @fill_doc
    def connect(self, stream_name=None, timeout=5, force=False):
        """
        Search for the available streams on the LSL network and connect to the
        appropriate ones. If a LSL stream fulfills the requirements (name...),
        a connection is established.

        This function is called while instantiated a StreamReceiver and can be
        recall to reconnect to the LSL streams.

        Parameters
        ----------
        %(stream_name)s
        timeout : int | float
            Timeout duration in seconds after which the search is abandoned.
        force : bool
            If ``True``, force reconnect if the StreamReceiver was already
            connected.
        """
        _check_type(timeout, ("numeric",), item_name="timeout")
        _check_type(force, (bool,), item_name="force")

        if not force and self._connected:
            return True

        stream_name = StreamReceiver._check_format_stream_name(stream_name)
        if stream_name is not None and force and self._connected:
            self._stream_name.extend(stream_name)
        else:
            self._stream_name = stream_name

        self._connected = False
        self._streams = dict()

        if self._stream_name is None:
            logger.info("Looking for available LSL streaming servers...")
        else:
            logger.info(
                "Looking for server(s): '%s'...", ", ".join(self._stream_name)
            )

        watchdog = Timer()
        while watchdog.sec() <= timeout:
            streamInfos = pylsl.resolve_streams()
            for streamInfo in streamInfos:

                # connect to a specific amp only?
                if (
                    self._stream_name is not None
                    and streamInfo.name() not in self._stream_name
                ):
                    logger.info("Stream %s skipped.", streamInfo.name())
                    continue

                # EEG stream
                if streamInfo.type().lower() == "eeg":
                    self._streams[streamInfo.name()] = StreamEEG(
                        streamInfo, self._bufsize, self._winsize
                    )
                # Marker stream
                elif streamInfo.nominal_srate() == 0:
                    self._streams[streamInfo.name()] = StreamMarker(
                        streamInfo, self._bufsize, self._winsize
                    )

                self._connected = True

            if self._connected:
                break
        else:
            logger.error(
                "Connection timeout. Could not connect to an LSL stream."
            )
            return False

        for stream in self._streams:
            if stream not in self._acquisition_threads:
                self._acquisition_threads[stream] = None

            if stream not in self._mne_infos:
                if isinstance(self._streams[stream], StreamEEG):
                    ch_names = self._streams[stream].ch_list
                    sfreq = self._streams[stream].sample_rate
                    ch_types = ["eeg"] * len(ch_names)
                    self._mne_infos[stream] = mne.create_info(
                        ch_names, sfreq, ch_types
                    )
                else:
                    self._mne_infos[stream] = None

        self.show_info()
        logger.info("Ready to receive data from the connected streams.")
        return True

    def show_info(self):
        """
        Display the information about the connected streams.
        """
        for stream in self._streams:
            logger.info(
                "--------------------------------"
                "--------------------------------"
            )
            logger.info("The stream %s is connected to:", stream)
            self._streams[stream].show_info()

    def disconnect(self, stream_name=None):
        """
        Disconnects the stream ``stream_name`` from the StreamReceiver.
        If ``stream_name`` is a `list`, disconnects all streams in the list.
        If ``stream_name`` is ``None``, disconnects all streams.

        Parameters
        ----------
        stream_name : str | list | None
            Servers' name or list of servers' name to disconnect from.
            If ``None``, disconnect from all streams.
        """
        if not self._connected:
            return

        stream_name = StreamReceiver._check_format_stream_name(stream_name)
        if stream_name is None:
            stream_name = list(self._streams)

        for stream in list(self._streams):
            if stream not in stream_name:
                continue

            self._streams[stream]._inlet.close_stream()
            del self._streams[stream]
            del self._acquisition_threads[stream]
            if self._stream_name is not None:
                idx = self._stream_name.index(stream)
                del self._stream_name[idx]

        if len(self._streams) == 0:
            self._connected = False

    def acquire(self):
        """
        Read data from the streams and fill their buffer using threading.
        """
        if not self._connected:
            raise RuntimeError(
                "StreamReceiver is not connected to any " "streams."
            )

        for stream in self._streams:
            if (
                self._acquisition_threads[stream] is not None
                and self._acquisition_threads[stream].is_alive()
            ):
                continue

            thread = Thread(target=self._streams[stream].acquire, args=[])
            thread.daemon = True
            thread.start()
            self._acquisition_threads[stream] = thread

    @fill_doc
    def get_window(self, stream_name=None, return_raw=False):
        """
        Get the latest window from a stream's buffer.
        If several streams are connected, specify the name.

        Parameters
        ----------
        %(receiver_get_stream_name)s
        %(receiver_get_return_raw)s

        Returns
        -------
        %(receiver_data)s
        %(receiver_timestamps)s

        Notes
        -----
        %(receiver_get_unit)s
        """
        if not self._connected:
            raise RuntimeError(
                "StreamReceiver is not connected to any " "streams."
            )

        if stream_name is None and len(self._streams) == 1:
            stream_name = list(self._streams)[0]
        elif stream_name is None and len(self._streams) > 1:
            raise RuntimeError(
                "StreamReceiver is connected to multiple streams. Please "
                "provide the stream_name argument."
            )

        try:
            self._acquisition_threads[stream_name].join()
        except KeyError:
            raise KeyError(
                "StreamReceiver is not connected to '%s'." % stream_name
            )
        except AttributeError:
            raise AttributeError(
                ".acquire() must be called before .get_window()."
            )

        winsize = self._streams[stream_name].buffer.winsize
        window = np.array(self._streams[stream_name].buffer.data[-winsize:])
        timestamps = np.array(
            self._streams[stream_name].buffer.timestamps[-winsize:]
        )
        if len(timestamps) != winsize:
            logger.warning(
                "The buffer of %s does not contain enough samples. Returning "
                "the available samples.",
                stream_name,
            )

        if len(timestamps) > 0:
            if bool(return_raw) and self._mne_infos[stream_name] is not None:
                window = mne.io.RawArray(
                    window.T, self._mne_infos[stream_name]
                )
                window._filenames = [f"BSL {stream_name}"]
            elif bool(return_raw) and self._mne_infos[stream_name] is None:
                logger.warning(
                    "The LSL stream %s can not be converted to MNE raw "
                    "instance. Returning numpy arrays.",
                    stream_name,
                )
        else:
            logger.warning(
                "The LSL stream %s did not return any data."
                "Returning empty numpy arrays.",
                stream_name,
            )
            window = np.empty((0, len(self._streams[stream_name].ch_list)))
            timestamps = np.array([])

        return window, timestamps

    @fill_doc
    def get_buffer(self, stream_name=None, return_raw=False):
        """
        Get the entire buffer of a stream.
        If several streams are connected, specify the name.

        Parameters
        ----------
        %(receiver_get_stream_name)s
        %(receiver_get_return_raw)s

        Returns
        -------
        %(receiver_data)s
        %(receiver_timestamps)s

        Notes
        -----
        %(receiver_get_unit)s
        """
        if not self._connected:
            raise RuntimeError(
                "StreamReceiver is not connected to any " "streams."
            )

        if stream_name is None and len(self._streams) == 1:
            stream_name = list(self._streams)[0]
        elif stream_name is None and len(self._streams) > 1:
            raise RuntimeError(
                "StreamReceiver is connected to multiple streams. Please "
                "provide the stream_name argument."
            )

        try:
            self._acquisition_threads[stream_name].join()
        except KeyError:
            raise KeyError(
                "StreamReceiver is not connected to '%s'." % stream_name
            )
        except AttributeError:
            raise AttributeError(
                ".acquire() must be called before .get_buffer()."
            )

        window = np.array(self._streams[stream_name].buffer.data)
        timestamps = np.array(self._streams[stream_name].buffer.timestamps)
        if len(self._streams[stream_name].buffer.timestamps) > 0:
            if bool(return_raw) and self._mne_infos[stream_name] is not None:
                window = mne.io.RawArray(
                    window.T, self._mne_infos[stream_name]
                )
                window._filenames = [f"BSL {stream_name}"]
            elif bool(return_raw) and self._mne_infos[stream_name] is None:
                logger.warning(
                    "The LSL stream %s can not be converted to MNE raw "
                    "instance. Returning numpy arrays.",
                    stream_name,
                )
        else:
            logger.warning(
                "The LSL stream %s did not return any data."
                "Returning empty numpy arrays.",
                stream_name,
            )
            window = np.empty((0, len(self._streams[stream_name].ch_list)))
            timestamps = np.array([])

        return window, timestamps

    def _get_buffer(self):
        """
        Get the entire buffer of the only connected stream.
        This method is intended for use by the StreamViewer.
        """
        stream_name = list(self._streams)[0]
        self._acquisition_threads[stream_name].join()
        window = np.array(self._streams[stream_name].buffer.data)
        timestamps = np.array(self._streams[stream_name].buffer.timestamps)
        if len(self._streams[stream_name].buffer.timestamps) == 0:
            window = np.empty((0, len(self._streams[stream_name].ch_list)))
            timestamps = np.array([])
        return window, timestamps

    def reset_buffer(self, stream_name=None):
        """
        Clear the stream's buffer.

        Parameters
        ----------
        stream_name : str | list | None
            Name of the stream(s) to reset its buffer.
            If ``None``, reset all stream's buffer.
        """
        stream_name = StreamReceiver._check_format_stream_name(stream_name)
        if stream_name is None:
            stream_name = list(self._streams)

        for stream in self._streams:
            if stream not in stream_name:
                continue

            self._streams[stream].buffer.reset_buffer()

    # --------------------------------------------------------------------
    def __del__(self):
        """Destructor method."""
        self.disconnect()

    def __repr__(self):
        """Representation of the instance."""
        status = "ON" if self._connected else "OFF"
        if self._connected:
            streams = str(tuple(self._streams))
        else:
            streams = (
                "()"
                if self._stream_name is None
                else str(tuple(self._stream_name))
            )
        repr_str = (
            f"<Receiver: {streams} | {status} | "
            + f"buf: {self._bufsize}s - win: {self._winsize}s>"
        )
        return repr_str

    # --------------------------------------------------------------------
    @staticmethod
    def _check_bufsize(bufsize, winsize):
        """
        Check that bufsize is positive and bigger than the winsize.
        """
        _check_type(bufsize, ("numeric",), item_name="bufsize")
        bufsize = float(bufsize)
        if bufsize <= 0:
            raise ValueError(
                "Argument bufsize must be a strictly positive int or a "
                "float. Provided: %s" % bufsize
            )
        if bufsize < winsize:
            logger.error(
                "Buffer size %.2f is smaller than window size. "
                "Setting to %.2f.",
                bufsize,
                winsize,
            )
            bufsize = winsize

        return bufsize

    @staticmethod
    def _check_winsize(winsize):
        """
        Check that winsize is positive.
        """
        _check_type(winsize, ("numeric",), item_name="winsize")
        winsize = float(winsize)
        if winsize <= 0:
            raise ValueError(
                "Argument winsize must be a strictly positive int or a "
                "float. Provided: %s" % winsize
            )

        return winsize

    @staticmethod
    def _check_format_stream_name(stream_name):
        """
        Check the format of stream_name.
        """
        _check_type(
            stream_name, (None, str, list, tuple), item_name="stream_name"
        )
        if isinstance(stream_name, (list, tuple)):
            stream_name = list(stream_name)
            if not all(isinstance(name, str) for name in stream_name):
                raise TypeError(
                    "Argument stream_name must be a string or a list of "
                    "strings. Provided: %s" % stream_name
                )
        elif isinstance(stream_name, str):
            stream_name = [stream_name]

        return stream_name

    # --------------------------------------------------------------------
    @property
    def winsize(self):
        """
        Window's size ``[sec]``.

        :type: int | float
        """
        return self._winsize

    @property
    def bufsize(self):
        """
        Buffer's size ``[sec]``.

        :type: int | float
        """
        return self._bufsize

    @property
    def stream_name(self):
        """
        Connected stream's name.

        :type: None | list
        """
        return self._stream_name

    @property
    def connected(self):
        """
        Connected status.

        :type: bool
        """
        return self._connected

    @property
    def mne_infos(self):
        """
        Dictionary containing the Info for the compatible streams.

        :type: dict
        """
        return self._mne_infos

    @property
    def streams(self):
        """
        Connected streams dictionary ``{stream_name: _Stream}``.

        :type: dict
        """
        return self._streams
