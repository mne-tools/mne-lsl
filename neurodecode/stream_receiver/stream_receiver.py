from threading import Thread

import pylsl
import numpy as np

from ._stream import StreamEEG, StreamMarker
from .. import logger
from ..utils.timer import Timer


class StreamReceiver:
    """
    Class for data acquisition from LSL streams.

    It supports the streams of:
        - EEG
        - Markers

    Parameters
    ----------
    bufsize : int
        The buffer's size [secs]. 1-day is the maximum size.
        Large buffer may lead to a delay if not pulled frequently.
    winsize : int
            To extract the latest winsize samples from the buffer [secs].
    stream_name : list | str
        Servers' name or list of servers' name to connect to.
        None: no constraint.
    """

    def __init__(self, bufsize=1, winsize=1, stream_name=None):
        self._acquisition_threads = dict()
        self._bufsize = bufsize
        self._winsize = winsize
        self._stream_name = StreamReceiver._check_format_stream_name(
            stream_name)
        self.connect(self._stream_name)

    def connect(self, timeout=10):
        """
        Search for the available streams on the LSL network and connect to the
        appropriate ones. If a LSL stream fullfills the requirements (name...),
        a connection is established.

        This function is called while instanciating a StreamReceiver and can be
        recall to reconnect to the LSL streams

        Parameters
        ----------
        timeout : int
            Timeout duration in seconds after which the search is abandonned.
        """
        self._streams = dict()
        self._connected = False

        if self._stream_name is None:
            logger.info(
                "Looking for available LSL streaming servers...")
        else:
            logger.info(
                f"Looking for server(s): '{', '.join(self._stream_name)}'...")

        watchdog = Timer()

        while watchdog.sec() <= timeout:
            streamInfos = pylsl.resolve_streams()
            for streamInfo in streamInfos:

                # connect to a specific amp only?
                if streamInfo.name() not in self._stream_name:
                    logger.info(f'Stream {streamInfo.name()} skipped.')
                    continue
                # TODO: To be removed.
                if streamInfo.name() == 'StreamRecorderInfo':
                    continue

                # EEG stream
                if streamInfo.type().lower() == "eeg":
                    self._streams[streamInfo.name()] = StreamEEG(
                        streamInfo, self._bufsize, self._winsize)
                # Marker stream
                elif streamInfo.nominal_srate() == 0:
                    self._streams[streamInfo.name()] = StreamMarker(
                        streamInfo, self._bufsize, self._winsize)

                self._connected = True

            if self._connected:
                break
        else:
            logger.error(
                'Connection timeout. Could not connect to an LSL stream.')
            return False

        for stream in self._streams:
            if stream not in self._acquisition_threads:
                self._acquisition_threads[stream] = None

        self.show_info()
        logger.info('Ready to receive data from the connected streams.')
        return True

    def show_info(self):
        """
        Display the informations about the connected streams.
        """
        for stream in self._streams:
            logger.info("--------------------------------"
                        "--------------------------------")
            logger.info(f"The stream {stream} is connected to:")
            self._streams[stream].show_info()

    # --------------------------------------------------------------------
    def reconnect(func):
        """
        Decorator to check if the StreamReceiver is connected to LSL Streams.
        """
        def wrapper(self, *args, **kwargs):
            if not self._connected:
                self.connect()
            func(*args, **kwargs)
        return wrapper

    def check_arg_stream_name(func):
        """
        Decorator to check the argument stream_name.
        Checks if the list of connected streams is not empty and checks if
        stream_name is set to None that the list of connected streams only
        contains one stream.
        """
        def wrapper(self, stream_name):
            if len(self._streams) == 0:
                logger.error(
                    'The StreamReceiver is not connected to any streams.')
                raise RuntimeError
            elif len(self._streams) == 1 and stream_name is None:
                stream_name = list(self._streams.keys())[0]
            elif len(self._streams) > 1 and stream_name is None:
                logger.error(
                    "Multiple streams connected. "
                    "Please provide a stream to remove it.")
                raise ValueError

            func(self, stream_name)
        return wrapper

    def check_acquisition_thread(func):
        def wrapper(self, stream_name):
            try:
                self._acquisition_threads[stream_name].join()
            except KeyError:
                logger.error(
                    f"The StreamReceiver is not connected to '{stream_name}'.")
                raise RuntimeError
            except AttributeError:
                logger.warning(
                    '.acquire() must be called before .get_window().')
                return (np.empty((0, len(self._streams[stream_name].ch_list))),
                        np.array([])) #TODO: Does this work?

            func(self, stream_name)
        return wrapper

    # --------------------------------------------------------------------
    @check_arg_stream_name
    def disconnect_stream(self, stream_name=None):
        """
        Disconnects the stream 'stream_name' from the StreamReceiver.
        If several streams are connected, specify the name.

        Parameters
        ----------
        stream_name : str
            The name of the stream to extract from.
        """
        try:
            self._streams[stream_name]._inlet.close_stream()
            del self._streams[stream_name]
            del self._acquisition_threads[stream_name]
        except KeyError:
            logger.error(
                f"The stream '{stream_name}' does not exist. Skipping.")

    @reconnect
    def acquire(self):
        """
        Read data from the streams and fill their buffer using threading.
        """
        for stream in self._streams:
            if self._acquisition_threads[stream] is not None and \
                    self._acquisition_threads[stream].is_alive():
                continue

            thread = Thread(target=self._streams[stream].acquire, args=[])
            thread.daemon = True
            thread.start()
            self._acquisition_threads[stream] = thread

    @reconnect
    @check_arg_stream_name
    @check_acquisition_thread
    def get_window(self, stream_name=None):
        """
        Get the latest window from a stream's buffer.
        If several streams are connected, specify the name.

        Parameters
        ----------
        stream_name : str
            The name of the stream to extract from.

        Returns
        -------
        data : np.array
             The data [samples x channels]
        timestamps : np.array
             The timestamps [samples]
        """
        winsize = self._streams[stream_name].buffer.winsize

        try:
            window = self._streams[stream_name].buffer.data[-winsize:]
            timestamps = self._streams[stream_name].buffer.timestamps[-winsize:]
        except IndexError:
            logger.warning(
                f"The buffer of {self._streams[stream_name].name} "
                "does not contain enough samples.")
            window = self._streams[stream_name].buffer.data[:]
            timestamps = self._streams[stream_name].buffer.timestamps[:]

        if len(timestamps) > 0:
            return (np.array(window), np.array(timestamps))
        else:
            return (np.empty((0, len(self._streams[stream_name].ch_list))),
                    np.array([]))

    @reconnect
    @check_arg_stream_name
    @check_acquisition_thread
    def get_buffer(self, stream_name=None):
        """
        Get the entire buffer of a stream in numpy format.
        If several streams are connected, specify the name.

        Parameters
        ----------
        stream_name : str
            The name of the stream to extract from.

        Returns
        -------
        data : np.array
            The data [samples x channels]
        timestamps : np.array
            The timestamps [samples]
        """
        if len(self._streams[stream_name].buffer.timestamps) > 0:
            return (np.array(self._streams[stream_name].buffer.data),
                    np.array(self._streams[stream_name].buffer.timestamps))
        else:
            return (np.empty((0, len(self._streams[stream_name].ch_list))),
                    np.array([]))

    @check_arg_stream_name
    def reset_buffer(self, stream_name=None):
        """
        Clear the stream's buffer.
        If several streams are connected, specify the name.

        Parameters
        ----------
        stream_name : str
            The stream's name.
        """
        self._streams[stream_name].buffer.reset_buffer()

    def reset_all_buffers(self):
        """
        Clear all the streams' buffer.
        """
        for stream in self._streams:
            self.reset_buffer(stream)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_format_stream_name(stream_name):
        if isinstance(stream_name, (list, tuple)):
            stream_name = list(stream_name)
            if not all(isinstance(name, str) for name in stream_name):
                logger.error(
                    'Stream name must be a string or a list of strings.')
                raise TypeError
        elif isinstance(stream_name, str):
            stream_name = [stream_name]
        else:
            stream_name = None

        return stream_name

    # --------------------------------------------------------------------
    @property
    def bufsize(self):
        return self._bufsize

    @property
    def winsize(self):
        return self._winsize

    @property
    def stream_name(self):
        return self._stream_name

    @stream_name.setter
    def stream_name(self, stream_name):
        old_stream_name = self._stream_name
        self._stream_name = StreamReceiver._check_format_stream_name(
            stream_name)
        try:
            self.connect()
        except RuntimeError:
            self._stream_name = old_stream_name
            logger.warning(
                "Could not connected to '{', '.join(self._stream_name)}'.")

    @property
    def streams(self):
        """
        The connected streams list.
        """
        return self._streams

    @streams.setter
    def streams(self, streams):
        logger.warning("The connected streams cannot be modified.")
