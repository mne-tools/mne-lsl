from threading import Thread

import mne
import pylsl
import numpy as np

from ._stream import StreamEEG, StreamMarker
from .. import logger
from ..utils import Timer
from ..utils._docs import fill_doc


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
        self._winsize = StreamReceiver._check_winsize(winsize)
        self._bufsize = StreamReceiver._check_bufsize(bufsize, winsize)
        self._stream_name = StreamReceiver._check_format_stream_name(
            stream_name)
        self._connected = False
        self._acquisition_threads = dict()
        self._mne_infos = dict()
        self.connect()

    def connect(self, timeout=5, force=False):
        """
        Search for the available streams on the LSL network and connect to the
        appropriate ones. If a LSL stream fullfills the requirements (name...),
        a connection is established.

        This function is called while instanciating a `~bsl.StreamReceiver`
        and can be recall to reconnect to the LSL streams.

        Parameters
        ----------
        timeout : `int` | `float`
            Timeout duration in seconds after which the search is abandonned.
        force : `bool`
            If ``True``, force reconnect if the `~bsl.StreamReceiver` was
            already connected.
        """
        if not force and self._connected:
            return True
        self._connected = False
        self._streams = dict()

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
                if self._stream_name is not None and \
                   streamInfo.name() not in self._stream_name:
                    logger.info(f'Stream {streamInfo.name()} skipped.')
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

            if stream not in self._mne_infos:
                if isinstance(self._streams[stream], StreamEEG):
                    ch_names = self._streams[stream].ch_list
                    sfreq = self._streams[stream].sample_rate
                    ch_types = ['eeg'] * len(ch_names)
                    self._mne_infos[stream] = mne.create_info(
                        ch_names, sfreq, ch_types)
                else:
                    self._mne_infos[stream] = None

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

    def disconnect(self, stream_name=None):
        """
        Disconnects the stream ``stream_name`` from the `~bsl.StreamReceiver`.
        If ``stream_name`` is a `list`, disconnects all streams in the list.
        If ``stream_name`` is `None`, disconnects all streams.

        Parameters
        ----------
        stream_name : `str` | `list` | `None`
            Servers' name or list of servers' name to disconnect from.
            If `None`, disconnect from all streams.
        """
        stream_name = StreamReceiver._check_format_stream_name(stream_name)
        if stream_name is None:
            stream_name = list(self._streams)

        for stream in list(self._streams):
            if stream not in stream_name:
                continue

            self._streams[stream]._inlet.close_stream()
            del self._streams[stream]
            del self._acquisition_threads[stream]

        if len(self._streams) == 0:
            self._connected = False

    def acquire(self):
        """
        Read data from the streams and fill their buffer using threading.
        """
        if not self._connected:
            logger.error(
                'The Stream Receiver is not connected to any streams. ')
            raise RuntimeError

        for stream in self._streams:
            if self._acquisition_threads[stream] is not None and \
                    self._acquisition_threads[stream].is_alive():
                continue

            thread = Thread(target=self._streams[stream].acquire, args=[])
            thread.daemon = True
            thread.start()
            self._acquisition_threads[stream] = thread

    @fill_doc
    def get_window(self, stream_name=None, return_raw=False, verbose=True):
        """
        Get the latest window from a stream's buffer.
        If several streams are connected, specify the name.

        Parameters
        ----------
        %(receiver_get_stream_name)s
        %(receiver_get_return_raw)s
        %(receiver_get_verbose)s

        Returns
        -------
        %(receiver_data)s
        %(receiver_timestamps)s

        Notes
        -----
        %(receiver_get_unit)s
        """
        if not self._connected:
            logger.error(
                'The Stream Receiver is not connected to any streams. ')
            raise RuntimeError

        if stream_name is None and len(self._streams) == 1:
            stream_name = list(self._streams)[0]
        elif stream_name is None and len(self._streams) > 1:
            logger.error('The Stream Receiver is connected to multiple '
                         'streams. Please provide the stream_name argument. ')
            raise RuntimeError

        try:
            self._acquisition_threads[stream_name].join()
        except KeyError as error:
            logger.error(
                f"The Stream Receiver is not connected to '{stream_name}'.",
                exc_info=True)
            raise error
        except AttributeError as error:
            logger.warning(
                '.acquire() must be called before .get_buffer().',
                exc_info=True)
            raise error

        winsize = self._streams[stream_name].buffer.winsize
        try:
            window = np.array(self._streams[
                stream_name].buffer.data[-winsize:])
            timestamps = np.array(self._streams[
                stream_name].buffer.timestamps[-winsize:])
        except IndexError:
            logger.warning(
                f"The buffer of {stream_name} does not contain enough "
                "samples. Returning the available samples.")
            window = np.array(self._streams[stream_name].buffer.data)
            timestamps = np.array(self._streams[stream_name].buffer.timestamps)

        if len(timestamps) > 0:
            if bool(return_raw) and self._mne_infos[stream_name] is not None:
                window = mne.io.RawArray(
                    window.T, self._mne_infos[stream_name])
                window._filenames = [f'BSL {stream_name}']
            elif bool(return_raw) and self._mne_infos[stream_name] is None:
                logger.warning(
                    f'The LSL stream {stream_name} can not be converted to'
                    'MNE raw instance. Returning numpy arrays.')
        else:
            if verbose:
                logger.warning(
                    f'The LSL stream {stream_name} did not return any data.'
                    'Returning empty numpy arrays.')
            window = np.empty((0, len(self._streams[stream_name].ch_list)))
            timestamps = np.array([])

        return window, timestamps

    @fill_doc
    def get_buffer(self, stream_name=None, return_raw=False, verbose=True):
        """
        Get the entire buffer of a stream.
        If several streams are connected, specify the name.

        Parameters
        ----------
        %(receiver_get_stream_name)s
        %(receiver_get_return_raw)s
        %(receiver_get_verbose)s

        Returns
        -------
        %(receiver_data)s
        %(receiver_timestamps)s

        Notes
        -----
        %(receiver_get_unit)s
        """
        if not self._connected:
            logger.error(
                'The Stream Receiver is not connected to any streams. ')
            raise RuntimeError

        if stream_name is None and len(self._streams) == 1:
            stream_name = list(self._streams)[0]
        elif stream_name is None and len(self._streams) > 1:
            logger.error('The Stream Receiver is connected to multiple '
                         'streams. Please provide the stream_name argument. ')
            raise RuntimeError

        try:
            self._acquisition_threads[stream_name].join()
        except KeyError as error:
            logger.error(
                f"The Stream Receiver is not connected to '{stream_name}'.")
            raise error
        except AttributeError as error:
            logger.warning('.acquire() must be called before .get_buffer().')
            raise error

        window = np.array(self._streams[stream_name].buffer.data)
        timestamps = np.array(self._streams[stream_name].buffer.timestamps)
        if len(self._streams[stream_name].buffer.timestamps) > 0:
            if bool(return_raw) and self._mne_infos[stream_name] is not None:
                window = mne.io.RawArray(
                    window.T, self._mne_infos[stream_name])
                window._filenames = [f'BSL {stream_name}']
            elif bool(return_raw) and self._mne_infos[stream_name] is None:
                logger.warning(
                    f'The LSL stream {stream_name} can not be converted to'
                    'MNE raw instance. Returning numpy arrays.')
        else:
            if verbose:
                logger.warning(
                    f'The LSL stream {stream_name} did not return any data.'
                    'Returning empty numpy arrays.')
            window = np.empty((0, len(self._streams[stream_name].ch_list)))
            timestamps = np.array([])

        return window, timestamps

    def reset_buffer(self, stream_name=None):
        """
        Clear the stream's buffer.

        Parameters
        ----------
        stream_name : `str` | `list` | `None`
            Name of the stream(s) to reset its buffer.
            If `None`, reset all stream's buffer.
        """
        stream_name = StreamReceiver._check_format_stream_name(stream_name)
        if stream_name is None:
            stream_name = list(self._streams)

        for stream in self._streams:
            if stream not in stream_name:
                continue

            self._streams[stream].buffer.reset_buffer()

    # --------------------------------------------------------------------
    @staticmethod
    def _check_winsize(winsize):
        """
        Check that winsize is positive.
        """
        if winsize <= 0:
            logger.error(f'Invalid window size {winsize}.')
            raise ValueError

        return winsize

    @staticmethod
    def _check_bufsize(bufsize, winsize):
        """
        Check that bufsize is positive and bigger than the winsize.
        """
        if bufsize <= 0:
            logger.error(f'Invalid buffer size {bufsize}.')
            raise ValueError

        if bufsize < winsize:
            logger.error(
                f'Buffer size  {bufsize:.1f} is smaller than window size. '
                f'Setting to {winsize:.1f}')
            bufsize = winsize

        return bufsize

    @staticmethod
    def _check_format_stream_name(stream_name):
        """
        Check the format of stream_name.
        """
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
        """
        Buffer's size ``[sec]``.

        :setter: Checks that the bufsize is valid.
        :type: `int` | `float`
        """
        return self._bufsize

    @bufsize.setter
    def bufsize(self, bufsize):
        self._bufsize = StreamReceiver._check_bufsize(bufsize, self._winsize)
        self.connect(force=True)

    @property
    def winsize(self):
        """
        Window's size ``[sec]``.

        :setter: Checks that the winsize is smaller than the bufsize.
        :type: `int` | `float`
        """
        return self._winsize

    @winsize.setter
    def winsize(self, winsize):
        self._winsize = StreamReceiver._check_winsize(winsize)
        self._bufsize = StreamReceiver._check_bufsize(self._bufsize, winsize)
        self.connect(force=True)

    @property
    def stream_name(self):
        """
        Connected stream's name.

        :setter: Try to connect to the new stream(s), revert to old stream(s)
            if failed.
        :type: `list` | `None`
        """
        return self._stream_name

    @stream_name.setter
    def stream_name(self, stream_name):
        old_stream_name = self._stream_name
        self._stream_name = StreamReceiver._check_format_stream_name(
            stream_name)
        self.connect(force=True)

        if not self._connected:
            logger.error(
                'The Stream Receiver could not connect to the new stream '
                'names. Reconnecting to the old stream names.')
            self._stream_name = old_stream_name
            self.connect(force=True)

    @property
    def streams(self):
        """
        Connected streams dictionary ``{stream_name: _Stream}``.

        :type: `dict`
        """
        return self._streams

    @property
    def mne_infos(self):
        """
        Dictionary containing the `mne.Info` for the compatible streams.

        :type: `dict`
        """
        return self._mne_infos

    @property
    def connected(self):
        """
        Connected status.

        :type: `bool`
        """
        return self._connected
