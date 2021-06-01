import math
import numpy as np
import pylsl
import time

from abc import ABC, abstractmethod

from ._buffer import Buffer
from .. import logger
from ..utils.timer import Timer
from ..utils.preprocess.events import find_event_channel
from ..utils.lsl import lsl_channel_list

_MAX_PYLSL_STREAM_BUFSIZE = 10  # max 10 sec of data buffered
MAX_BUF_SIZE = 86400  # 24h max buffer length
HIGH_LSL_OFFSET_THRESHOLD = 0.1  # Threshold above which the offset is high


class _Stream(ABC):
    """
    Abstract class representing a base receiver's stream.

    Parameters
    ----------
    streamInfo : LSL StreamInfo.
        Contain all the info from the LSL stream to connect to.
    bufsize : int
        The buffer's size [secs]. 1-day is the maximum size.
        Large buffer may lead to a delay if not pulled frequently.
    winsize : int
        To extract the latest winsize samples from the buffer [secs].
    """
    # ----------------------------- Init -----------------------------
    @abstractmethod
    def __init__(self, streamInfo, bufsize, winsize):

        winsize = _Stream._check_window_size(winsize)
        bufsize = _Stream._check_buffer_size(bufsize, winsize)

        self._streamInfo = streamInfo
        self._sample_rate = self.streamInfo.nominal_srate()
        self._lsl_bufsize = min(_MAX_PYLSL_STREAM_BUFSIZE, bufsize)

        if self._sample_rate is not None:
            samples_per_sec = self.sample_rate
            self._inlet = pylsl.StreamInlet(
                streamInfo,
                max_buflen=math.ceil(self._lsl_bufsize))
        else:
            samples_per_sec = 100
            self._inlet = pylsl.StreamInlet(
                streamInfo,
                max_buflen=math.ceil(self._lsl_bufsize)*samples_per_sec)

        self._extract_stream_info()
        self._create_ch_name_list()
        self._lsl_time_offset = None

        self._watchdog = Timer()
        self._blocking = True
        self._blocking_time = 5

        bufsize = _Stream._convert_sec_to_samples(bufsize, samples_per_sec)
        winsize = _Stream._convert_sec_to_samples(winsize, samples_per_sec)
        self._buffer = Buffer(bufsize, winsize)

    def _create_ch_name_list(self):
        """
        Create the channels' name list.
        """
        self._ch_list = lsl_channel_list(self._inlet)

        if not self._ch_list:
            self._ch_list = [f"ch_{i+1}" for i in range(
                self.streamInfo.channel_count())]

    def _extract_stream_info(self):
        """
        Extract the name, serial number and if it's a slave.
        """
        self._name = self.streamInfo.name()
        self._serial = self._inlet.info().desc().child(
            'acquisition').child_value('serial_number')

        if self._serial == '':
            self._serial = 'N/A'

        self.is_slave = 'true' == self._inlet.info().desc().child(
            'amplifier').child('settings').child(
                'is_slave').first_child().value()

    # ---------------------------- Method ----------------------------
    def show_info(self):
        """
        Display the stream's info.
        """
        logger.info(f'Server: {self.name}({self.serial}) '
                    f'/ type:{self.streamInfo.type()} '
                    f'@ {self.streamInfo.hostname()} '
                    f'(v{self.streamInfo.version()}).')
        logger.info(f'Source sampling rate: {self.sample_rate}')
        logger.info(f'Channels: {self.streamInfo.channel_count()}')
        logger.info(f'{self.ch_list}')

        # Check for high LSL offset
        if self.lsl_time_offset is None:
            logger.warning(
                'No LSL timestamp offset computed, no data received yet.')
        elif abs(self.lsl_time_offset) > HIGH_LSL_OFFSET_THRESHOLD:
            logger.warning(
                f'LSL server {self.name}({self.serial}) has a high timestamp '
                f'offset [offset={self.lsl_time_offset:.3f}].')
        else:
            logger.info(
                f'LSL server {self.name}({self.serial}) synchronized '
                f'[offset={self.lsl_time_offset:.3f}]')

    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.

        Returns
        -------
        chunk : list
            data [samples x channels]
        tslist : list
            timestamps [samples]
        """
        chunk = []
        tslist = []
        received = False
        self._watchdog.reset()

        # If first data acquisition, compute LSL offset
        if len(self.buffer.timestamps) == 0:
            compute_timestamp_offset = True
        else:
            compute_timestamp_offset = False

        # Acquire the data
        while not received:
            while self._watchdog.sec() < self._blocking_time:
                if len(tslist) == 0:
                    chunk, tslist = self._inlet.pull_chunk(
                        timeout=0.0, max_samples=self._lsl_bufsize)
                    if self._blocking == False and len(tslist) == 0:
                        received = True
                        break
                if len(tslist) > 0:
                    if compute_timestamp_offset is True:
                        self._compute_offset(tslist)
                    received = True
                    tslist = self._correct_lsl_offset(tslist)
                    break
                time.sleep(0.0005)
            else:
                # Give up and return empty values to avoid deadlock
                logger.warning(
                    f'Timeout occurred [{self.blocking_time}secs] while '
                    f'acquiring data from {self.name}({self.serial}). '
                    'Amp driver bug?')
                received = True

        return chunk, tslist

    def _compute_offset(self, timestamps):
        """
        Compute the LSL offset coming from some devices.

        It has to be called just after acquiring the data/timestamps in order
        to be valid.

        Parameters
        ----------
        timestamps : list
            The acquired timestamps.
        """
        self._lsl_time_offset = timestamps[-1] - pylsl.local_clock()
        return self._lsl_time_offset

    def _correct_lsl_offset(self, timestamps):
        """
        Correct the timestamps if there is a high LSL offset.

        Parameters
        ----------
        timestamps : list
            The timestamps from the last
        """
        if abs(self._lsl_time_offset) > HIGH_LSL_OFFSET_THRESHOLD:
            timestamps = [t - self._lsl_time_offset for t in timestamps]

        return timestamps

    # ---------------------------- Static ----------------------------
    @staticmethod
    def _check_window_size(window_size):
        """
        Check that the window size is positive.

        Parameters
        ----------
        window_size : float
            The window size to verify [secs].

        Returns
        -------
        secs
            The verified window's size.
        """
        if window_size <= 0:
            logger.error(f'Invalid window_size {window_size}.')
            raise ValueError
        return window_size

    @staticmethod
    def _check_buffer_size(buffer_size, window_size):
        """
        Check that buffer's size is positive and bigger than the window's size.

        Parameters
        ----------
        buffer_size : float
            The buffer size to verify [secs].
        window_size : float
            The window's size to compare to buffer_size [secs].

        Returns
        -------
        secs
            The verified buffer size.
        """
        if buffer_size <= 0 or buffer_size > MAX_BUF_SIZE:
            logger.error(
                f'Improper buffer size {buffer_size:.1f}. '
                f'Setting to {MAX_BUF_SIZE:.1f}.')
            buffer_size = MAX_BUF_SIZE

        elif buffer_size < window_size:
            logger.error(
                f'Buffer size  {buffer_size:.1f} is smaller than window size. '
                f'Setting to {window_size:.1f}')
            buffer_size = window_size

        return buffer_size

    @staticmethod
    def _convert_sec_to_samples(bufsec, sample_rate):
        """
        Convert a buffer's/window's size from sec to samples.

        Parameters
        ----------
        bufsec : float
            The buffer_size's size [secs].
        sample_rate : float
             The sampling rate of the LSL server.
             If irregular sampling rate, empirical number of samples per sec.
        Returns
        -------
        samples : int
            The converted buffer_size's size.
        """
        return round(bufsec * sample_rate)

    # -------------------------- Properties --------------------------
    @property
    def name(self):
        """
        The stream's name.
        """
        return self._name

    @name.setter
    def name(self, name):
        logger.warning("The stream's name cannot be changed.")

    @property
    def serial(self):
        """
        The stream's serial number.
        """
        return self._serial

    @serial.setter
    def serial(self, serial):
        logger.warning("The stream's serial cannot be changed.")

    @property
    def sample_rate(self):
        """
        The stream's sampling rate.
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sr):
        logger.warning("The stream's sampling rate cannot be changed.")

    @property
    def streamInfo(self):
        """
        The stream info received from the LSL inlet.
        """
        return self._streamInfo

    @streamInfo.setter
    def streamInfo(self, si):
        logger.warning("The stream's info cannot be changed.")

    @property
    def ch_list(self):
        """
        The channels' name list.
        """
        return self._ch_list

    @ch_list.setter
    def ch_list(self, ch_list):
        logger.warning("The channels' names list cannot be changed.")

    @property
    def buffer(self):
        """
        The buffer containing the data and the timestamps.
        """
        return self._buffer

    @buffer.setter
    def buffer(self, buf):
        logger.warning("The buffer cannot be changed.")

    @property
    def blocking(self):
        """
        If True, the stream wait to receive data.
        """
        return self._buffer

    @blocking.setter
    def blocking(self, block):
        self._blocking = block

    @property
    def blocking_time(self):
        """
        If blocking is True, how long to wait to receive data [secs].
        """
        return self._blocking_time

    @blocking_time.setter
    def blocking_time(self, block_time):
        self._blocking_time = block_time

    @property
    def lsl_time_offset(self):
        """
        The difference between the local and the stream's LSL clocks, used for
        timestamps correction.

        Some LSL servers (like OpenVibe) often have a bug of sending its own
        running time instead of LSL time.
        """
        return self._lsl_time_offset

    @lsl_time_offset.setter
    def lsl_time_offset(self, lsl_time_offset):
        logger.warning("This attribute cannot be changed.")


class StreamMarker(_Stream):
    """
    Class representing a receiver's markers stream.

    Notice the irregular sampling rate.
    This stream is instanciated as non-blocking.

    Parameters
    -----------
    streamInfo : LSL StreamInfo
        Contain all the info from the LSL stream to connect to.
    bufsize : int
        The buffer's size [secs]. 1-day is the maximum size.
        Large buffer may lead to a delay if not pulled frequently.
    winsize : int
        To extract the latest winsize samples from the buffer [secs].
    """

    def __init__(self, streamInfo, bufsize=1, winsize=1):

        super().__init__(streamInfo, bufsize, winsize)

        self.blocking = False
        self._blocking_time = np.Inf

    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.
        """
        chunk, tslist = super().acquire()

        # Fill its buffer
        self.buffer.fill(chunk, tslist)


class StreamEEG(_Stream):
    """
    Class representing a receiver's EEG stream.

    This stream is instanciated as blocking.

    Parameters
    -----------
    streamInfo : LSL StreamInfo
        Contain all the info from the LSL stream to connect to.
    bufsize : int
        The buffer's size [secs]. 1-day is the maximum size.
        Large buffer may lead to a delay if not pulled frequently.
    winsize : int
        To extract the latest winsize samples from the buffer [secs].
    """

    def __init__(self, streamInfo, bufsize=1, winsize=1):

        super().__init__(streamInfo, bufsize, winsize)

        # self._multiplier = 10 ** -6  # # change uV -> V unit
        self._multiplier = 1

    def _create_ch_name_list(self):
        """
        Create the channel info.

        Trigger channel will always move to the first position.
        """
        super()._create_ch_name_list()

        self._find_lsl_trig_ch()
        self._lsl_eeg_channels = list(range(len(self.ch_list)))

        if self._lsl_tr_channel is not None:
            self._ch_list.pop(self._lsl_tr_channel)
            self._lsl_eeg_channels.pop(self._lsl_tr_channel)

        self._ch_list = ['TRIGGER'] + self._ch_list

    def _find_lsl_trig_ch(self):
        """
        Look for the trigger channel index at the LSL inlet level.
        """
        if 'USBamp' in self.name:
            self._lsl_tr_channel = 16

        elif 'BioSemi' in self.name:
            self._lsl_tr_channel = 0

        elif 'SmartBCI' in self.name:
            self._lsl_tr_channel = 23

        elif 'openvibeSignal' in self.name:
            self._multiplier = 10E6
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)

        elif 'openvibeMarkers' in self.name:
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)

        elif 'actiCHamp' in self.name:
            self._lsl_tr_channel = -1

        else:
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)

    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.
        """
        chunk, tslist = super().acquire()

        if not chunk:
            return

        data = np.array(chunk)

        # BioSemi has pull-up resistor instead of pull-down
        if 'BioSemi' in self.name and self._lsl_tr_channel is not None:
            datatype = data.dtype
            data[:, self._lsl_tr_channel] = (
                np.bitwise_and(255, data[:, self._lsl_tr_channel].astype(int))
                - 1).astype(datatype)

        # multiply values (to change unit)
        if self._multiplier != 1:
            data[:, self._lsl_eeg_channels] *= self._multiplier

        if self._lsl_tr_channel is not None:
            # move trigger channel to 0 and add back to the buffer
            data = np.concatenate((data[:, self._lsl_tr_channel].reshape(-1, 1),
                                   data[:, self._lsl_eeg_channels]), axis=1)
        else:
            # add an empty channel with zeros to channel 0
            data = np.concatenate((np.zeros((data.shape[0], 1)),
                                   data[:, self._lsl_eeg_channels]), axis=1)

        data = data.tolist()

        # Fill its buffer
        self.buffer.fill(data, tslist)
