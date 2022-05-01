import math
import time
from abc import ABC, abstractmethod

import numpy as np

from ._buffer import Buffer
from ..externals import pylsl
from ..utils import Timer
from ..utils._logs import logger
from ..utils import find_event_channel
from ..utils.lsl import lsl_channel_list
from ..utils._checks import _check_type
from ..utils._docs import fill_doc, copy_doc


_MAX_PYLSL_STREAM_BUFSIZE = 10  # max 10 sec of data buffered for LSL
MAX_BUF_SIZE = 86400  # 24h max buffer length
HIGH_LSL_OFFSET_THRESHOLD = 0.1  # Threshold above which the offset is high


@fill_doc
class _Stream(ABC):
    """
    Abstract class representing a base StreamReceiver stream.

    Parameters
    ----------
    %(receiver_streamInfo)s
    %(receiver_bufsize)s
    %(receiver_winsize)s
    """

    @abstractmethod
    def __init__(self, streamInfo, bufsize, winsize):
        winsize = _Stream._check_winsize(winsize)
        bufsize = _Stream._check_bufsize(bufsize, winsize)

        self._streamInfo = streamInfo
        self._sample_rate = self._streamInfo.nominal_srate()
        self._lsl_bufsize = min(_MAX_PYLSL_STREAM_BUFSIZE, bufsize)

        if self._sample_rate is not None:
            samples_per_sec = self._sample_rate
            # max_buflen: seconds
            self._inlet = pylsl.StreamInlet(
                streamInfo, max_buflen=math.ceil(self._lsl_bufsize)
            )
        else:
            samples_per_sec = 100
            # max_buflen: samples x100
            self._inlet = pylsl.StreamInlet(
                streamInfo,
                max_buflen=math.ceil(self._lsl_bufsize * samples_per_sec),
            )
        self._inlet.open_stream()

        self._extract_stream_info()
        self._create_ch_name_list()
        self._lsl_time_offset = None

        self._watchdog = Timer()
        self._blocking = True
        self._blocking_time = 5

        # Convert to samples
        bufsize = _Stream._convert_sec_to_samples(bufsize, samples_per_sec)
        winsize = _Stream._convert_sec_to_samples(winsize, samples_per_sec)
        self._lsl_bufsize = _Stream._convert_sec_to_samples(
            self._lsl_bufsize, samples_per_sec
        )
        self._buffer = Buffer(bufsize, winsize)

    def _create_ch_name_list(self):
        """
        Create the channels' name list.
        """
        self._ch_list = lsl_channel_list(self._inlet)

        if not self._ch_list:
            self._ch_list = [
                f"ch_{i+1}" for i in range(self._streamInfo.channel_count())
            ]

    def _extract_stream_info(self):
        """
        Extract the name, serial number and if it's a slave.
        """
        self._name = self._streamInfo.name()
        self._serial = (
            self._inlet.info()
            .desc()
            .child("acquisition")
            .child_value("serial_number")
        )

        if self._serial == "":
            self._serial = "N/A"

        self._is_slave = (
            self._inlet.info()
            .desc()
            .child("amplifier")
            .child("settings")
            .child("is_slave")
            .first_child()
            .value()
            == "true"
        )

    def show_info(self):
        """
        Display the stream's info.
        """
        logger.info(
            f"Server: {self._name}({self._serial}) "
            f"/ type:{self._streamInfo.type()} "
            f"@ {self._streamInfo.hostname()} "
            f"(v{self._streamInfo.version()})."
        )
        logger.info("Source sampling rate: %s", self._sample_rate)
        logger.info("Channels: %s", self._streamInfo.channel_count())

        # Check for high LSL offset
        if self._lsl_time_offset is None:
            logger.info(
                "No LSL timestamp offset computed, no data received yet."
            )
        elif abs(self._lsl_time_offset) > HIGH_LSL_OFFSET_THRESHOLD:
            logger.warning(
                f"LSL server {self._name}({self._serial}) has a high "
                f"timestamp offset [offset={self._lsl_time_offset:.3f}]."
            )
        else:
            logger.info(
                f"LSL server {self._name}({self._serial}) synchronized "
                f"[offset={self._lsl_time_offset:.3f}]"
            )

    @abstractmethod
    @fill_doc
    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.

        Returns
        -------
        chunk : list
            Data [samples x channels]
        %(receiver_tslist)s
        """
        chunk = []
        tslist = []
        received = False
        self._watchdog.reset()

        # Acquire the data
        while not received:
            while self._watchdog.sec() < self._blocking_time:
                if len(tslist) == 0:
                    chunk, tslist = self._inlet.pull_chunk(
                        timeout=0.0, max_samples=self._lsl_bufsize
                    )
                    if not self._blocking and len(tslist) == 0:
                        received = True
                        break
                if len(tslist) > 0:
                    if self._lsl_time_offset is None:  # First acquisition
                        self._compute_offset(tslist)
                    received = True
                    tslist = self._correct_lsl_offset(tslist)
                    break
                time.sleep(0.0005)  # TODO: test without
            else:
                # Give up and return empty values to avoid deadlock
                logger.error(
                    "Timeout occurred [%ssecs] while acquiring data "
                    "from %s(%s). ",
                    self._blocking_time,
                    self._name,
                    self._serial,
                )
                received = True

        return chunk, tslist

    @fill_doc
    def _compute_offset(self, tslist):
        """
        Compute the LSL offset coming from some devices.

        It has to be called just after acquiring the data/timestamps in order
        to be valid.

        Parameters
        ----------
        %(receiver_tslist)s
        """
        self._lsl_time_offset = tslist[-1] - pylsl.local_clock()
        return self._lsl_time_offset

    @fill_doc
    def _correct_lsl_offset(self, tslist):
        """
        Correct the timestamps if there is a high LSL offset.

        Parameters
        ----------
        %(receiver_tslist)s
        """
        if abs(self._lsl_time_offset) > HIGH_LSL_OFFSET_THRESHOLD:
            tslist = [t - self._lsl_time_offset for t in tslist]

        return tslist

    # --------------------------------------------------------------------
    @staticmethod
    @fill_doc
    def _check_winsize(winsize):
        """
        Check that winsize is positive.

        Parameters
        ----------
        %(receiver_winsize)s
        """
        _check_type(winsize, ("numeric",), item_name="winsize")
        if winsize <= 0:
            raise ValueError("Invalid window size %s." % winsize)

        return winsize

    @staticmethod
    @fill_doc
    def _check_bufsize(bufsize, winsize):
        """
        Check that bufsize is positive and bigger than winsize.

        Parameters
        ----------
        %(receiver_bufsize)s
        %(receiver_winsize)s
        """
        _check_type(bufsize, ("numeric",), item_name="bufsize")

        if bufsize <= 0 or bufsize > MAX_BUF_SIZE:
            logger.error(
                "Improper buffer size %.1f. Setting to %.1f.",
                bufsize,
                MAX_BUF_SIZE,
            )
            bufsize = MAX_BUF_SIZE
        elif bufsize < winsize:
            logger.error(
                "Buffer size %.1f is smaller than window size. Setting to "
                "%.1f.",
                bufsize,
                winsize,
            )
            bufsize = winsize

        return bufsize

    @staticmethod
    @fill_doc
    def _convert_sec_to_samples(bufsec, sample_rate):
        """
        Convert a buffer's/window's size from sec to samples.

        Parameters
        ----------
        %(receiver_bufsize)s
        sample_rate : float
             Sampling rate of the LSL server.
             If irregular sampling rate, empirical number of samples per sec.

        Returns
        -------
        int
            Buffer_size's size [samples].
        """
        return round(bufsec * sample_rate)

    # --------------------------------------------------------------------
    @property
    def streamInfo(self):
        """
        Stream info received from the LSL inlet.
        """
        return self._streamInfo

    @property
    def sample_rate(self):
        """
        Stream's sampling rate.
        """
        return self._sample_rate

    @property
    def name(self):
        """
        Stream's name.
        """
        return self._name

    @property
    def serial(self):
        """
        Stream's serial number.
        """
        return self._serial

    @property
    def is_slave(self):
        """
        Value stored in LSL ['amplifier']['settings']['is_slave'].
        """
        return self._is_slave

    @property
    def ch_list(self):
        """
        Channels' name list.
        """
        return self._ch_list

    @property
    def lsl_time_offset(self):
        """
        The difference between the local and the stream's LSL clocks, used for
        timestamps correction.

        Some LSL servers (like OpenVibe) often have a bug of sending its own
        running time instead of LSL time.
        """
        return self._lsl_time_offset

    @property
    def blocking(self):
        """
        If True, the stream wait to receive data.

        :setter: Change the blocking status.
        :type: bool
        """
        return self._blocking

    @blocking.setter
    def blocking(self, blocking):
        self._blocking = bool(blocking)

    @property
    def blocking_time(self):
        """
        If blocking is True, how long to wait to receive data in seconds.

        :setter: Change the blocking time duration (seconds).
        :type: float
        """
        return self._blocking_time

    @blocking_time.setter
    def blocking_time(self, blocking_time):
        self._blocking_time = float(blocking_time)

    @property
    def buffer(self):
        """
        Buffer containing the data and the timestamps.
        """
        return self._buffer


@fill_doc
class StreamMarker(_Stream):
    """
    Class representing a StreamReceiver markers stream.

    Notice the irregular sampling rate.
    This stream is instanciated as non-blocking.

    Parameters
    -----------
    %(receiver_streamInfo)s
    %(receiver_bufsize)s
    %(receiver_winsize)s
    """

    def __init__(self, streamInfo, bufsize=1, winsize=1):
        super().__init__(streamInfo, bufsize, winsize)

        self._blocking = False
        self._blocking_time = float("inf")

    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.
        """
        chunk, tslist = super().acquire()
        self._buffer.fill(chunk, tslist)  # Fill its buffer


@fill_doc
class StreamEEG(_Stream):
    """
    Class representing a StreamReceiver EEG stream.

    This stream is instanciated as blocking.

    Parameters
    -----------
    %(receiver_streamInfo)s
    %(receiver_bufsize)s
    %(receiver_winsize)s
    """

    def __init__(self, streamInfo, bufsize=1, winsize=1):
        super().__init__(streamInfo, bufsize, winsize)

        # self._scaling_factor = 10 ** -6  # change uV -> V unit
        self._scaling_factor = 1

    @copy_doc(_Stream._create_ch_name_list)
    def _create_ch_name_list(self):
        """
        Trigger channel will always move to the first position.
        """
        super()._create_ch_name_list()

        self._find_lsl_trig_ch()
        self._lsl_eeg_channels = list(range(len(self._ch_list)))

        if self._lsl_tr_channel is not None:
            self._ch_list.pop(self._lsl_tr_channel)
            self._lsl_eeg_channels.pop(self._lsl_tr_channel)

        self._ch_list = ["TRIGGER"] + self._ch_list

    def _find_lsl_trig_ch(self):
        """
        Look for the trigger channel index at the LSL inlet level.
        """
        if "USBamp" in self._name:
            self._lsl_tr_channel = 16

        elif "BioSemi" in self._name:
            self._lsl_tr_channel = 0

        elif "SmartBCI" in self._name:
            self._lsl_tr_channel = 23

        elif "openvibeSignal" in self._name:
            # TODO: Test if this is correct or should be 1E6
            self._scaling_factor = 10e6
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)

        elif "openvibeMarkers" in self._name:
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)

        elif "actiCHamp" in self._name:
            self._lsl_tr_channel = -1

        else:
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)

        # TODO: patch to be improved for multi-trig channel recording
        if isinstance(self._lsl_tr_channel, list):
            self._lsl_tr_channel = self._lsl_tr_channel[0]

        if self._lsl_tr_channel is not None:
            logger.debug("Trigger channel idx: %d", self._lsl_tr_channel)
        else:
            logger.debug("Trigger channel was not found.")

    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.
        """
        chunk, tslist = super().acquire()

        if len(chunk) == 0:
            return

        # TODO: Is it not more efficient to keep working on lists?
        data = np.array(chunk)

        # BioSemi has pull-up resistor instead of pull-down
        if "BioSemi" in self._name and self._lsl_tr_channel is not None:
            datatype = data.dtype
            data[:, self._lsl_tr_channel] = (
                np.bitwise_and(
                    255, data[:, self._lsl_tr_channel].astype(int, copy=False)
                )
                - 1
            ).astype(datatype, copy=False)

        # multiply values (to change unit)
        if self._scaling_factor != 1:
            data[:, self._lsl_eeg_channels] *= self._scaling_factor

        if self._lsl_tr_channel is not None:
            # move trigger channel to 0 and add back to the buffer
            data = np.concatenate(
                (
                    data[:, self._lsl_tr_channel].reshape(-1, 1),
                    data[:, self._lsl_eeg_channels],
                ),
                axis=1,
            )
        else:
            # add an empty channel with zeros to channel 0
            data = np.concatenate(
                (
                    np.zeros((data.shape[0], 1)),
                    data[:, self._lsl_eeg_channels],
                ),
                axis=1,
            )

        data = data.tolist()

        # Fill its buffer
        self._buffer.fill(data, tslist)

    # --------------------------------------------------------------------
    @property
    def scaling_factor(self):
        """
        Scaling factor applied to the data to convert to the desired unit.

        :setter: Change the scaling factor applied to the data.
        :type: float
        """
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, scaling_factor):
        _check_type(scaling_factor, ("numeric",), item_name="scaling_factor")
        if scaling_factor <= 0:
            raise ValueError(
                "Property scaling_factor must be a strictly "
                "positive number. Provided: %s" % scaling_factor
            )
        self._scaling_factor = scaling_factor
