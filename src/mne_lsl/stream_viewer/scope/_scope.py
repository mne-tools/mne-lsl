import math
from abc import ABC, abstractmethod

from ...utils.logs import logger

_BUFFER_DURATION = 30  # seconds


class _Scope(ABC):
    """Class representing a base scope.

    Parameters
    ----------
    inlet : StreamInlet
    """

    # ---------------------------- INIT ----------------------------
    @abstractmethod
    def __init__(self, inlet):
        self._inlet = inlet
        self._stream_name = self._inlet.get_sinfo().name

        # Infos from stream
        self._sample_rate = self._inlet.get_sinfo().sfreq

        # Variables
        self._duration_buffer = _BUFFER_DURATION
        self._duration_buffer_samples = math.ceil(_BUFFER_DURATION * self._sample_rate)

        # Buffers
        self._ts_list = list()

        logger.debug("Scope connected to %s", self._stream_name)
        logger.debug("Data sample rate is %f", self._sample_rate)
        logger.debug("Scope buffer duration is %d seconds", self._duration_buffer)

    # -------------------------- Main Loop -------------------------
    @abstractmethod
    def update_loop(self):  # noqa
        """Update loop acquiring data from the stream and filling the scope's buffer."""

    @abstractmethod
    def _read_lsl_stream(self):
        """Acquires data from the connected LSL stream."""
        # (n_samples, n_channels)
        self._data_acquired, self._ts_list = self._inlet.pull_chunk()

        if len(self._ts_list) > 0:
            logger.debug("Signal acquired by the scope.")

    # --------------------------------------------------------------------
    @property
    def stream_name(self):
        """Name of the connected stream."""
        return self._stream_name

    @property
    def sample_rate(self):
        """Sample rate of the connected stream [Hz]."""
        return self._sample_rate

    @property
    def duration_buffer(self):
        """Duration of the scope's buffer [seconds]."""
        return self._duration_buffer

    @property
    def duration_buffer_samples(self):
        """Duration of the scope's buffer [samples]."""
        return self._duration_buffer_samples

    @property
    def ts_list(self):
        """Timestamps buffer [samples, ]."""
        return self._ts_list
