import math
from abc import ABC, abstractmethod

from ...utils._docs import fill_doc
from ...utils._logs import logger

_BUFFER_DURATION = 30  # seconds


@fill_doc
class _Scope(ABC):
    """
    Class representing a base scope.

    Parameters
    ----------
    %(viewer_scope_stream_receiver)s
    %(viewer_scope_stream_name)s
    """

    # ---------------------------- INIT ----------------------------
    @abstractmethod
    def __init__(self, stream_receiver, stream_name):
        assert stream_name in stream_receiver.streams
        self._sr = stream_receiver
        self._stream_name = stream_name

        # Infos from stream
        self._sample_rate = int(
            self._sr.streams[self._stream_name].sample_rate
        )

        # Variables
        self._duration_buffer = _BUFFER_DURATION
        self._duration_buffer_samples = math.ceil(
            _BUFFER_DURATION * self._sample_rate
        )

        # Buffers
        self._ts_list = list()

        logger.debug("Scope connected to %s", self._stream_name)
        logger.debug("Data sample rate is %f", self._sample_rate)
        logger.debug(
            "Scope buffer duration is %d seconds", self._duration_buffer
        )

    # -------------------------- Main Loop -------------------------
    @abstractmethod
    def update_loop(self):
        """
        Main update loop acquiring data from the LSL stream and filling the
        scope's buffer.
        """
        pass

    @abstractmethod
    def _read_lsl_stream(self):
        """
        Acquires data from the connected LSL stream.
        """
        self._sr.acquire()
        self._data_acquired, self._ts_list = self._sr._get_buffer()
        self._sr.reset_buffer()

        if len(self._ts_list) > 0:
            logger.debug("Signal acquired by the scope.")

    # --------------------------------------------------------------------
    @property
    def stream_name(self):
        """
        Name of the connected stream.
        """
        return self._stream_name

    @property
    def sample_rate(self):
        """
        Sample rate of the connected stream [Hz].
        """
        return self._sample_rate

    @property
    def duration_buffer(self):
        """
        Duration of the scope's buffer [seconds].
        """
        return self._duration_buffer

    @property
    def duration_buffer_samples(self):
        """
        Duration of the scope's buffer [samples].
        """
        return self._duration_buffer_samples

    @property
    def ts_list(self):
        """
        Timestamps buffer [samples, ].
        """
        return self._ts_list
