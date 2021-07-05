import math
from abc import ABC, abstractmethod

_BUFFER_DURATION = 30  # seconds


class _Scope(ABC):
    """
    Class representing a base scope.

    Parameters
    ----------
    stream_receiver : neurodecode.stream_receiver.StreamReceiver
        The connected stream receiver.
    stream_name : str
        The stream to connect to.
    """

    # ---------------------------- INIT ----------------------------
    @abstractmethod
    def __init__(self, stream_receiver, stream_name):
        assert stream_name in stream_receiver.streams
        self._sr = stream_receiver
        self._stream_name = stream_name

        # Infos
        self._sample_rate = int(
            self._sr.streams[self._stream_name].sample_rate)

        # Buffers
        self._duration_buffer = _BUFFER_DURATION
        self._n_samples_buffer = math.ceil(_BUFFER_DURATION * self._sample_rate)
        self.ts_list = list()

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
        self._data_acquired, self.ts_list = self._sr.get_buffer()
        self._sr.reset_buffer()

        if len(self.ts_list) == 0:
            return

    # --------------------------------------------------------------------
    @property
    def stream_name(self):
        """
        The name of the connected stream.
        """
        return self._stream_name

    @property
    def sample_rate(self):
        """
        The sample rate of the connected stream.
        """
        return self._sample_rate

    @property
    def duration_buffer(self):
        """
        The duration of the scope's buffer.
        """
        return self._duration_buffer
