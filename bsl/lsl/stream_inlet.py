from ctypes import byref, c_double, c_int, c_void_p
from typing import Optional

from ..utils._checks import _check_type
from ..utils._docs import copy_doc
from .constants import (
    cf_string,
    fmt2pull_chunk,
    fmt2pull_sample,
    fmt2string,
    fmt2type,
)
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo
from .utils import free_char_p_array_memory, handle_error


class StreamInlet:
    def __init__(
        self,
        sinfo: _BaseStreamInfo,
        max_buflen=360,
        max_chunklen=0,
        recover=True,
        processing_flags=0,
    ):
        _check_type(sinfo, (_BaseStreamInfo,), "sinfo")
        self.obj = lib.lsl_create_inlet(
            sinfo.obj, max_buflen, max_chunklen, recover
        )
        self.obj = c_void_p(self.obj)
        if not self.obj:
            raise RuntimeError("The StreamInlet could not be created.")

        # set preprocessing of the inlet
        if processing_flags > 0:
            handle_error(
                lib.lsl_set_postprocessing(self.obj, processing_flags)
            )

        # properties from the StreamInfo
        self._channel_format = sinfo._channel_format
        self._name = sinfo.name
        self._n_channels = sinfo.n_channels
        self._sfreq = sinfo.sfreq
        self._stype = sinfo.stype

        # properties from the inlet
        self._do_pull_sample = fmt2pull_sample[self._channel_format]
        self._do_pull_chunk = fmt2pull_chunk[self._channel_format]
        self._value_type = fmt2type[self._channel_format]
        self._sample_type = self._value_type * self._n_channels
        self._sample = self._sample_type()  # required
        self._buffers = {}

    def __del__(self):
        """Destroy a `~bsl.lsl.StreamInlet`.

        The inlet will automatically disconnect.
        """
        try:
            lib.lsl_destroy_inlet(self.obj)
        except Exception:
            pass

    def open_stream(self, timeout: Optional[float] = None) -> None:
        timeout = StreamInlet._check_timeout(timeout)
        errcode = c_int()
        lib.lsl_open_stream(self.obj, c_double(timeout), byref(errcode))
        handle_error(errcode)

    def close_stream(self) -> None:
        lib.lsl_close_stream(self.obj)

    def time_correction(self, timeout: Optional[float] = None) -> float:
        timeout = StreamInlet._check_timeout(timeout)
        errcode = c_int()
        result = lib.lsl_time_correction(
            self.obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return result

    def pull_sample(self, timeout: Optional[float] = None, sample=None):
        timeout = StreamInlet._check_timeout(timeout)

        errcode = c_int()
        timestamp = self._do_pull_sample(
            self.obj,
            byref(self._sample),
            self._n_channels,
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)

        if timestamp:
            sample = [v for v in self._sample]
            if self._channel_format == cf_string:
                sample = [v.decode("utf-8") for v in sample]
            return sample, timestamp
        else:
            return None, None

    def pull_chunk(
        self, timeout: Optional[float] = 0.0, max_samples=1024, dest_obj=None
    ):
        timeout = StreamInlet._check_timeout(timeout)
        # look up a pre-allocated buffer of appropriate length
        num_channels = self._n_channels
        max_values = max_samples * num_channels

        if max_samples not in self._buffers:
            self._buffers[max_samples] = (
                (self._value_type * max_values)(),
                (c_double * max_samples)(),
            )
        if dest_obj is not None:
            data_buff = (self._value_type * max_values).from_buffer(dest_obj)
        else:
            data_buff = self._buffers[max_samples][0]
        ts_buff = self._buffers[max_samples][1]

        # read data into the buffer
        errcode = c_int()
        num_elements = self._do_pull_chunk(
            self.obj,
            byref(data_buff),
            byref(ts_buff),
            max_values,
            max_samples,
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)
        # return results (note: could offer a more efficient format in the
        # future, e.g., a numpy array)
        num_samples = num_elements / num_channels
        if dest_obj is None:
            samples = [
                [data_buff[s * num_channels + c] for c in range(num_channels)]
                for s in range(int(num_samples))
            ]
            if self._channel_format == cf_string:
                samples = [[v.decode("utf-8") for v in s] for s in samples]
                free_char_p_array_memory(data_buff, max_values)
        else:
            samples = None
        timestamps = [ts_buff[s] for s in range(int(num_samples))]
        return samples, timestamps

    def samples_available(self) -> int:
        """Query whether samples are currently available on the Outlet.

        Note that it is not a good idea to use this method to determine if a
        pull call would block. Instead, set the pull timeout to 0.0 or an
        acceptably low value.

        Returns
        -------
        n_samples : int
            Number of available samples.
        """
        return lib.lsl_samples_available(self.obj)

    def flush(self) -> int:
        """Drop all queued and not-yet pulled samples.

        Returns
        -------
        n_dropped : int
            Number of dropped samples.
        """
        return lib.lsl_inlet_flush(self.obj)

    def was_clock_reset(self) -> bool:
        """Query if the clock was potentially reset since the last call."""
        return bool(lib.lsl_was_clock_reset(self.obj))

    # -------------------------------------------------------------------------
    @copy_doc(_BaseStreamInfo.channel_format)
    @property
    def channel_format(self) -> str:
        return fmt2string[self._channel_format]

    @copy_doc(_BaseStreamInfo.n_channels)
    @property
    def n_channels(self) -> int:
        return self._n_channels

    @copy_doc(_BaseStreamInfo.name)
    @property
    def name(self) -> str:
        return self._name

    @copy_doc(_BaseStreamInfo.sfreq)
    @property
    def sfreq(self) -> float:
        return self._sfreq

    @copy_doc(_BaseStreamInfo.stype)
    @property
    def stype(self) -> str:
        return self._stype

    # -------------------------------------------------------------------------
    def get_sinfo(self, timeout: Optional[float] = None) -> _BaseStreamInfo:
        """`~bsl.lsl.StreamInfo` corresponding to this Inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.
        """
        timeout = StreamInlet._check_timeout(timeout)
        errcode = c_int()
        result = lib.lsl_get_fullinfo(
            self.obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return _BaseStreamInfo(result)

    # -------------------------------------------------------------------------
    @staticmethod
    def _check_timeout(timeout: Optional[float]) -> float:
        """Check that the provided timeout is valid.

        Parameters
        ----------
        timeout : float | None
            Timeout (in seconds) or None to disable timeout.

        Returns
        -------
        timeout : float
            Timeout (in seconds). If None was provided, a very large float is
            provided.
        """
        # with _check_type, the execution takes 800-900 ns.
        # with the try/except below, the execution takes 110 ns.
        if timeout is None:
            return 32000000.0  # about 1 year
        try:
            raise_ = timeout < 0
        except Exception:
            raise TypeError(
                "The argument 'timeout' must be a strictly positive number."
            )
        if raise_:
            raise ValueError(
                "The argument 'timeout' must be a strictly positive number. "
                f"{timeout} is invalid."
            )
        return timeout
