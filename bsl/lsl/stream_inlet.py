from ctypes import byref, c_double, c_int, c_void_p
from typing import Optional

import numpy as np

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
from .utils import _free_char_p_array_memory, handle_error


class StreamInlet:
    """An inlet to retrieve data and metadata on the network.

    Parameters
    ----------
    sinfo : StreamInfo
        The `~bsl.lsl.StreamInfo` object describing the stream. Stays constant
        over the lifetime of the inlet.
    """

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
        """Subscribe to a data stream.

        All samples pushed in at the other end from this moment onwards will be
        queued and eventually be delivered in response to
        `~bsl.lsl.StreamInlet.pull_sample` or `~bsl.lsl.StreamInlet.pull_chunk`
        calls. Pulling a sample without subscribing to the stream with this
        method is permitted (the stream will be opened implicitly).

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.
        """
        timeout = StreamInlet._check_timeout(timeout)
        errcode = c_int()
        lib.lsl_open_stream(self.obj, c_double(timeout), byref(errcode))
        handle_error(errcode)

    def close_stream(self) -> None:
        """Drop the current data stream.

        All samples that are still buffered or in flight will be dropped and
        transmission and buffering of data for this inlet will be stopped.
        This method is used if an application stops being interested in data
        from a source (temporarily or not) but keeps the outlet alive, to not
        waste unnecessary system and network resources.
        """
        lib.lsl_close_stream(self.obj)

    def time_correction(self, timeout: Optional[float] = None) -> float:
        """Retrieve an estimated time correction offset for the given stream.

        The first call to this function takes several milliseconds until a
        reliable first estimate is obtained. Subsequent calls are instantaneous
        (and rely on periodic background updates). The precision of these
        estimates should be below 1 ms (empirically within +/-0.2 ms).

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.

        Returns
        -------
        time_correction : float
            Current estimate of the time correction. This number needs to be
            added to a timestamp that was remotely generated via
            ``local_clock()`` to map it into the local clock domain of the
            client machine.
        """
        timeout = StreamInlet._check_timeout(timeout)
        errcode = c_int()
        result = lib.lsl_time_correction(
            self.obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return result

    def pull_sample(self, timeout: Optional[float] = None):
        """Pull a single sample from the inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.

        Returns
        -------
        sample : list | array of shape (n_channels,) | None
            If the channel format is ``'string^``, returns a list of values for
            each channel. Else, returns a numpy array of shape
            ``(n_channels,)``.
        timestamp : float | None
            Acquisition timestamp on the remote machine.

        Notes
        -----
        To map the timestamp to the local clock of the client machine, add the
        estimated time correction returned by
        `~bsl.lsl.StreamInlet.time_correction`.
        """
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
            if self._channel_format == cf_string:
                sample = [v.decode("utf-8") for v in self._sample]
            else:
                sample = np.array(self._sample)
        else:
            sample = None
            timestamp = None
        return sample, timestamp

    def pull_chunk(
        self,
        timeout: Optional[float] = None,
        n_samples: int = 1024,
        dest_obj=None,
    ):
        """Pull a chunk of samples from the inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.
        max_samples : int
            Number of samples to return. The function is blocking until this
            number of samples is available.
        dest_obj : Buffer
            A python object that supports the buffer interface. If this
            argument is provided, the the destination object will be updated
            in place and the samples returned by this method will be None.
            The timestamps are returned regardless of this argument.
            A numpy buffer must use ``order='C'``.

        Returns
        -------
        samples : list of list | array of shape (n_channels, n_samples) |
            If the channel format is ``'string^``, returns a list of list of
            values for each channel and sample. Else, returns a numpy array of
            shape ``(n_channels, n_samples)``.
        timestamps : array of shape (n_samples,) | None
            Acquisition timestamps on the remote machine.
        """
        timeout = StreamInlet._check_timeout(timeout)
        if not isinstance(n_samples, int):
            n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(
                "The argument 'n_samples' must be a strictly positive "
                f"integer. {n_samples} is invalid."
            )
        # look up a pre-allocated buffer of appropriate length
        max_values = n_samples * self._n_channels

        # create buffer
        if n_samples not in self._buffers:
            self._buffers[n_samples] = (
                (self._value_type * max_values)(),  # data
                (c_double * n_samples)(),  # timestamps
            )
        if dest_obj is None:
            data_buffer = self._buffers[n_samples][0]
        else:
            data_buffer = (self._value_type * max_values).from_buffer(dest_obj)
        ts_buffer = self._buffers[n_samples][1]

        # read data into the buffer
        errcode = c_int()
        num_elements = self._do_pull_chunk(
            self.obj,
            byref(data_buffer),
            byref(ts_buffer),
            max_values,
            n_samples,
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)
        # return results (note: could offer a more efficient format in the
        # future, e.g., a numpy array)
        num_samples = num_elements / self._n_channels
        if dest_obj is None:
            if self._channel_format == cf_string:
                samples = [
                    [
                        data_buffer[s * self._n_channels + c]
                        for c in range(self._n_channels)
                    ]
                    for s in range(int(num_samples))
                ]
                samples = [[v.decode("utf-8") for v in s] for s in samples]
                _free_char_p_array_memory(data_buffer, max_values)
            else:
                # this is 400-500x faster than the list approach
                samples = (
                    np.frombuffer(data_buffer, dtype=self._value_type)
                    .reshape(-1, self._n_channels)
                    .T
                )

        else:
            samples = None

        # %timeit [ts_buff[s] for s in range(int(num_samples))]
        # 68.8 µs ± 635 ns per loop
        # %timeit np.array(ts_buff)
        # 854 ns ± 2.99 ns per loop
        # %timeit np.frombuffer(ts_buff)
        # 192 ns ± 1.11 ns per loop
        timestamps = np.frombuffer(ts_buffer)  # requires numpy ≥ 1.20
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
