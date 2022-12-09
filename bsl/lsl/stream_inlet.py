from ctypes import byref, c_double, c_int, c_void_p
from functools import reduce
from typing import List, Optional, Union

import numpy as np

from ..utils._checks import _check_type, _check_value
from ..utils._docs import copy_doc
from .constants import (
    cf_string,
    fmt2pull_chunk,
    fmt2pull_sample,
    fmt2string,
    fmt2type,
    post_processing_flags,
)
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo
from .utils import _check_timeout, _free_char_p_array_memory, handle_error


class StreamInlet:
    """An inlet to retrieve data and metadata on the network.

    Parameters
    ----------
    sinfo : StreamInfo
        Description of the stream to connect to.
    chunk_size : int ``≥ 1`` | ``0``
        The desired chunk granularity in samples. By default, the
        ``chunk_size`` defined by the sender (outlet) is used.
    max_buffered : float ``≥ 0``
        The maximum amount of data to buffer in the Outlet.
        The number of samples buffered is ``max_buffered * 100`` if the
        sampling rate is irregular, else it's ``max_buffered`` seconds.
    recover : bool
        Attempt to silently recover lost streams that are recoverable (requires
        a ``source_id`` to be specified in the `~bsl.lsl.StreamInfo`).
    processing_flags : list of str | ``'all'`` | None
        Set the post-processing options. By default, post-processing is
        disabled. Any combination of the processing flags is valid. The
        available flags are:
        - ``'clocksync'``: Automatic clock synchronization, equivalent to
          manually adding the estimated `~bsl.lsl.StreamInlet.time_correction`.
        - ``'dejitter'``: Remove jitter on the received timestamps with a
          smoothing algorithm.
        - ``'monotize'``: Force the timestamps to be monotically ascending.
          This option should not be enable if ``'dejitter'`` is not enabled.
        - ``'threadsafe'``: Post-processing is thread-safe, thus the same
          inlet can be read from multiple threads.
    """

    def __init__(
        self,
        sinfo: _BaseStreamInfo,
        chunk_size: int = 0,
        max_buffered: float = 360,
        recover: bool = True,
        processing_flags: Optional[Union[str, List[str]]] = None,
    ):
        _check_type(sinfo, (_BaseStreamInfo,), "sinfo")
        _check_type(chunk_size, ("int",), "chunk_size")
        if chunk_size < 0:
            raise ValueError(
                "The argument 'chunk_size' must contain a positive integer. "
                f"{chunk_size} is invalid."
            )
        _check_type(max_buffered, ("numeric",), "max_buffered")
        if max_buffered < 0:
            raise ValueError(
                "The argument 'max_buffered' must contain a positive number. "
                f"{max_buffered} is invalid."
            )
        _check_type(recover, (bool,), "recover")

        self._obj = lib.lsl_create_inlet(
            sinfo._obj, max_buffered, chunk_size, recover
        )
        self._obj = c_void_p(self._obj)
        if not self._obj:
            raise RuntimeError("The StreamInlet could not be created.")

        # set preprocessing of the inlet
        if processing_flags is not None:
            _check_type(processing_flags, (list, str), "processing_flags")
            if isinstance(processing_flags, str):
                _check_value(processing_flags, ("all",), "processing_flags")
                processing_flags = reduce(
                    lambda x, y: x | y, post_processing_flags.values()
                )
            else:
                for flag in processing_flags:
                    _check_type(flag, (str,), "processing_flag")
                    _check_value(flag, post_processing_flags, flag)
                # bitwise OR between the flags
                processing_flags = reduce(lambda x, y: x | y, processing_flags)
            assert processing_flags > 0  # sanity-check
            handle_error(
                lib.lsl_set_postprocessing(self._obj, processing_flags)
            )

        # properties from the StreamInfo
        self._dtype = sinfo._dtype
        self._name = sinfo.name
        self._n_channels = sinfo.n_channels
        self._sfreq = sinfo.sfreq
        self._stype = sinfo.stype

        # inlet properties
        self._do_pull_sample = fmt2pull_sample[self._dtype]
        self._do_pull_chunk = fmt2pull_chunk[self._dtype]
        self._value_type = fmt2type[self._dtype]
        self._buffer_data = {1: (self._value_type * self._n_channels)()}
        self._buffer_ts = {}

    def __del__(self):
        """Destroy a `~bsl.lsl.StreamInlet`.

        The inlet will automatically disconnect.
        """
        try:
            lib.lsl_destroy_inlet(self._obj)
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

        Notes
        -----
        Opening a stream is a non-blocking operation. Thus, samples pushed on
        an outlet while the stream is not yet open will be missed.
        """
        timeout = _check_timeout(timeout)
        errcode = c_int()
        lib.lsl_open_stream(self._obj, c_double(timeout), byref(errcode))
        handle_error(errcode)

    def close_stream(self) -> None:
        """Drop the current data stream.

        All samples that are still buffered or in flight will be dropped and
        transmission and buffering of data for this inlet will be stopped.
        This method is used if an application stops being interested in data
        from a source (temporarily or not) but keeps the outlet alive, to not
        waste unnecessary system and network resources.
        """
        lib.lsl_close_stream(self._obj)

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
        timeout = _check_timeout(timeout)
        errcode = c_int()
        result = lib.lsl_time_correction(
            self._obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return result

    def pull_sample(self, timeout: Optional[float] = 0.0):
        """Pull a single sample from the inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. None correspond to
            a very large value, effectively disabling the timeout. ``0.`` makes
            this function non-blocking even if no sample is available. See
            notes for additional details.
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.

        Returns
        -------
        sample : list | array of shape (n_channels,) | None
            If the channel format is ``'string^``, returns a list of values for
            each channel. Else, returns a numpy array of shape
            ``(n_channels,)``.
        timestamp : float | None
            Acquisition timestamp on the remote machine. To map the timestamp
            to the local clock of the client machine, add the estimated time
            correction return by `~bsl.lsl.StreamInlet.time_correction`.
            None if no sample was retrieved.

        Notes
        -----
        Note that if ``timeout`` is reached and no sample is available, empty
        ``sample`` arrays is returned.
        """
        timeout = _check_timeout(timeout)

        errcode = c_int()
        timestamp = self._do_pull_sample(
            self._obj,
            byref(self._buffer_data[1]),
            self._n_channels,
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)

        if timestamp:
            if self._dtype == cf_string:
                sample = [v.decode("utf-8") for v in self._buffer_data[1]]
                _free_char_p_array_memory(self._buffer_data[1])
            else:
                sample = np.frombuffer(
                    self._buffer_data[1], dtype=self._value_type
                )
        else:
            sample = None
            timestamp = None
        return sample, timestamp

    def pull_chunk(
        self,
        timeout: Optional[float] = 0.0,
        max_samples: int = 1024,
    ):
        """Pull a chunk of samples from the inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. None correspond to
            a very large value, effectively disabling the timeout. ``0.`` makes
            this function non-blocking even if no sample is available. See
            notes for additional details.
        max_samples : int
            Maximum umber of samples to return. The function is blocking until this
            number of samples is available. See notes for additional details.

        Returns
        -------
        samples : list of list | array of shape (n_samples, n_channels) | None
            If the channel format is ``'string'``, returns a list of list of
            values for each channel and sample. Each sublist represents an
            entire channel. Else, returns a numpy array of shape
            ``(n_samples, n_channels)``.
        timestamps : array of shape (n_samples,) | None
            Acquisition timestamps on the remote machine.

        Notes
        -----
        The argument ``timeout`` and ``max_samples`` control the blocking
        behavior of the pull operation. If the number of available sample is
        inferior to ``n_samples``, the pull operation is blocking until
        ``timeout`` is reached. Thus, to return all the available samples at a
        given time, regardless of the number of samples requested, ``timeout``
        must be set to ``0``.

        Note that if ``timeout`` is reached and no sample is available, empty
        ``samples`` and ``timestamps`` arrays are returned.
        """
        timeout = _check_timeout(timeout)
        if not isinstance(max_samples, int):
            max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError(
                "The argument 'max_samples' must be a strictly positive "
                f"integer. {max_samples} is invalid."
            )

        # look up or create a pre-allocated buffers of appropriate length
        max_samples_data = max_samples * self._n_channels
        if max_samples_data not in self._buffer_data:
            self._buffer_data[max_samples_data] = (
                self._value_type * max_samples_data
            )()
        if max_samples not in self._buffer_ts:
            self._buffer_ts[max_samples] = (c_double * max_samples)()
        data_buffer = self._buffer_data[max_samples_data]
        ts_buffer = self._buffer_ts[max_samples]

        # read data into it
        errcode = c_int()
        n_samples_data = self._do_pull_chunk(
            self._obj,
            byref(data_buffer),
            byref(ts_buffer),
            max_samples_data,
            max_samples,
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)

        n_samples = int(n_samples_data / self._n_channels)
        if self._dtype == cf_string:
            samples = [
                [
                    data_buffer[s * self._n_channels + c].decode("utf-8")
                    for c in range(self._n_channels)
                ]
                for s in range(int(n_samples))
            ]
            _free_char_p_array_memory(data_buffer)
        else:
            # this is 400-500x faster than the list approach
            samples = np.frombuffer(data_buffer, dtype=self._value_type)[
                :n_samples_data
            ].reshape(-1, self._n_channels)

        # %timeit [ts_buff[s] for s in range(int(num_samples))]
        # 68.8 µs ± 635 ns per loop
        # %timeit np.array(ts_buff)
        # 854 ns ± 2.99 ns per loop
        # %timeit np.frombuffer(ts_buff)
        # 192 ns ± 1.11 ns per loop
        # requires numpy ≥ 1.20
        timestamps = np.frombuffer(ts_buffer)[:n_samples]
        return samples, timestamps

    def flush(self) -> int:
        """Drop all queued and not-yet pulled samples.

        Returns
        -------
        n_dropped : int
            Number of dropped samples.
        """
        return lib.lsl_inlet_flush(self._obj)

    # -------------------------------------------------------------------------
    @copy_doc(_BaseStreamInfo.dtype)
    @property
    def dtype(self) -> str:
        return fmt2string[self._dtype]

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

    @property
    def samples_available(self) -> int:
        """Number of currently available samples on the Outlet.

        Returns
        -------
        n_samples : int
            Number of available samples.
        """
        # 354 ns ± 6.04 ns per loop
        return lib.lsl_samples_available(self._obj)

    @property
    def was_clock_reset(self) -> bool:
        """True if the clock was potentially reset since the last call."""
        return bool(lib.lsl_was_clock_reset(self._obj))

    # -------------------------------------------------------------------------
    def get_sinfo(self, timeout: Optional[float] = None) -> _BaseStreamInfo:
        """`~bsl.lsl.StreamInfo` corresponding to this Inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.

        Returns
        -------
        sinfo : StreamInfo
            Description of the stream connected to the inlet.
        """
        timeout = _check_timeout(timeout)
        errcode = c_int()
        result = lib.lsl_get_fullinfo(
            self._obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return _BaseStreamInfo(result)
