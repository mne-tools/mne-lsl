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

        self.obj = lib.lsl_create_inlet(
            sinfo.obj, max_buffered, chunk_size, recover
        )
        self.obj = c_void_p(self.obj)
        if not self.obj:
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
                lib.lsl_set_postprocessing(self.obj, processing_flags)
            )

        # properties from the StreamInfo
        self._channel_format = sinfo._channel_format
        self._name = sinfo.name
        self._n_channels = sinfo.n_channels
        self._sfreq = sinfo.sfreq
        self._stype = sinfo.stype

        # inlet properties
        self._do_pull_sample = fmt2pull_sample[self._channel_format]
        self._do_pull_chunk = fmt2pull_chunk[self._channel_format]
        self._value_type = fmt2type[self._channel_format]
        self._sample_type = self._value_type * self._n_channels
        self._sample = self._sample_type()
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

        Notes
        -----
        Opening a stream is a non-blocking operation. Thus, samples pushed on
        an outlet while the stream is not yet open will be missed.
        """
        timeout = _check_timeout(timeout)
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
        timeout = _check_timeout(timeout)
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
            Acquisition timestamp on the remote machine. To map the timestamp
            to the local clock of the client machine, add the estimated time
            correction return by `~bsl.lsl.StreamInlet.time_correction`.

        Notes
        -----
        If the outlet did not push any new sample (i.e. if the number of
        available samples is 0), this function is blocking.
        """
        timeout = _check_timeout(timeout)

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
                sample = np.frombuffer(self._sample, dtype=self._value_type)
        else:
            sample = None
            timestamp = None
        return sample, timestamp

    def pull_chunk(
        self,
        timeout: Optional[float] = None,
        n_samples: int = 1024,
    ):
        """Pull a chunk of samples from the inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout
            is disabled.
        n_samples : int
            Number of samples to return. The function is blocking until this
            number of samples is available.

        Returns
        -------
        samples : list of list | array of shape (n_channels, n_samples) | None
            If the channel format is ``'string'``, returns a list of list of
            values for each channel and sample. Each sublist represents an
            entire channel. Else, returns a numpy array of shape
            ``(n_channels, n_samples)``.
        timestamps : array of shape (n_samples,) | None
            Acquisition timestamps on the remote machine.
        """
        timeout = _check_timeout(timeout)
        if not isinstance(n_samples, int):
            n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(
                "The argument 'n_samples' must be a strictly positive "
                f"integer. {n_samples} is invalid."
            )

        # look up or create a pre-allocated buffer of appropriate length
        max_values = n_samples * self._n_channels
        if n_samples not in self._buffers:
            self._buffers[n_samples] = (
                (self._value_type * max_values)(),  # data
                (c_double * n_samples)(),  # timestamps
            )
        data_buffer = self._buffers[n_samples][0]
        ts_buffer = self._buffers[n_samples][1]

        # read data into it
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

        num_samples = int(num_elements / self._n_channels)
        if self._channel_format == cf_string:
            samples = [
                [
                    data_buffer[s + c * num_samples].decode("utf-8")
                    for s in range(num_samples)
                ]
                for c in range(self._n_channels)
            ]
            _free_char_p_array_memory(data_buffer, max_values)
        else:
            # this is 400-500x faster than the list approach
            samples = np.frombuffer(
                data_buffer, dtype=self._value_type
            ).reshape(self._n_channels, -1)

        # %timeit [ts_buff[s] for s in range(int(num_samples))]
        # 68.8 µs ± 635 ns per loop
        # %timeit np.array(ts_buff)
        # 854 ns ± 2.99 ns per loop
        # %timeit np.frombuffer(ts_buff)
        # 192 ns ± 1.11 ns per loop
        timestamps = np.frombuffer(ts_buffer)  # requires numpy ≥ 1.20
        return samples, timestamps

    def flush(self) -> int:
        """Drop all queued and not-yet pulled samples.

        Returns
        -------
        n_dropped : int
            Number of dropped samples.
        """
        return lib.lsl_inlet_flush(self.obj)

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

    @property
    def samples_available(self) -> int:
        """Number of currently available samples on the Outlet.

        Returns
        -------
        n_samples : int
            Number of available samples.
        """
        return lib.lsl_samples_available(self.obj)

    @property
    def was_clock_reset(self) -> bool:
        """True if the clock was potentially reset since the last call."""
        return bool(lib.lsl_was_clock_reset(self.obj))

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
            self.obj, c_double(timeout), byref(errcode)
        )
        handle_error(errcode)
        return _BaseStreamInfo(result)
