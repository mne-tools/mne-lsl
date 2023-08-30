from __future__ import annotations  # c.f. PEP 563, PEP 649

import time
from ctypes import byref, c_char_p, c_double, c_int, c_size_t, c_void_p
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from ..utils._checks import check_type, check_value, ensure_int
from ..utils._docs import copy_doc
from .constants import fmt2numpy, fmt2pull_chunk, fmt2pull_sample, post_processing_flags
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo
from .utils import _check_timeout, _free_char_p_array_memory, handle_error

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union

    from numpy.typing import DTypeLike, NDArray


class StreamInlet:
    """An inlet to retrieve data and metadata on the network.

    Parameters
    ----------
    sinfo : StreamInfo
        Description of the stream to connect to.
    chunk_size : int ``≥ 1`` | ``0``
        The desired chunk granularity in samples. By default, the ``chunk_size`` defined
        by the sender (outlet) is used.
    max_buffered : float ``≥ 0``
        The maximum amount of data to buffer in the Outlet. The number of samples
        buffered is ``max_buffered * 100`` if the sampling rate is irregular, else it's
        ``max_buffered`` seconds.
    recover : bool
        Attempt to silently recover lost streams that are recoverable (requires a
        ``source_id`` to be specified in the `~bsl.lsl.StreamInfo`).
    processing_flags : sequence of str | ``'all'`` | None
        Set the post-processing options. By default, post-processing is disabled. Any
        combination of the processing flags is valid. The available flags are:

        * ``'clocksync'``: Automatic clock synchronization, equivalent to
          manually adding the estimated `~bsl.lsl.StreamInlet.time_correction`.
        * ``'dejitter'``: Remove jitter on the received timestamps with a
          smoothing algorithm.
        * ``'monotize'``: Force the timestamps to be monotically ascending.
          This option should not be enable if ``'dejitter'`` is not enabled.
        * ``'threadsafe'``: Post-processing is thread-safe, thus the same
          inlet can be read from multiple threads.
    """

    def __init__(
        self,
        sinfo: _BaseStreamInfo,
        chunk_size: int = 0,
        max_buffered: float = 360,
        recover: bool = True,
        processing_flags: Optional[Union[str, Sequence[str]]] = None,
    ):
        check_type(sinfo, (_BaseStreamInfo,), "sinfo")
        chunk_size = ensure_int(chunk_size, "chunk_size")
        if chunk_size < 0:
            raise ValueError(
                "The argument 'chunk_size' must contain a positive integer. "
                f"{chunk_size} is invalid."
            )
        check_type(max_buffered, ("numeric",), "max_buffered")
        if max_buffered < 0:
            raise ValueError(
                "The argument 'max_buffered' must contain a positive number. "
                f"{max_buffered} is invalid."
            )
        check_type(recover, (bool,), "recover")

        self._obj = lib.lsl_create_inlet(sinfo._obj, max_buffered, chunk_size, recover)
        self._obj = c_void_p(self._obj)
        if not self._obj:
            raise RuntimeError("The StreamInlet could not be created.")

        # set preprocessing of the inlet
        if processing_flags is not None:
            check_type(processing_flags, (list, tuple, str), "processing_flags")
            if isinstance(processing_flags, str):
                check_value(processing_flags, ("all",), "processing_flags")
                processing_flags = reduce(
                    lambda x, y: x | y, post_processing_flags.values()
                )
            else:
                for flag in processing_flags:
                    check_type(flag, (str,), "processing_flag")
                    check_value(flag, post_processing_flags, flag)
                if (
                    "monotize" in processing_flags
                    and "dejitter" not in processing_flags
                ):
                    raise ValueError(
                        "The processing flag 'monotize' should not be used without the "
                        "processing flag 'dejitter'."
                    )
                # bitwise OR between the flags
                processing_flags = [
                    post_processing_flags[key] for key in processing_flags
                ]
                processing_flags = reduce(lambda x, y: x | y, processing_flags)
            assert processing_flags > 0  # sanity-check
            handle_error(lib.lsl_set_postprocessing(self._obj, processing_flags))

        # properties from the StreamInfo
        self._dtype = sinfo._dtype
        self._name = sinfo.name
        self._n_channels = sinfo.n_channels
        self._sfreq = sinfo.sfreq
        self._stype = sinfo.stype

        # inlet properties
        self._do_pull_sample = fmt2pull_sample[self._dtype]
        self._do_pull_chunk = fmt2pull_chunk[self._dtype]
        self._buffer_data = {1: (self._dtype * self._n_channels)()}
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

        All samples pushed in at the other end from this moment onwards will be queued
        and eventually be delivered in response to `~bsl.lsl.StreamInlet.pull_sample` or
        `~bsl.lsl.StreamInlet.pull_chunk` calls. Pulling a sample without subscribing to
        the stream with this method is permitted (the stream will be opened implicitly).

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout is
            disabled.

        Notes
        -----
        Opening a stream is a non-blocking operation. Thus, samples pushed on an outlet
        while the stream is not yet open will be missed.
        """
        timeout = _check_timeout(timeout)
        errcode = c_int()
        lib.lsl_open_stream(self._obj, c_double(timeout), byref(errcode))
        handle_error(errcode)
        # block a bit longer because of a bug in liblsl
        # this sleep can be removed once the minimum version supported includes
        # a fix for https://github.com/sccn/liblsl/issues/176
        time.sleep(0.5)

    def close_stream(self) -> None:
        """Drop the current data stream.

        All samples that are still buffered or in flight will be dropped and
        transmission and buffering of data for this inlet will be stopped. This method
        is used if an application stops being interested in data from a source
        (temporarily or not) but keeps the outlet alive, to not waste unnecessary system
        and network resources.

        .. warning::

            At the moment, ``liblsl`` is released in version 1.16. Closing and
            re-opening a stream does not work and new samples pushed to the outlet do
            not arrive at the inlet. c.f. this
            `github issue <https://github.com/sccn/liblsl/issues/180>`_.
        """
        lib.lsl_close_stream(self._obj)

    def time_correction(self, timeout: Optional[float] = None) -> float:
        """Retrieve an estimated time correction offset for the given stream.

        The first call to this function takes several milliseconds until a reliable
        first estimate is obtained. Subsequent calls are instantaneous (and rely on
        periodic background updates). The precision of these estimates should be below
        1 ms (empirically within +/-0.2 ms).

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. By default, timeout is
            disabled.

        Returns
        -------
        time_correction : float
            Current estimate of the time correction. This number needs to be added to a
            timestamp that was remotely generated via ``local_clock()`` to map it into
            the local clock domain of the client machine.
        """
        timeout = _check_timeout(timeout)
        errcode = c_int()
        result = lib.lsl_time_correction(self._obj, c_double(timeout), byref(errcode))
        handle_error(errcode)
        return result

    def pull_sample(
        self, timeout: Optional[float] = 0.0
    ) -> Tuple[Union[List[str], NDArray[float]], Optional[float]]:
        """Pull a single sample from the inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. None correspond to a very
            large value, effectively disabling the timeout. ``0.`` makes this function
            non-blocking even if no sample is available. See notes for additional
            details.

        Returns
        -------
        sample : list of str | array of shape (n_channels,)
            If the channel format is ``'string``, returns a list of values for each
            channel. Else, returns a numpy array of shape ``(n_channels,)``.
        timestamp : float | None
            Acquisition timestamp on the remote machine. To map the timestamp to the
            local clock of the client machine, add the estimated time correction return
            by `~bsl.lsl.StreamInlet.time_correction`. None if no sample was retrieved.

        Notes
        -----
        Note that if ``timeout`` is reached and no sample is available, an empty
        ``sample`` is returned and ``timestamp`` is set to None.
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
            if self._dtype == c_char_p:
                sample = [v.decode("utf-8") for v in self._buffer_data[1]]
                _free_char_p_array_memory(self._buffer_data[1])
            else:
                sample = np.frombuffer(self._buffer_data[1], dtype=self._dtype)
        else:
            sample = [] if self._dtype == c_char_p else np.empty(0, dtype=self._dtype)
            timestamp = None
        return sample, timestamp

    def pull_chunk(
        self,
        timeout: Optional[float] = 0.0,
        max_samples: int = 1024,
    ) -> Tuple[Union[List[List[str]], NDArray[float]], NDArray[float]]:
        """Pull a chunk of samples from the inlet.

        Parameters
        ----------
        timeout : float | None
            Optional timeout (in seconds) of the operation. None correspond to a very
            large value, effectively disabling the timeout. ``0.`` makes this function
            non-blocking even if no sample is available. See notes for additional
            details.
        max_samples : int
            Maximum number of samples to return. The function is blocking until this
            number of samples is available or until ``timeout`` is reached. See notes
            for additional details.

        Returns
        -------
        samples : list of list of str | array of shape (n_samples, n_channels)
            If the channel format is ``'string'``, returns a list of list of values for
            each channel and sample. Each sublist represents an entire channel. Else,
            returns a numpy array of shape ``(n_samples, n_channels)``.
        timestamps : array of shape (n_samples,)
            Acquisition timestamps on the remote machine.

        Notes
        -----
        The argument ``timeout`` and ``max_samples`` control the blocking behavior of
        the pull operation. If the number of available sample is inferior to
        ``n_samples``, the pull operation is blocking until ``timeout`` is reached.
        Thus, to return all the available samples at a given time, regardless of the
        number of samples requested, ``timeout`` must be set to ``0``.

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
            self._buffer_data[max_samples_data] = (self._dtype * max_samples_data)()
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
            c_size_t(max_samples_data),
            c_size_t(max_samples),
            c_double(timeout),
            byref(errcode),
        )
        handle_error(errcode)

        n_samples = int(n_samples_data / self._n_channels)
        if self._dtype == c_char_p:
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
            samples = np.frombuffer(data_buffer, dtype=self._dtype)[
                :n_samples_data
            ].reshape(-1, self._n_channels)

        # %timeit [ts_buff[s] for s in range(int(num_samples))]
        # 68.8 µs ± 635 ns per loop
        # %timeit np.array(ts_buff)
        # 854 ns ± 2.99 ns per loop
        # %timeit np.frombuffer(ts_buff)
        # 192 ns ± 1.11 ns per loop
        # requires numpy ≥ 1.20
        timestamps = np.frombuffer(ts_buffer, dtype=np.float64)[:n_samples]
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
    def dtype(self) -> Union[str, DTypeLike]:
        return fmt2numpy.get(self._dtype, "string")

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
            Optional timeout (in seconds) of the operation. By default, timeout is
            disabled.

        Returns
        -------
        sinfo : StreamInfo
            Description of the stream connected to the inlet.
        """
        timeout = _check_timeout(timeout)
        errcode = c_int()
        result = lib.lsl_get_fullinfo(self._obj, c_double(timeout), byref(errcode))
        handle_error(errcode)
        return _BaseStreamInfo(result)
