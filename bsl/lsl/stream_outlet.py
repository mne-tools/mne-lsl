from ctypes import c_char_p, c_double, c_int, c_long, c_void_p
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..utils._checks import _check_type
from ..utils._docs import copy_doc
from .constants import fmt2idx, fmt2push_chunk, fmt2push_sample, fmt2string
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo
from .utils import _check_timeout, handle_error


class StreamOutlet:
    """An outlet to share data and metadata on the network.

    Parameters
    ----------
    sinfo : StreamInfo
        The `~bsl.lsl.StreamInfo` object describing the stream. Stays constant
        over the lifetime of the outlet.
    chunk_size : int ``≥ 1``
        The desired chunk granularity in samples. By default, each push
        operation yields one chunk. An Inlet can override this setting.
    max_buffered : float ``≥ 0``
        The maximum amount of data to buffer in the Outlet.
        The number of samples buffered is ``max_buffered * 100`` if the
        sampling rate is irregular, else it's ``max_buffered`` seconds.
    """

    def __init__(
        self,
        sinfo: _BaseStreamInfo,
        chunk_size: int = 1,
        max_buffered: float = 360,
    ):
        _check_type(sinfo, (_BaseStreamInfo,), "sinfo")
        _check_type(chunk_size, ("int",), "chunk_size")
        if chunk_size < 1:
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
        self._obj = lib.lsl_create_outlet(sinfo._obj, chunk_size, max_buffered)
        self._obj = c_void_p(self._obj)
        if not self._obj:
            raise RuntimeError("The StreamOutlet could not be created.")

        # properties from the StreamInfo
        self._dtype = sinfo._dtype
        self._name = sinfo.name
        self._n_channels = sinfo.n_channels
        self._sfreq = sinfo.sfreq
        self._stype = sinfo.stype

        # outlet properties
        self._do_push_sample = fmt2push_sample[fmt2idx[self._dtype]]
        self._do_push_chunk = fmt2push_chunk[fmt2idx[self._dtype]]
        self._buffer_sample = self._dtype * self._n_channels

    def __del__(self):
        """Destroy a `~bsl.lsl.StreamOutlet`.

        The outlet will no longer be discoverable after destruction and all
        connected inlets will stop delivering data.
        """
        try:
            lib.lsl_destroy_outlet(self._obj)
        except Exception:
            pass

    def push_sample(
        self,
        x: Union[List[Union[float, str]], NDArray[float]],
        timestamp: float = 0.0,
        pushThrough: bool = True,
    ) -> None:
        """Push a sample into the `~bsl.lsl.StreamOutlet`.

        Parameters
        ----------
        x : list | array of shape (n_channels,)
            Sample to push, with one element for each channel.
        timestamp : float optional
            The acquisition timestamp of the sample, in agreement with
            `~bsl.lsl.local_clock`. The default, `0`, uses the current time.
        pushThrough : bool, optional
            If True, push the sample through to the receivers instead of
            buffering it with subsequent samples. Note that the ``chunk_size``
            defined when creating a `~bsl.lsl.StreamOutlet` takes precedence
            over the ``pushThrough`` flag.
        """
        assert isinstance(x, (list, np.ndarray)), "'x' must be a list or array"
        if isinstance(x, np.ndarray) and x.ndim != 1:
            raise ValueError(
                "The sample to push 'x' must contain one element per channel. "
                f"Thus, the shape should be (n_channels,), {x.shape} is "
                "invalid."
            )
        elif len(x) != self._n_channels:
            raise ValueError(
                "The sample to push 'x' must contain one element per channel. "
                f"Thus, {self._n_channels} elements are expected. {len(x)} "
                "is invalid."
            )

        if self._dtype == c_char_p:
            x = [v.encode("utf-8") for v in x]
        handle_error(
            self._do_push_sample(
                self._obj,
                self._buffer_sample(*x),
                c_double(timestamp),
                c_int(pushThrough),
            )
        )

    def push_chunk(
        self,
        x: Union[List[List[Union[float, str]]], NDArray[float]],
        timestamp: float = 0.0,
        pushthrough: bool = True,
    ) -> None:
        """Push a chunk of samples into the `~bsl.lsl.StreamOutlet`.

        Parameters
        ----------
        x : list of list | array of shape (n_samples, n_channels)
            Samples to push, with one element for each channel at every time
            point. If a list of list, each sublist has ``(n_channels,)``
            elements and contain an entire sample. Numpy arrays are
            recommended for numerical formats.
        timestamp : float
            The acquisition timestamp of the sample, in agreement with
            `~bsl.lsl.local_clock`. The default, ``0``, uses the current time.
        pushThrough : bool
            If True, push the sample through to the receivers instead of
            buffering it with subsequent samples. Note that the ``chunk_size``
            defined when creating a `~bsl.lsl.StreamOutlet` takes precedence
            over the ``pushThrough`` flag.
        """
        assert isinstance(
            x, (list, np.ndarray)
        ), "'x' must be a list of list or an array"
        if isinstance(x, np.ndarray):
            if x.ndim != 2 or x.shape[1] != self._n_channels:
                raise ValueError(
                    "The samples to push 'x' must contain one element per "
                    "channel at each time-point. Thus, the shape should be "
                    f"(n_samples, n_channels), {x.shape} is invalid."
                )
            x = x if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x)
            n_samples = x.size
            data_buffer = (self._dtype * n_samples).from_buffer(x)
        else:
            # we do not check the input, specifically, that all elements in the
            # list are list and that all list have the correct number of
            # element to avoid slowing down the execution.
            x = [v for sample in x for v in sample]  # flatten
            n_samples = len(x)
            if n_samples % self._n_channels != 0:  # quick incomplete test
                raise ValueError(
                    "The samples to push 'x' must contain one element per "
                    "channel at each time-point. Thus, the shape should be "
                    "(n_samples, n_channels)."
                )
            if self._dtype == c_char_p:
                x = [v.encode("utf-8") for v in x]
            data_buffer = (self._dtype * n_samples)(*x)

        handle_error(
            self._do_push_chunk(
                self._obj,
                data_buffer,
                c_long(n_samples),
                c_double(timestamp),
                c_int(pushthrough),
            )
        )

    def wait_for_consumers(self, timeout: Optional[float]) -> bool:
        """Wait (block) until at least one `~bsl.lsl.StreamInlet` connects.

        Parameters
        ----------
        timeout : float
            Timeout duration in seconds.

        Returns
        -------
        success : bool
            True if the wait was successful, False if the ``timeout`` expired.

        Notes
        -----
        This function does not filter the search for `bsl.lsl.StreamInlet`.
        Any application inlet will be recognized.
        """
        timeout = _check_timeout(timeout)
        return bool(lib.lsl_wait_for_consumers(self._obj, c_double(timeout)))

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
    def has_consumers(self) -> bool:
        """Check whether `~bsl.lsl.StreamInlet` are currently connected.

        While it does not hurt, there is technically no reason to push samples
        if there is no one connected.

        Returns
        -------
        consumers : bool
            True if at least one consumer is connected.

        Notes
        -----
        This function does not filter the search for `bsl.lsl.StreamInlet`. Any
        application inlet will be recognized.
        """
        return bool(lib.lsl_have_consumers(self._obj))

    # -------------------------------------------------------------------------
    def get_sinfo(self) -> _BaseStreamInfo:
        """`~bsl.lsl.StreamInfo` corresponding to this Outlet.

        Returns
        -------
        sinfo : StreamInfo
            Description of the stream connected to the outlet.
        """
        return _BaseStreamInfo(lib.lsl_get_info(self._obj))
