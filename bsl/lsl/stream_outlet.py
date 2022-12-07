from ctypes import c_double, c_int, c_long, c_void_p
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from ..utils._checks import _check_type
from ..utils._docs import copy_doc
from .constants import (
    cf_string,
    fmt2push_chunk,
    fmt2push_sample,
    fmt2string,
    fmt2type,
)
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo
from .utils import handle_error


class StreamOutlet:

    def __init__(self, sinfo, chunk_size=0, max_buffered=360):
        _check_type(sinfo, (_BaseStreamInfo,), "sinfo")
        _check_type(chunk_size, ("numeric",), "chunk_size")
        if chunk_size < 1:
            raise ValueError(
                "The argument 'chunk_size' must contain a positive number. "
                f"{chunk_size} is invalid."
            )
        _check_type(max_buffered, ("numeric",), "max_buffered")
        if max_buffered < 0:
            raise ValueError(
                "The argument 'max_buffered' must contain a positive number. "
                f"{max_buffered} is invalid."
            )
        self.obj = lib.lsl_create_outlet(sinfo.obj, chunk_size, max_buffered)
        self.obj = c_void_p(self.obj)
        if not self.obj:
            raise RuntimeError("The StreamOutlet could not be created.")

        # properties from the StreamInfo
        self._channel_format = sinfo._channel_format
        self._name = sinfo.name
        self._n_channels = sinfo.n_channels
        self._sfreq = sinfo.sfreq
        self._stype = sinfo.stype

        # outlet properties
        self.do_push_sample = fmt2push_sample[self._channel_format]
        self.do_push_chunk = fmt2push_chunk[self._channel_format]
        self.value_type = fmt2type[self._channel_format]
        self.sample_type = self.value_type * self._n_channels

    def __del__(self):
        # noinspection PyBroadException
        try:
            lib.lsl_destroy_outlet(self.obj)
        except:
            pass

    def push_sample(self, x, timestamp=0.0, pushthrough=True):
        if len(x) == self._n_channels:
            if self._channel_format == cf_string:
                x = [v.encode('utf-8') for v in x]
            handle_error(self.do_push_sample(self.obj, self.sample_type(*x),
                                             c_double(timestamp),
                                             c_int(pushthrough)))
        else:
            raise ValueError("length of the sample (" + str(len(x)) + ") must "
                             "correspond to the stream's channel count ("
                             + str(self._n_channels) + ").")

    def push_chunk(self, x, timestamp=0.0, pushthrough=True):
        try:
            n_values = self._n_channels * len(x)
            data_buff = (self.value_type * n_values).from_buffer(x)
            handle_error(self.do_push_chunk(self.obj, data_buff,
                                            c_long(n_values),
                                            c_double(timestamp),
                                            c_int(pushthrough)))
        except TypeError:
            if len(x):
                if type(x[0]) is list:
                    x = [v for sample in x for v in sample]
                if self._channel_format == cf_string:
                    x = [v.encode('utf-8') for v in x]
                if len(x) % self._n_channels == 0:
                    constructor = self.value_type * len(x)
                    # noinspection PyCallingNonCallable
                    handle_error(self.do_push_chunk(self.obj, constructor(*x),
                                                    c_long(len(x)),
                                                    c_double(timestamp),
                                                    c_int(pushthrough)))
                else:
                    raise ValueError("Each sample must have the same number of channels ("
                                     + str(self._n_channels) + ").")

    def have_consumers(self):
        return bool(lib.lsl_have_consumers(self.obj))

    def wait_for_consumers(self, timeout):
        return bool(lib.lsl_wait_for_consumers(self.obj, c_double(timeout)))

    def get_info(self):
        outlet_info = lib.lsl_get_info(self.obj)
        return _BaseStreamInfo(outlet_info)

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
    def get_sinfo(self) -> _BaseStreamInfo:
        """`~bsl.lsl.StreamInfo` corresponding to this Outlet."""
        return _BaseStreamInfo(lib.lsl_get_info(self.obj))


class NewStreamOutlet:
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

    def __init__(self, sinfo, chunk_size=1, max_buffered=360):
        _check_type(sinfo, (_BaseStreamInfo,), "sinfo")
        _check_type(chunk_size, ("numeric",), "chunk_size")
        if chunk_size < 1:
            raise ValueError(
                "The argument 'chunk_size' must contain a positive number. "
                f"{chunk_size} is invalid."
            )
        _check_type(max_buffered, ("numeric",), "max_buffered")
        if max_buffered < 0:
            raise ValueError(
                "The argument 'max_buffered' must contain a positive number. "
                f"{max_buffered} is invalid."
            )

        self.obj = lib.lsl_create_outlet(sinfo.obj, chunk_size, max_buffered)
        self.obj = c_void_p(self.obj)
        if not self.obj:
            raise RuntimeError("The StreamOutlet could not be created.")

        # properties from the StreamInfo
        self._channel_format = sinfo._channel_format
        self._name = sinfo.name
        self._n_channels = sinfo.n_channels
        self._sfreq = sinfo.sfreq
        self._stype = sinfo.stype

        # outlet properties
        self._do_push_sample = fmt2push_sample[self._channel_format]
        self._do_push_chunk = fmt2push_chunk[self._channel_format]
        self._value_type = fmt2type[self._channel_format]
        self._sample_type = self._value_type * self._n_channels

    def __del__(self):
        """Destroy a `~bsl.lsl.StreamOutlet`.

        The outlet will no longer be discoverable after destruction and all
        connected inlets will stop delivering data.
        """
        try:
            lib.lsl_destroy_outlet(self.obj)
        except Exception:
            pass

    def push_sample(
        self,
        x: Union[List, NDArray],
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

        if self._channel_format == cf_string:
            x = [v.encode("utf-8") for v in x]
        handle_error(
            self._do_push_sample(
                self.obj,
                self._sample_type(*x),
                c_double(timestamp),
                c_int(pushThrough),
            )
        )

    def push_chunk(self, x, timestamp=0.0, pushthrough=True) -> None:
        """Push a chunk of samples into the `~bsl.lsl.StreamOutlet`.

        Parameters
        ----------
        x : list of list | array of shape (n_channels, n_samples)
            Samples to push, with one element for each channel at every time
            point.
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
        if isinstance(x, np.ndarray):
            if x.ndim != 2 or x.shape[0] != self._n_channels:
                raise ValueError(
                    "The samples to push 'x' must contain one element per "
                    "channel at each time-point. Thus, the shape should be "
                    f"(n_channels, n_samples), {x.shape} is invalid."
                )
            n_values = x.size
            data_buffer = (self._value_type * n_values).from_buffer(x)
        else:
            # we do not check the input, specifically, that all elements in the
            # list are list and that all list have the correct number of
            # element to avoid slowing down the execution.
            x = [v for sample in x for v in sample]  # flatten
            n_values = len(x)
            if n_values % self._n_channels != 0:  # quick incomplete test
                raise ValueError(
                    "The samples to push 'x' must contain one element per "
                    "channel at each time-point. Thus, the shape should be "
                    "(n_channels, n_samples)."
                )
            if self._channel_format == cf_string:
                x = [v.encode("utf-8") for v in x]
            constructor = self._value_type * n_values
            data_buffer = constructor(*x)

        handle_error(
            self._do_push_chunk(
                self.obj,
                data_buffer,
                c_long(n_values),
                c_double(timestamp),
                c_int(pushthrough),
            )
        )

    def have_consumers(self) -> bool:
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
        return bool(lib.lsl_have_consumers(self.obj))

    def wait_for_consumers(self, timeout: float) -> bool:
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
        return bool(lib.lsl_wait_for_consumers(self.obj, c_double(timeout)))

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
    def get_sinfo(self) -> _BaseStreamInfo:
        """`~bsl.lsl.StreamInfo` corresponding to this Outlet."""
        return _BaseStreamInfo(lib.lsl_get_info(self.obj))
