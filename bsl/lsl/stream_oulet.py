from ctypes import c_double, c_int, c_long, c_void_p
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from ..utils._docs import copy_doc
from .constants import cf_string, fmt2push_chunk, fmt2push_sample, fmt2type
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo
from .utils import handle_error


class StreamOutlet:
    def __init__(self, sinfo, chunk_size=0, max_buffered=360):
        self.obj = lib.lsl_create_outlet(sinfo.obj, chunk_size, max_buffered)
        self.obj = c_void_p(self.obj)
        if not self.obj:
            raise RuntimeError("The StreamOutlet could not be created.")

        # properties from the StreamInfo
        self._channel_format = sinfo.channel_format
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
        self, x: Union[List, NDArray], timestamp=0.0, pushthrough=True
    ):
        if isinstance(x, np.ndarray) and len(x.shape) != 1:
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
                c_int(pushthrough),
            )
        )

    def push_chunk(self, x, timestamp=0.0, pushthrough=True):
        try:
            n_values = self._n_channels * len(x)
            data_buff = (self._value_type * n_values).from_buffer(x)
            handle_error(
                self._do_push_chunk(
                    self.obj,
                    data_buff,
                    c_long(n_values),
                    c_double(timestamp),
                    c_int(pushthrough),
                )
            )
        except TypeError:
            if len(x):
                if type(x[0]) is list:
                    x = [v for sample in x for v in sample]
                if self._channel_format == cf_string:
                    x = [v.encode("utf-8") for v in x]
                if len(x) % self._n_channels == 0:
                    constructor = self.value_type * len(x)
                    handle_error(
                        self.do_push_chunk(
                            self.obj,
                            constructor(*x),
                            c_long(len(x)),
                            c_double(timestamp),
                            c_int(pushthrough),
                        )
                    )
                else:
                    raise ValueError(
                        "Each sample must have the same number of channels ("
                        + str(self._n_channels)
                        + ")."
                    )

    def have_consumers(self) -> bool:
        """Check whether `~bsl.lsl.StreamInlet` are currently connected.

        While it does not hurt, there is technically no reason to push samples
        if there is no one connected.
        """
        return bool(lib.lsl_have_consumers(self.obj))

    def wait_for_consumers(self, timeout: float) -> bool:
        """Wait until at least one `~bsl.lsl.StreamInlet` connects.

        Returns True if the wait was successful, False if the timeout expired.
        """
        return bool(lib.lsl_wait_for_consumers(self.obj, c_double(timeout)))

    # -------------------------------------------------------------------------
    @copy_doc(_BaseStreamInfo.channel_format)
    @property
    def channel_format(self) -> str:
        return self._channel_format

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

    @property
    def sinfo(self) -> _BaseStreamInfo:
        return _BaseStreamInfo(lib.lsl_get_info(self.obj))
