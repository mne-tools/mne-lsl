from collections.abc import Sequence as Sequence

import numpy as np
from _typeshed import Incomplete
from numpy.typing import DTypeLike as DTypeLike
from numpy.typing import NDArray as NDArray

from .._typing import ScalarArray as ScalarArray
from ..utils._checks import check_type as check_type
from ..utils._checks import check_value as check_value
from ..utils._checks import ensure_int as ensure_int
from ..utils._docs import copy_doc as copy_doc
from ..utils.logs import warn as warn
from ._utils import check_timeout as check_timeout
from ._utils import free_char_p_array_memory as free_char_p_array_memory
from ._utils import handle_error as handle_error
from .constants import fmt2numpy as fmt2numpy
from .constants import fmt2pull_chunk as fmt2pull_chunk
from .constants import fmt2pull_sample as fmt2pull_sample
from .constants import post_processing_flags as post_processing_flags
from .load_liblsl import lib as lib
from .stream_info import _BaseStreamInfo as _BaseStreamInfo

class StreamInlet:
    """An inlet to retrieve data and metadata on the network.

    Parameters
    ----------
    sinfo : StreamInfo
        Description of the stream to connect to.
    chunk_size : int ``≥ 1`` | ``0``
        The desired chunk granularity in samples. By default, the ``chunk_size`` defined
        by the sender (outlet) is used.
    max_buffered : int ``≥ 0``
        The maximum amount of data to buffer in the Outlet. The number of samples
        buffered is ``max_buffered * 100`` if the sampling rate is irregular, else it's
        ``max_buffered`` seconds.
    recover : bool
        Attempt to silently recover lost streams that are recoverable (requires a
        ``source_id`` to be specified in the :class:`~mne_lsl.lsl.StreamInfo`).
    processing_flags : sequence of str | ``'all'`` | None
        Set the post-processing options. By default, post-processing is disabled. Any
        combination of the processing flags is valid. The available flags are:

        * ``'clocksync'``: Automatic clock synchronization, equivalent to
          manually adding the estimated
          :meth:`~mne_lsl.lsl.StreamInlet.time_correction`.
        * ``'dejitter'``: Remove jitter on the received timestamps with a
          smoothing algorithm.
        * ``'monotize'``: Force the timestamps to be monotically ascending.
          This option should not be enable if ``'dejitter'`` is not enabled.
        * ``'threadsafe'``: Post-processing is thread-safe, thus the same
          inlet can be read from multiple threads.
    """

    _lock: Incomplete
    _dtype: Incomplete
    _name: Incomplete
    _n_channels: Incomplete
    _sfreq: Incomplete
    _stype: Incomplete
    _do_pull_sample: Incomplete
    _do_pull_chunk: Incomplete
    _buffer_data: Incomplete
    _buffer_ts: Incomplete
    _stream_is_open: bool

    def __init__(
        self,
        sinfo: _BaseStreamInfo,
        chunk_size: int = 0,
        max_buffered: float = 360,
        recover: bool = True,
        processing_flags: str | Sequence[str] | None = None,
    ) -> None: ...
    @property
    def _obj(self): ...
    __obj: Incomplete

    @_obj.setter
    def _obj(self, obj) -> None: ...
    def __del__(self) -> None:
        """Destroy a :class:`~mne_lsl.lsl.StreamInlet`.

        The inlet will automatically disconnect.
        """

    def open_stream(self, timeout: float | None = None) -> None:
        """Subscribe to a data stream.

        All samples pushed in at the other end from this moment onwards will be queued
        and eventually be delivered in response to
        :meth:`~mne_lsl.lsl.StreamInlet.pull_sample` or
        :meth:`~mne_lsl.lsl.StreamInlet.pull_chunk` calls. Pulling a sample without
        subscribing to the stream with this method is permitted (the stream will be
        opened implicitly).

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

    def time_correction(self, timeout: float | None = None) -> float:
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
            the :func:`~mne_lsl.lsl.local_clock` domain of the client machine.
        """

    def pull_sample(
        self, timeout: float | None = 0.0
    ) -> tuple[list[str] | ScalarArray, float | None]:
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
            by :meth:`~mne_lsl.lsl.StreamInlet.time_correction`. None if no sample was
            retrieved.

        Notes
        -----
        Note that if ``timeout`` is reached and no sample is available, an empty
        ``sample`` is returned and ``timestamp`` is set to None.
        """

    def pull_chunk(
        self, timeout: float | None = 0.0, max_samples: int = 1024
    ) -> tuple[list[list[str]] | ScalarArray, NDArray[np.float64]]:
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
            Acquisition timestamp on the remote machine. To map the timestamp to the
            local clock of the client machine, add the estimated time correction return
            by :meth:`~mne_lsl.lsl.StreamInlet.time_correction`.

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

    def flush(self) -> int:
        """Drop all queued and not-yet pulled samples.

        Returns
        -------
        n_dropped : int
            Number of dropped samples.
        """

    @property
    def dtype(self) -> str | DTypeLike:
        """Channel format of a stream.

        All channels in a stream have the same format.

        :type: :class:`~numpy.dtype` | ``"string"``
        """

    @property
    def n_channels(self) -> int:
        """Number of channels.

        A stream must have at least one channel. The number of channels remains constant
        for all samples.

        :type: :class:`int`
        """

    @property
    def name(self) -> str:
        """Name of the stream.

        The name of the stream is defined by the application creating the LSL outlet.
        Streams with identical names can coexist, at the cost of ambiguity for the
        recording application and/or the experimenter.

        :type: :class:`str`
        """

    @property
    def sfreq(self) -> float:
        """Sampling rate of the stream, according to the source (in Hz).

        If a stream is irregularly sampled, the sampling rate is set to ``0``.

        :type: :class:`float`
        """

    @property
    def stype(self) -> str:
        """Type of the stream.

        The content type is a short string, such as ``"EEG"``, ``"Gaze"``, ... which
        describes the content carried by the channel. If a stream contains mixed
        content, this value should be an empty string and the type should be stored in
        the description of individual channels.

        :type: :class:`str`
        """

    @property
    def samples_available(self) -> int:
        """Number of available samples on the :class:`~mne_lsl.lsl.StreamOutlet`.

        :type: :class:`int`
        """

    @property
    def was_clock_reset(self) -> bool:
        """True if the clock was potentially reset since the last call.

        :type: :class:`bool`
        """

    def get_sinfo(self, timeout: float | None = None) -> _BaseStreamInfo:
        """:class:`~mne_lsl.lsl.StreamInfo` corresponding to this Inlet.

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
