from _typeshed import Incomplete
from numpy.typing import DTypeLike as DTypeLike

from .._typing import ScalarArray as ScalarArray
from .._typing import ScalarFloatArray as ScalarFloatArray
from ..utils._checks import check_type as check_type
from ..utils._checks import ensure_int as ensure_int
from ..utils._docs import copy_doc as copy_doc
from ..utils.logs import logger as logger
from ._utils import check_timeout as check_timeout
from ._utils import handle_error as handle_error
from .constants import fmt2numpy as fmt2numpy
from .constants import fmt2push_chunk as fmt2push_chunk
from .constants import fmt2push_chunk_n as fmt2push_chunk_n
from .constants import fmt2push_sample as fmt2push_sample
from .load_liblsl import lib as lib
from .stream_info import _BaseStreamInfo as _BaseStreamInfo

class StreamOutlet:
    """An outlet to share data and metadata on the network.

    Parameters
    ----------
    sinfo : StreamInfo
        The :class:`~mne_lsl.lsl.StreamInfo` object describing the stream. Stays
        constant over the lifetime of the outlet.
    chunk_size : int ``≥ 1``
        The desired chunk granularity in samples. By default, each push operation yields
        one chunk. A :class:`~mne_lsl.lsl.StreamInlet` can override this setting.
    max_buffered : float ``≥ 0``
        The maximum amount of data to buffer in the Outlet. The number of samples
        buffered is ``max_buffered * 100`` if the sampling rate is irregular, else it's
        ``max_buffered`` seconds.
    """

    _lock: Incomplete
    _dtype: Incomplete
    _name: Incomplete
    _n_channels: Incomplete
    _sfreq: Incomplete
    _stype: Incomplete
    _do_push_sample: Incomplete
    _do_push_chunk: Incomplete
    _do_push_chunk_n: Incomplete
    _buffer_sample: Incomplete

    def __init__(
        self, sinfo: _BaseStreamInfo, chunk_size: int = 1, max_buffered: float = 360
    ) -> None: ...
    @property
    def _obj(self): ...
    __obj: Incomplete

    @_obj.setter
    def _obj(self, obj) -> None: ...
    def __del__(self) -> None:
        """Destroy a :class:`~mne_lsl.lsl.StreamOutlet`.

        The outlet will no longer be discoverable after destruction and all connected
        inlets will stop delivering data.
        """

    def push_sample(
        self,
        x: list[str] | ScalarArray,
        timestamp: float = 0.0,
        pushThrough: bool = True,
    ) -> None:
        """Push a sample into the :class:`~mne_lsl.lsl.StreamOutlet`.

        Parameters
        ----------
        x : list | array of shape (n_channels,)
            Sample to push, with one element for each channel. If strings are
            transmitted, a list is required. If numericals are transmitted, a numpy
            array is required.
        timestamp : float
            The acquisition timestamp of the sample, in agreement with
            :func:`mne_lsl.lsl.local_clock`. The default, ``0``, uses the current time.
        pushThrough : bool
            If True, push the sample through to the receivers instead of buffering it
            with subsequent samples. Note that the ``chunk_size`` defined when creating
            a :class:`~mne_lsl.lsl.StreamOutlet` takes precedence over the
            ``pushThrough`` flag.
        """

    def push_chunk(
        self,
        x: list[list[str]] | ScalarArray,
        timestamp: float | ScalarFloatArray | None = None,
        pushThrough: bool = True,
    ) -> None:
        """Push a chunk of samples into the :class:`~mne_lsl.lsl.StreamOutlet`.

        Parameters
        ----------
        x : list of list | array of shape (n_samples, n_channels)
            Samples to push, with one element for each channel at every time point. If
            strings are transmitted, a list of sublist containing ``(n_channels,)`` is
            required. If numericals are transmitted, a numpy array of shape
            ``(n_samples, n_channels)`` is required.
        timestamp : float | array of shape (n_samples,) | None
            Acquisition timestamp in agreement with :func:`mne_lsl.lsl.local_clock`.
            If a float, the acquisition timestamp of the last sample. ``None`` (default)
            uses the current time. If an array, the acquisition timestamp of each
            sample.
        pushThrough : bool
            If True, push the sample through to the receivers instead of buffering it
            with subsequent samples. Note that the ``chunk_size`` defined when creating
            a :class:`~mne_lsl.lsl.StreamOutlet` takes precedence over the
            ``pushThrough`` flag.
        """

    def wait_for_consumers(self, timeout: float | None) -> bool:
        """Wait (block) until at least one :class:`~mne_lsl.lsl.StreamInlet` connects.

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
        This function does not filter the search for :class:`mne_lsl.lsl.StreamInlet`.
        Any application inlet will be recognized.
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
    def has_consumers(self) -> bool:
        """True if at least one :class:`~mne_lsl.lsl.StreamInlet` is connected.

        While it does not hurt, there is technically no reason to push samples if there
        is no one connected.

        :type: :class:`bool`

        Notes
        -----
        This function does not filter the search for :class:`mne_lsl.lsl.StreamInlet`.
        Any application inlet will be recognized.
        """

    def get_sinfo(self) -> _BaseStreamInfo:
        """:class:`~mne_lsl.lsl.StreamInfo` corresponding to this Outlet.

        Returns
        -------
        sinfo : StreamInfo
            Description of the stream connected to the outlet.
        """
