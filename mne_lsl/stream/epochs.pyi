import numpy as np
from _typeshed import Incomplete
from mne import Info
from numpy.typing import NDArray

from .._typing import ScalarArray as ScalarArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..utils._checks import check_type as check_type
from ..utils._checks import check_value as check_value
from ..utils._checks import ensure_int as ensure_int
from ..utils._docs import fill_doc as fill_doc
from ..utils._fixes import find_events as find_events
from ..utils._time import high_precision_sleep as high_precision_sleep
from ..utils.logs import logger as logger
from ..utils.logs import warn as warn
from .base import BaseStream as BaseStream

class EpochsStream:
    """Stream object representing a single real-time stream of epochs.

    Note that a stream of epochs is necessarily connected to a regularly sampled stream
    of continuous data, from which epochs are extracted depending on an internal event
    channel or to an external event stream.

    Parameters
    ----------
    stream : ``Stream``
        Stream object to connect to, from which the epochs are extracted. The stream
        must be regularly sampled.
    bufsize : int
        Number of epochs to keep in the buffer. The buffer size is defined by this
        number of epochs and by the duration of individual epochs, defined by the
        argument ``tmin`` and ``tmax``.

        .. note::

            For a new epoch to be added to the buffer, the epoch must be fully
            acquired, i.e. the last sample of the epoch must be received. Thus, an
            epoch is acquired at least ``tmax`` seconds after the event onset.
    event_id : int | dict | None
        The ID of the events to consider from the event source. The event source can be
        a channel from the connected Stream or a separate event stream. In both case the
        event should be defined either as :class:`int`. If a :class:`dict` is provided,
        it should map event names to event IDs. For example
        ``dict(auditory=1, visual=2)``. If the event source is an irregularly sampled
        stream, the numerical values within the channels are ignored and this argument
        is ignored in which case it should be set to ``None``.
    event_channels : str | list of str
        Channel(s) to monitor for incoming events. The event channel(s) must be part of
        the connected Stream or of the ``event_stream`` if provided. See notes for
        details.
    event_stream : ``Stream`` | None
        Source from which events should be retrieved. If provided, event channels in the
        connected ``stream`` are ignored in favor of the event channels in this separate
        ``event_stream``. See notes for details.

        .. note::

            If a separate event stream is provided, time synchronization between the
            connected stream and the event stream is very important. For
            :class:`~mne_lsl.stream.StreamLSL` objects, provide
            ``processing_flags='all'`` as argument during connection with
            :meth:`~mne_lsl.stream.StreamLSL.connect`.
    tmin, tmax : float
        Start and end time of the epochs in seconds, relative to the time-locked
        event. The closest or matching samples corresponding to the start and end
        time are included. Defaults to ``-0.2`` and ``0.5``, respectively.
    baseline : None | tuple of length 2
        The time interval to consider as "baseline" when applying baseline
        correction. If ``None``, do not apply baseline correction.
        If a tuple ``(a, b)``, the interval is between ``a`` and ``b``
        (in seconds), including the endpoints.
        If ``a`` is ``None``, the **beginning** of the data is used; and if ``b``
        is ``None``, it is set to the **end** of the data.
        If ``(None, None)``, the entire time interval is used.

        .. note::
            The baseline ``(a, b)`` includes both endpoints, i.e. all timepoints ``t``
            such that ``a <= t <= b``.

        Correction is applied **to each epoch and channel individually** in the
        following way:

        1. Calculate the mean signal of the baseline period.
        2. Subtract this mean from the **entire** epoch.
    picks : str | array-like | slice | None
        Channels to include. Slices and lists of integers will be interpreted as
        channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
        ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick
        :term:`data channels`. None (default) will pick all channels.
    reject : dict | None
        Reject epochs based on **maximum** peak-to-peak signal amplitude (PTP),
        i.e. the absolute difference between the lowest and the highest signal
        value. In each individual epoch, the PTP is calculated for every channel.
        If the PTP of any one channel exceeds the rejection threshold, the
        respective epoch will be dropped.

        The dictionary keys correspond to the different channel types; valid
        **keys** can be any channel type present in the object.

        Example::

            reject = dict(
                grad=4000e-13,  # unit: T / m (gradiometers)
                mag=4e-12,  # unit: T (magnetometers)
                eeg=40e-6,  # unit: V (EEG channels)
                eog=250e-6,  # unit: V (EOG channels)
            )

        .. note:: Since rejection is based on a signal **difference**
                  calculated for each channel separately, applying baseline
                  correction does not affect the rejection procedure, as the
                  difference will be preserved.

        .. note:: To constrain the time period used for estimation of signal
                  quality, pass the ``reject_tmin`` and ``reject_tmax`` parameters.

        If ``reject`` is ``None`` (default), no rejection is performed.
    flat : dict | None
        Reject epochs based on **minimum** peak-to-peak signal amplitude (PTP).
        Valid **keys** can be any channel type present in the object. The
        **values** are floats that set the minimum acceptable PTP. If the PTP
        is smaller than this threshold, the epoch will be dropped. If ``None``
        then no rejection is performed based on flatness of the signal.

        .. note:: To constrain the time period used for estimation of signal
                  quality, pass the ``reject_tmin`` and ``reject_tmax`` parameters.
    reject_tmin, reject_tmax : float | None
        Start and end of the time window used to reject epochs based on
        peak-to-peak (PTP) amplitudes as specified via ``reject`` and ``flat``.
        The default ``None`` corresponds to the first and last time points of the
        epochs, respectively.

        .. note:: This parameter controls the time period used in conjunction with
                  both, ``reject`` and ``flat``.
    detrend : int | str | None
        The type of detrending to use. Can be ``'constant'`` or ``0`` for constant (DC)
        detrend, ``'linear'`` or ``1`` for linear detrend, or ``None`` for no
        detrending. Note that detrending is performed before baseline correction.

    Notes
    -----
    Since events can be provided from multiple source, the arguments ``event_channels``,
    ``event_source`` and ``event_id`` must work together to select which events should
    be considered.

    - if ``event_stream`` is ``None``, the events are extracted from channels within the
      connected ``stream``. This ``stream`` is necessarily regularly sampled, thus the
      event channels must correspond to MNE ``'stim'`` channels, i.e. channels on which
      :func:`mne.find_events` can be applied.
    - if ``event_stream`` is provided and is regularly sampled, the events are extracted
      from channels in the ``event_stream``. The event channels must correspond to MNE
      ``'stim'`` channels, i.e. channels on which :func:`mne.find_events` can be
      applied.
    - if ``event_stream`` is provided and is irregularly sampled, the events are
      extracted from channels in the ``event_stream``. The numerical value within the
      channels are ignored and the appearance of a new value in the stream is considered
      as a new event named after the channel name. Thus, the argument ``event_id`` is
      ignored. This last case can be useful when working with a ``Player`` replaying
      annotations from a file as one-hot encoded events.

    Event streams irregularly sampled and a ``str`` datatype are not yet supported.

    .. note::

        In the 2 last cases where ``event_stream`` is provided, all ``'stim'`` channels
        in the connected ``stream`` are ignored.

    Read about the :ref:`processing applied to the underlying
    buffer <resources/implementations:EpochsStream>`.
    """

    _stream: Incomplete
    _tmin: Incomplete
    _tmax: Incomplete
    _event_stream: Incomplete
    _event_channels: Incomplete
    _bufsize: Incomplete
    _event_id: Incomplete
    _baseline: Incomplete
    _detrend_type: Incomplete
    _picks_init: Incomplete
    _times: Incomplete

    def __init__(
        self,
        stream: BaseStream,
        bufsize: int,
        event_id: int | dict[str, int] | None,
        event_channels: str | list[str],
        event_stream: BaseStream | None = None,
        tmin: float = -0.2,
        tmax: float = 0.5,
        baseline: tuple[float | None, float | None] | None = (None, 0),
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        reject: dict[str, float] | None = None,
        flat: dict[str, float] | None = None,
        reject_tmin: float | None = None,
        reject_tmax: float | None = None,
        detrend: int | str | None = None,
    ) -> None: ...
    def __del__(self) -> None:
        """Delete the epoch stream object."""

    def __repr__(self) -> str:
        """Representation of the instance."""

    def acquire(self) -> None:
        """Pull new epochs in the buffer.

        This method is used to manually acquire new epochs in the buffer. If used, it is
        up to the user to call this method at the desired frequency, else it might miss
        some of the events and associated epochs.

        Notes
        -----
        This method is not needed if the :class:`mne_lsl.stream.EpochsStream` was
        connected with an acquisition delay different from ``0``. In this case, the
        acquisition is done automatically in a background thread.
        """
    _acquisition_delay: Incomplete
    _picks: Incomplete
    _info: Incomplete
    _tmin_shift: Incomplete
    _ch_idx_by_type: Incomplete
    _buffer: Incomplete
    _buffer_events: Incomplete
    _executor: Incomplete

    def connect(self, acquisition_delay: float = 0.001) -> EpochsStream:
        """Start acquisition of epochs from the connected Stream.

        Parameters
        ----------
        acquisition_delay : float
            Delay in seconds between 2 updates at which the event stream is queried for
            new events, and thus at which the epochs are updated.

            .. note::

                For a new epoch to be added to the buffer, the epoch must be fully
                acquired, i.e. the last sample of the epoch must be received. Thus, an
                epoch is acquired ``tmax`` seconds after the event onset.

        Returns
        -------
        epochs_stream : instance of EpochsStream
            The :class:`~mne_lsl.stream.EpochsStream` instance modified in-place.
        """

    def disconnect(self) -> EpochsStream:
        """Stop acquisition of epochs from the connected Stream.

        Returns
        -------
        epochs : instance of :class:`~mne_lsl.stream.EpochsStream`
            The epochs instance modified in-place.
        """
    _n_new_epochs: int

    def get_data(
        self,
        n_epochs: int | None = None,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        exclude: str | list[str] | tuple[str] = "bads",
    ) -> ScalarArray:
        """Retrieve the latest epochs from the buffer.

        Parameters
        ----------
        n_epochs : int | None
            Number of epochs to retrieve from the buffer. If None, all epochs are
            returned.
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Note that
            channels in ``info['bads']`` *will be included* if their names or indices
            are explicitly provided.
        exclude : str | list of str | tuple of str
            Set of channels to exclude, only used when picking based on types (e.g.,
            ``exclude="bads"`` when ``picks="meg"``) or when picking is set to ``None``.

        Returns
        -------
        data : array of shape (n_epochs, n_channels, n_samples)
            Data in the buffer.

        Notes
        -----
        The number of newly available epochs stored in the property ``n_new_epochs``
        is reset at every function call, even if all channels were not selected with the
        argument ``picks``.
        """
    _last_ts: Incomplete

    def _acquire(self) -> None:
        """Update function looking for new epochs."""

    def _check_connected(self, name: str) -> None:
        """Check that the epochs stream is connected before calling 'name'."""

    def _reset_variables(self) -> None:
        """Reset variables defined after connection."""

    def _submit_acquisition_job(self) -> None:
        """Submit a new acquisition job, if applicable."""

    @property
    def connected(self) -> bool:
        """Connection status of the :class:`~mne_lsl.stream.EpochsStream`.

        :type: :class:`bool`
        """

    @property
    def events(self) -> NDArray[np.int16]:
        """Events of the epoched LSL stream.

        Contrary to the events stored in ``mne.Epochs.events``, only the integer code
        of the event is stored in a :class:`~mne_lsl.stream.EpochsStream` object.

        :type: :class:`numpy.ndarray`
        """

    @property
    def info(self) -> Info:
        """Info of the epoched LSL stream.

        :type: :class:`~mne.Info`
        """

    @property
    def n_new_epochs(self) -> int:
        """Number of new epochs available in the buffer.

        The number of new epochs is reset at every ``Stream.get_data`` call.

        :type: :class:`int`
        """

    @property
    def times(self) -> NDArray[np.float64]:
        """The time of each sample in the epochs.

        :type: :class:`numpy.ndarray`
        """

def _check_event_channels(
    event_channels: list[str], stream: BaseStream, event_stream: BaseStream | None
) -> None:
    """Check that the event channels are valid."""

def _ensure_event_id(
    event_id: int | dict[str, int] | None, event_stream: BaseStream | None
) -> dict[str, int] | None:
    """Ensure event_ids is a dictionary or None."""

def _check_baseline(
    baseline: tuple[float | None, float | None] | None, tmin: float, tmax: float
) -> None:
    """Check that the baseline is valid."""

def _check_reject_flat(
    reject: dict[str, float] | None, flat: dict[str, float] | None, info: Info
) -> None:
    """Check that the PTP rejection dictionaries are valid."""

def _check_reject_tmin_tmax(
    reject_tmin: float | None, reject_tmax: float | None, tmin: float, tmax: float
) -> None:
    """Check that the rejection time window is valid."""

def _ensure_detrend_str(detrend: int | str | None) -> str | None:
    """Ensure detrend is an integer."""

def _find_events_in_stim_channels(
    data: ScalarArray,
    event_channels: list[str],
    sfreq: float,
    *,
    output: str = "onset",
    consecutive: bool | str = "increasing",
    min_duration: float = 0,
    shortest_event: int = 2,
    mask: int | None = None,
    uint_cast: bool = False,
    mask_type: str = "and",
    initial_event: bool = False,
) -> NDArray[np.int64]:
    """Find events in stim channels."""

def _prune_events(
    events: NDArray[np.int64],
    event_id: dict[str, int] | None,
    buffer_size: int,
    ts: NDArray[np.float64],
    last_ts: float | None,
    ts_events: NDArray[np.float64] | None,
    tmin_shift: float,
) -> NDArray[np.int64]:
    """Prune events based on criteria and buffer size."""

def _process_data(
    data: ScalarArray,
    baseline: tuple[float | None, float | None] | None,
    reject: dict[str, float] | None,
    flat: dict[str, float] | None,
    reject_tmin: float | None,
    reject_tmax: float | None,
    detrend_type: str | None,
    times: NDArray[np.float64],
    ch_idx_by_type: dict[str, list[int]],
) -> ScalarArray:
    """Apply the requested processing to the new epochs."""

def _remove_empty_elements(
    data: ScalarArray, ts: NDArray[np.float64]
) -> tuple[ScalarArray, NDArray[np.float64]]:
    """Remove empty elements from the data and ts array."""
