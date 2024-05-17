from __future__ import annotations  # c.f. PEP 563, PEP 649

from math import ceil
from threading import Timer
from typing import TYPE_CHECKING

import numpy as np
from mne import pick_info
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.pick import _picks_to_idx
elif check_version("mne", "1.5"):
    from mne.io.pick import _picks_to_idx
else:
    from mne.io.pick import _picks_to_idx

from ..utils._checks import check_type, ensure_int
from ..utils._docs import fill_doc
from ..utils.logs import logger, warn
from ._base import BaseStream

if TYPE_CHECKING:
    from typing import Optional, Union

    from mne import Info

    from .._typing import ScalarIntArray


@fill_doc
class EpochsStream:
    """Stream object representing a single real-time stream of epochs.

    Note that a stream of epochs is necessarily connected to a regularly sampled stream
    of continuous data, from which epochs are extracted depending on an internal event
    channel or to an external event stream.

    Parameters
    ----------
    stream : ``Stream``
        Stream object to connect to, from which the epochs are extracted. The stream
        must ber regularly sampled.
    event_channels : str | list of str
        Channel(s) to monitor for incoming events. The event channel(s) must be part of
        the connected Stream of of the ``event_stream`` if provided. See notes for
        details.
    event_stream : ``Stream`` | None
        Source from which events should be retrieved. If provided, event channels in the
        connected ``stream`` are ignored in favor of the event channels in this separate
        ``event_stream``. See notes for details.
    event_id : int | str | dict
        The ID of the events to consider from the event source. The event source can be
        a channel from the connected Stream, in which case the event should be defined
        as :class:`int`, or a separate event stream, in which case the event should be
        defined either as :class:`int` or :class:`str`. If a :class:`dict` is provided,
        it should map event names to event IDs. For example
        ``dict(auditory=1, visual=2)``.
    bufsize : int
        Number of epochs to keep in the buffer.
    %(epochs_tmin_tmax)s
    %(baseline_epochs)s
    %(picks_base)s all channels.
    %(reject_epochs)s
    %(flat)s
    %(epochs_reject_tmin_tmax)s
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
      extracted from channels in the ``event_stream``. If the stream ``dtype`` is
      :class:`str`, the value within the channels are used as separate events. If the
      stream ``dtype`` is ``numerical``, the value within the channels are ignored and
      the appearance of a new value in the stream is considered as a new event named
      after the channel name. This last case can be useful when working with a
      ``Player`` replaying annotations from a file as one-hot encoded events.

    .. note::

        In the 2 last cases where ``event_stream`` is provided, all channels in the
        connected ``stream`` are ignored.
    """

    def __init__(
        self,
        stream: BaseStream,
        event_channels: Union[str, list[str]],
        event_stream: Optional[BaseStream],
        event_id: Union[int, str, dict[str, Union[int, str]]],
        bufsize: int,
        tmin: float = -0.2,
        tmax: float = 0.5,
        baseline: Optional[tuple[Optional[float], Optional[float]]] = (None, 0),
        picks: Optional[Union[str, list[str], int, list[int], ScalarIntArray]] = None,
        reject: Optional[dict[str, float]] = None,
        flat: Optional[dict[str, float]] = None,
        reject_tmin: Optional[float] = None,
        reject_tmax: Optional[float] = None,
        detrend: Optional[Union[int, str]] = None,
    ) -> None:
        check_type(stream, (BaseStream,), "stream")
        if not stream.connected and stream._info["sfreq"] != 0:
            raise RuntimeError(
                "The Stream must be a connected regularly sampled stream before "
                "creating an EpochsStream."
            )
        self._stream = stream
        check_type(tmin, ("numeric",), "tmin")
        check_type(tmax, ("numeric",), "tmax")
        if tmax <= tmin:
            raise ValueError(
                f"Argument 'tmax' (provided: {tmax}) must be greater than 'tmin' "
                f"(provided: {tmin})."
            )
        # make sure the stream buffer is long enough to store an entire epoch, which is
        # simpler than handling the case where the buffer is too short and we need to
        # concatenate chunks to form a single epoch.
        if stream._bufsize < tmax - tmin:
            raise ValueError(
                "The buffer size of the Stream must be at least as long as the epoch "
                "duration (tmax - tmin)."
            )
        elif stream._bufsize < (tmax - tmin) * 1.2:
            warn(
                "The buffer size of the Stream is longer than the epoch duration, but "
                "not by at least 20%. It is recommended to have a buffer size at least "
                r"20% longer than the epoch duration to avoid data loss."
            )
        self._tmin = tmin
        self._tmax = tmax
        # check the event source(s)
        check_type(event_stream, (BaseStream, None), "event_stream")
        if event_stream is not None and not event_stream.connected:
            raise RuntimeError(
                "If 'event_stream' is provided, it must be connected before creating "
                "an EpochsStream."
            )
        self._event_stream = event_stream
        event_channels = (
            [event_channels] if isinstance(event_channels, str) else event_channels
        )
        check_type(event_channels, (list,), "event_channels")
        for elt in event_channels:
            check_type(elt, (str,), "event_channels")
            if self._event_stream is None and elt not in stream.ch_names:
                raise ValueError(
                    "The event channel(s) must be part of the connected Stream if an "
                    f"'event_stream' is not provided. '{elt}' was not found."
                )
            elif self._event_stream is not None and elt not in event_stream.ch_names:
                raise ValueError(
                    "If 'event_stream' is provided, the event channel(s) must be part "
                    f"of 'event_stream'. '{elt}' was not found."
                )
        self._event_channels = event_channels
        # check and store the epochs general settings
        self._bufsize = ensure_int(bufsize, "bufsize")
        if self._bufsize <= 0:
            raise ValueError(
                "The buffer size, i.e. the number of epochs in the buffer, must be a "
                "positive integer."
            )
        self._event_id = _ensure_event_id_dict(event_id)
        _check_baseline(baseline)
        self._baseline = baseline
        _check_reject_flat(reject, flat, stream.info)
        self._reject, self._flat = reject, flat
        _check_reject_tmin_tmax(reject_tmin, reject_tmax, tmin, tmax)
        self._reject_tmin, self._reject_tmax = reject_tmin, reject_tmax
        self._detrend = _ensure_detrend_int(detrend)
        # initialize the epoch buffer
        self._picks = _picks_to_idx(
            self._stream._info, picks, "all", "bads", allow_empty=False
        )
        self._info = pick_info(self._stream._info, self._picks)
        # define acquisition variables which need to be reset on disconnect
        self._reset_variables()
        # mark the stream(s) as being epoched, which will prevent further channel
        # modification and buffer size modifications.
        self._stream._epochs.append(self)
        if self._event_stream is not None:
            self._event_stream._epochs.append(self)

    def __del__(self) -> None:
        """Delete the epoch stream object."""
        logger.debug("Deleting %s", self)
        try:
            self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        """Representation of the instance."""
        try:
            status = "ON" if self.connected else "OFF"
        except Exception:
            status = "OFF"
        return (
            f"<EpochsStream {status} (n: {self._bufsize} between ({self._tmin}, "
            f"{self._tmax}) seconds> connected to\n\t{self._stream}"
        )

    def connect(self, acquisition_delay: float = 0.01) -> EpochsStream:
        """Start acquisition of epochs from the connected Stream.

        Parameters
        ----------
        acquisition_delay : float
            Delay in seconds between 2 updates at which the event stream is queried for
            new events, and thus at which the epochs are updated.

        Returns
        -------
        epochs_stream : instance of EpochsStream
            The :class:`~mne_lsl.stream.EpochsStream` instance modified in-place.
        """
        if self.connected:
            warn("The stream is already connected. Skipping.")
            return self
        check_type(acquisition_delay, ("numeric",), "acquisition_delay")
        if acquisition_delay < 0:
            raise ValueError(
                "The acquisition delay must be a positive number "
                "defining the delay at which the epochs might be updated in seconds. "
                "For instance, 0.2 corresponds to a query to the event source every "
                f"200 ms. The provided {acquisition_delay} is invalid."
            )
        self._acquisition_delay = acquisition_delay
        self._n_new_epochs = 0
        # create the buffer and start acquisition in a separate thread
        self._buffer = np.zeros(
            (
                self._bufsize,
                ceil((self._tmax - self._tmin) * self._info["sfreq"]),
                self._picks.size,
            ),
            dtype=self._stream._buffer.dtype,
        )
        self._create_acquisition_thread(0)
        return self

    def disconnect(self) -> None:
        """Stop acquisition of epochs from the connected Stream."""
        if not self.connected:
            warn("The stream is already disconnected. Skipping.")
            return
        self._interrupt = True
        while self._acquisition_thread.is_alive():
            self._acquisition_thread.cancel()
        self._reset_variables()
        self._stream._epoched -= 1
        if self._event_stream is not None:
            self._event_stream._epoched -= 1

    def _acquire(self) -> None:
        """Update function looking for new epochs."""

    def _create_acquisition_thread(self, delay: float) -> None:
        """Create and start the daemonic acquisition thread.

        Parameters
        ----------
        delay : float
            Delay after which the thread will call the acquire function.
        """
        self._acquisition_thread = Timer(delay, self._acquire)
        self._acquisition_thread.daemon = True
        self._acquisition_thread.start()

    def _reset_variables(self):
        """Reset variables defined after connection."""
        self._acquisition_thread = None
        self._acquisition_delay = None
        self._buffer = None
        self._interrupt = False
        self._n_new_epochs = 0

    # ----------------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        """Connection status of the :class:`~mne_lsl.stream.EpochsStream`.

        :type: :class:`bool`
        """
        attributes = (
            "_acquisition_delay",
            "_acquisition_thread",
            "_buffer",
        )
        if all(getattr(self, attr, None) is None for attr in attributes):
            return False
        else:
            # sanity-check
            assert not any(getattr(self, attr, None) is None for attr in attributes)
            return True

    @property
    def info(self) -> Info:
        """Info of the epoched LSL stream.

        :type: :class:`~mne.Info`
        """
        return self._info


def _ensure_event_id_dict(
    event_id: Union[int, str, dict[str, Union[int, str]]],
) -> dict[str, Union[str, int]]:
    """Ensure event_ids is a dictionary."""
    check_type(event_id, (int, str, dict), "event_id")
    raise_ = False
    if isinstance(event_id, str):
        if len(event_id) == 0:
            raise_ = True
        event_id = {event_id: event_id}
    elif isinstance(event_id, int):
        if event_id <= 0:
            raise_ = True
        event_id = {str(event_id): event_id}
    else:
        for key, value in event_id.items():
            check_type(key, (str,), "event_id")
            check_type(value, (int, str), "event_id")
            if len(key) == 0:
                raise_ = True
            if (isinstance(value, str) and len(value) == 0) or (
                isinstance(value, int) and value <= 0
            ):
                raise_ = True
    if raise_:
        raise ValueError(
            "The 'event_id' must be a non-empty string, a positive integer or a "
            "dictionary mapping non-empty strings to positive integers or non-empty "
            "strings."
        )
    return event_id


def _check_baseline(
    baseline: Optional[tuple[Optional[float], Optional[float]]],
    tmin: float,
    tmax: float,
) -> None:
    """Check that the baseline is valid."""
    check_type(baseline, (tuple, None), "baseline")
    if baseline is None:
        return
    if len(baseline) != 2:
        raise ValueError("The baseline must be a tuple of 2 elements.")
    check_type(baseline[0], ("numeric", None), "baseline[0]")
    check_type(baseline[1], ("numeric", None), "baseline[1]")
    if baseline[0] is not None and baseline[0] < tmin:
        raise ValueError(
            "The beginning of the baseline period must be greater than or equal to "
            "the beginning of the epoch period 'tmin'."
        )
    if baseline[1] is not None and tmax < baseline[1]:
        raise ValueError(
            "The end of the baseline period must be less than or equal to the end of "
            "the epoch period 'tmax'."
        )


def _check_reject_flat(
    reject: Optional[dict[str, float]], flat: Optional[dict[str, float]], info: Info
) -> None:
    """Check that the PTP rejection dictionaries are valid."""
    check_type(reject, (dict, None), "reject")
    check_type(flat, (dict, None), "flat")
    ch_types = info.get_channel_types(unique=True)
    if reject is not None:
        for key, value in reject.items():
            check_type(key, (str,), "reject")
            check_type(value, ("numeric",), "reject")
            if key not in ch_types:
                raise ValueError(
                    f"The channel type '{key}' in the rejection dictionary is not part "
                    "of the connected Stream."
                )
            check_type(value, (float,), "reject")
            if value <= 0:
                raise ValueError(
                    f"The peak-to-peak rejection value for channel type '{key}' must "
                    "be a positive number."
                )
    if flat is not None:
        for key, value in flat.items():
            check_type(key, (str,), "flat")
            check_type(value, ("numeric",), "flat")
            if key not in ch_types:
                raise ValueError(
                    f"The channel type '{key}' in the flat rejection dictionary is not "
                    "part of the connected Stream."
                )
            check_type(value, (float,), "flat")
            if value <= 0:
                raise ValueError(
                    f"The flat rejection value for channel type '{key}' must be a "
                    "positive number."
                )


def _check_reject_tmin_tmax(
    reject_tmin: Optional[float], reject_tmax: Optional[float], tmin: float, tmax: float
) -> None:
    """Check that the rejection time window is valid."""
    check_type(reject_tmin, ("numeric", None), "reject_tmin")
    check_type(reject_tmax, ("numeric", None), "reject_tmax")
    if reject_tmin is not None and reject_tmin < tmin:
        raise ValueError(
            "The beginning of the rejection time window must be greater than or equal "
            "to the beginning of the epoch period 'tmin'."
        )
    if reject_tmax is not None and tmax < reject_tmax:
        raise ValueError(
            "The end of the rejection time window must be less than or equal to the "
            "end of the epoch period 'tmax'."
        )
    if (
        reject_tmin is not None
        and reject_tmax is not None
        and reject_tmax <= reject_tmin
    ):
        raise ValueError(
            "The end of the rejection time window must be greater than the beginning "
            "of the rejection time window."
        )


def _ensure_detrend_int(detrend: Optional[Union[int, str]]) -> Optional[int]:
    """Ensure detrend is an integer."""
    if detrend is None:
        return None
    if isinstance(detrend, str):
        if detrend == "constant":
            return 0
        elif detrend == "linear":
            return 1
        else:
            raise ValueError(
                "The detrend argument must be 'constant', 'linear' or their integer "
                "equivalent 0 and 1."
            )
    detrend = ensure_int(detrend, "detrend")
    if detrend not in (0, 1):
        raise ValueError(
            "The detrend argument must be 'constant', 'linear' or their integer "
            "equivalent 0 and 1."
        )
    return detrend
