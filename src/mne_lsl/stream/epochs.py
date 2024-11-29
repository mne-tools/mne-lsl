from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from mne import pick_info
from mne.event import _find_unique_events
from mne.utils import check_version
from scipy.signal import detrend

if check_version("mne", "1.6"):
    from mne._fiff.pick import _picks_to_idx, channel_indices_by_type

elif check_version("mne", "1.5"):
    from mne.io.pick import _picks_to_idx, channel_indices_by_type

else:
    from mne.io.pick import _picks_to_idx, channel_indices_by_type

from ..utils._checks import check_type, check_value, ensure_int
from ..utils._docs import fill_doc
from ..utils._fixes import find_events
from ..utils._time import high_precision_sleep
from ..utils.logs import logger, warn
from .base import BaseStream

if TYPE_CHECKING:
    from mne import Info
    from numpy.typing import NDArray

    from .._typing import ScalarArray, ScalarIntArray


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
    ) -> None:
        check_type(stream, (BaseStream,), "stream")
        if not stream.connected or stream._info["sfreq"] == 0:
            raise RuntimeError(
                "The Stream must be a connected regularly sampled stream before "
                "creating an EpochsStream."
            )
        self._stream = stream
        # mark the stream(s) as being epoched, which will prevent further channel
        # modification and buffer size modifications.
        self._stream._epochs.append(self)
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
        if self._stream._bufsize < tmax - tmin:
            raise ValueError(
                "The buffer size of the Stream must be at least as long as the epoch "
                "duration (tmax - tmin)."
            )
        elif self._stream._bufsize < (tmax - tmin) * 1.2:
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
        if self._event_stream is not None:
            self._event_stream._epochs.append(self)
        event_channels = (
            [event_channels] if isinstance(event_channels, str) else event_channels
        )
        check_type(event_channels, (list,), "event_channels")
        _check_event_channels(event_channels, stream, event_stream)
        self._event_channels = event_channels
        # check and store the epochs general settings
        self._bufsize = ensure_int(bufsize, "bufsize")
        if self._bufsize <= 0:
            raise ValueError(
                "The buffer size, i.e. the number of epochs in the buffer, must be a "
                "positive integer."
            )
        self._event_id = _ensure_event_id(event_id, event_stream)
        _check_baseline(baseline, self._tmin, self._tmax)
        self._baseline = baseline
        _check_reject_flat(reject, flat, self._stream._info)
        self._reject, self._flat = reject, flat
        _check_reject_tmin_tmax(reject_tmin, reject_tmax, tmin, tmax)
        self._reject_tmin, self._reject_tmax = reject_tmin, reject_tmax
        self._detrend_type = _ensure_detrend_str(detrend)
        # store picks which are then initialized in the connect method
        self._picks_init = picks
        # define the times array based on tmin, tmax and the sampling frequency
        n_samples = ceil((self._tmax - self._tmin) * self._stream._info["sfreq"])
        self._times = np.linspace(
            self._tmin, self._tmax, n_samples, endpoint=False, dtype=np.float64
        )
        # define acquisition variables which need to be reset on disconnect
        self._reset_variables()

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
        repr_ = f"<EpochsStream {status}"
        if (
            hasattr(self, "_bufsize")
            and hasattr(self, "_tmin")
            and hasattr(self, "_tmax")
        ):
            repr_ += (
                f" (n: {self._bufsize} between ({self._tmin}, {self._tmax} seconds)"
            )
        repr_ += ">"
        if hasattr(self, "_stream"):
            repr_ += f" connected to:\n\t{self._stream}"
        return repr_

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
        self._check_connected("acquire")
        if (
            self._executor is not None and self._acquisition_delay == 0
        ):  # pragma: no cover
            raise RuntimeError(
                "The executor is not None despite the acquisition delay set to "
                f"{self._acquisition_delay} seconds. This should not happen, please "
                "contact the developers on GitHub."
            )
        elif self._executor is not None and self._acquisition_delay != 0:
            raise RuntimeError(
                "Acquisition is done automatically in a background thread. The method "
                "epochs.acquire() should not be called."
            )
        self._acquire()

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
        if self.connected:
            warn("The EpochsStream is already connected. Skipping.")
            return self
        if not self._stream.connected:
            raise RuntimeError(
                "The Stream was disconnected between initialization and connection "
                "of the EpochsStream object."
            )
        if self._event_stream is not None and not self._event_stream.connected:
            raise RuntimeError(
                "The event stream was disconnected between initialization and "
                "connection of the EpochsStream object."
            )
        check_type(acquisition_delay, ("numeric",), "acquisition_delay")
        if acquisition_delay < 0:
            raise ValueError(
                "The acquisition delay must be a positive number defining the delay at "
                "which the epochs might be updated in seconds. For instance, 0.2 "
                "corresponds to a query to the event source every 200 ms. 0 "
                f"corresponds to manual acquisition. The provided {acquisition_delay} "
                "is invalid."
            )
        self._acquisition_delay = acquisition_delay
        assert self._n_new_epochs == 0  # sanity-check
        # create the buffer and start acquisition in a separate thread
        self._picks = _picks_to_idx(
            self._stream._info, self._picks_init, "all", "bads", allow_empty=False
        )
        self._info = pick_info(self._stream._info, self._picks)
        self._tmin_shift = round(self._tmin * self._info["sfreq"])
        self._ch_idx_by_type = channel_indices_by_type(self._info)
        self._buffer = np.zeros(
            (
                self._bufsize,
                self._times.size,
                self._picks.size,
            ),
            dtype=self._stream._buffer.dtype,
        )
        self._buffer_events = np.zeros(self._bufsize, dtype=np.int16)
        self._executor = (
            ThreadPoolExecutor(max_workers=1) if self._acquisition_delay != 0 else None
        )
        # submit the first acquisition job
        if self._executor is not None:
            logger.debug("%s: ThreadPoolExecutor started.", self)
            self._executor.submit(self._acquire)
        return self

    def disconnect(self) -> EpochsStream:
        """Stop acquisition of epochs from the connected Stream.

        Returns
        -------
        epochs : instance of :class:`~mne_lsl.stream.EpochsStream`
            The epochs instance modified in-place.
        """
        if (
            hasattr(self, "_stream")
            and hasattr(self._stream, "_epochs")
            and self in self._stream._epochs
        ):
            self._stream._epochs.remove(self)
        if (
            hasattr(self, "_event_stream")
            and self._event_stream is not None
            and hasattr(self._event_stream, "_epochs")
            and self in self._event_stream._epochs
        ):
            self._event_stream._epochs.remove(self)
        if not self.connected:
            logger.info("The EpochsStream %s is already disconnected. Skipping.", self)
            return
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._reset_variables()
        return self

    @fill_doc
    def get_data(
        self,
        n_epochs: int | None = None,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        exclude: str | list[str] | tuple[str, ...] = "bads",
    ) -> ScalarArray:
        """Retrieve the latest epochs from the buffer.

        Parameters
        ----------
        n_epochs : int | None
            Number of epochs to retrieve from the buffer. If None, all epochs are
            returned.
        %(picks_all)s
        %(exclude)s

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
        try:
            picks = _picks_to_idx(self._info, picks, none="all", exclude=exclude)
            n_epochs = self._buffer.shape[0] if n_epochs is None else n_epochs
            if n_epochs <= 0:
                raise ValueError(
                    "The number of epochs to retrieve must be a positive integer. "
                    f"{n_epochs} is invalid."
                )
            if self._buffer.shape[0] < n_epochs:
                warn(
                    f"The number of epochs requested {n_epochs} is greater than the "
                    f"buffer size {self._buffer.shape[0]}. Selecting the entire buffer."
                )
                n_epochs = self._buffer.shape[0]
            self._n_new_epochs = 0  # reset the number of new epochs
            return np.transpose(self._buffer[-n_epochs:, :, picks], axes=(0, 2, 1))
        except Exception:
            if not self.connected:
                raise RuntimeError(
                    "The EpochsStream is not connected. Please connect it before "
                    "retrieving data from the buffer."
                )
            else:  # pragma: no cover
                logger.error(
                    "Something went wrong while retrieving data from a connected "
                    "EpochsStream. Please open an issue on GitHub and provide the "
                    "error traceback to the developers."
                )
            raise  # pragma: no cover

    def _acquire(self) -> None:
        """Update function looking for new epochs."""
        try:
            if self._stream._n_new_samples == 0 or (
                self._event_stream is not None
                and self._event_stream._n_new_samples == 0
            ):
                self._submit_acquisition_job()
                return
            # split the different acquisition scenarios to retrieve new events to add to
            # the buffer.
            data, ts = _remove_empty_elements(
                self._stream._buffer.T, self._stream._timestamps
            )
            if self._event_stream is None:
                picks_events = _picks_to_idx(
                    self._stream._info, self._event_channels, exclude="bads"
                )
                events = _find_events_in_stim_channels(
                    data[picks_events, :], self._event_channels, self._info["sfreq"]
                )
                events = _prune_events(
                    events,
                    self._event_id,
                    self._buffer.shape[1],
                    ts,
                    self._last_ts,
                    None,
                    self._tmin_shift,
                )
            elif (
                self._event_stream is not None
                and self._event_stream._info["sfreq"] != 0
            ):
                picks = _picks_to_idx(
                    self._event_stream._info,
                    self._event_channels,
                    none="all",
                    exclude=(),
                )
                data_events, ts_events = _remove_empty_elements(
                    self._event_stream._buffer[:, picks].T,
                    self._event_stream._timestamps,
                )
                events = _find_events_in_stim_channels(
                    data_events, self._event_channels, self._info["sfreq"]
                )
                events = _prune_events(
                    events,
                    self._event_id,
                    self._buffer.shape[1],
                    ts,
                    self._last_ts,
                    ts_events,
                    self._tmin_shift,
                )
            elif (
                self._event_stream is not None
                and self._event_stream._info["sfreq"] == 0
            ):
                # don't select only the new events as they might all fall outside of
                # the attached stream ts buffer, instead always look through all
                # available events.
                picks = _picks_to_idx(
                    self._event_stream._info,
                    self._event_channels,
                    none="all",
                    exclude=(),
                )
                data_events, ts_events = _remove_empty_elements(
                    self._event_stream._buffer[:, picks].T,
                    self._event_stream._timestamps,
                )
                events = np.vstack(
                    [
                        np.arange(ts_events.size, dtype=np.int64),
                        np.zeros(ts_events.size, dtype=np.int64),
                        np.argmax(data_events, axis=0),
                    ],
                    dtype=np.int64,
                ).T
                events = _prune_events(
                    events,
                    None,
                    self._buffer.shape[1],
                    ts,
                    self._last_ts,
                    ts_events,
                    self._tmin_shift,
                )
            else:  # pragma: no cover
                raise RuntimeError(
                    "This acquisition scenario should not happen. Please contact the "
                    "developers."
                )
            if events.shape[0] == 0:  # abort in case we don't have new events to add
                self._submit_acquisition_job()
                return
            self._last_ts = ts[events[-1, 0]]
            # select data, for loop is faster than the fancy indexing ideas tried and
            # will anyway operate on a small number of events most of the time.
            data_selection = np.empty(
                (
                    min(events.shape[0], self._bufsize),
                    self._buffer.shape[1],
                    self._picks.size,
                ),
                dtype=data.dtype,
            )
            for k, start in enumerate(events[:, 0][::-1]):
                start += self._tmin_shift
                data_selection[-(k + 1)] = data[
                    self._picks, start : start + self._buffer.shape[1]
                ].T
            # apply processing
            data_selection = _process_data(
                data_selection,
                self._baseline,
                self._reject,
                self._flat,
                self._reject_tmin,
                self._reject_tmax,
                self._detrend_type,
                self._times,
                self._ch_idx_by_type,
            )
            if data_selection.shape[0] == 0:
                self._submit_acquisition_job()
                return
            # roll buffer and add new epochs
            self._buffer = np.roll(self._buffer, -events.shape[0], axis=0)
            self._buffer[-events.shape[0] :, :, :] = data_selection
            self._buffer_events = np.roll(self._buffer_events, -events.shape[0])
            self._buffer_events[-events.shape[0] :] = events[:, 2]
            # update the last ts and the number of new epochs
            self._n_new_epochs += events.shape[0]
        except Exception as error:  # pragma: no cover
            logger.exception(error)
            self._reset_variables()
            if os.getenv("MNE_LSL_RAISE_STREAM_ERRORS", "false").lower() == "true":
                raise error
        else:
            self._submit_acquisition_job()

    def _check_connected(self, name: str) -> None:
        """Check that the epochs stream is connected before calling 'name'."""
        if not self.connected:
            raise RuntimeError(
                "The EpochsStream is not connected. Please connect to the EpochsStream "
                "with the method epochs.connect(...) to use "
                f"{type(self).__name__}.{name}."
            )

    def _reset_variables(self):
        """Reset variables defined after connection."""
        self._acquisition_delay = None
        self._buffer = None
        self._buffer_events = None
        self._ch_idx_by_type = None
        self._executor = None
        self._info = None
        self._last_ts = None
        self._n_new_epochs = 0
        self._picks = None
        self._tmin_shift = None

    def _submit_acquisition_job(self) -> None:
        """Submit a new acquisition job, if applicable."""
        if self._executor is None:
            return  # either shutdown or manual acquisition
        high_precision_sleep(self._acquisition_delay)
        try:
            self._executor.submit(self._acquire)
        except RuntimeError:  # pragma: no cover
            pass  # shutdown

    # ----------------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        """Connection status of the :class:`~mne_lsl.stream.EpochsStream`.

        :type: :class:`bool`
        """
        attributes = (
            "_acquisition_delay",
            "_buffer",
            "_ch_idx_by_type",
            "_info",
            "_picks",
        )
        if all(getattr(self, attr, None) is None for attr in attributes):
            return False
        else:
            # sanity-check
            assert not any(getattr(self, attr, None) is None for attr in attributes)
            return True

    @property
    def events(self) -> NDArray[np.int16]:
        """Events of the epoched LSL stream.

        Contrary to the events stored in ``mne.Epochs.events``, only the integer code
        of the event is stored in a :class:`~mne_lsl.stream.EpochsStream` object.

        :type: :class:`numpy.ndarray`
        """
        self._check_connected("events")
        return self._buffer_events

    @property
    def info(self) -> Info:
        """Info of the epoched LSL stream.

        :type: :class:`~mne.Info`
        """
        self._check_connected("info")
        return self._info

    @property
    def n_new_epochs(self) -> int:
        """Number of new epochs available in the buffer.

        The number of new epochs is reset at every ``Stream.get_data`` call.

        :type: :class:`int`
        """
        self._check_connected("n_new_epochs")
        return self._n_new_epochs

    @property
    def times(self) -> NDArray[np.float64]:
        """The time of each sample in the epochs.

        :type: :class:`numpy.ndarray`
        """
        return self._times


def _check_event_channels(
    event_channels: list[str],
    stream: BaseStream,
    event_stream: BaseStream | None,
) -> None:
    """Check that the event channels are valid."""
    for elt in event_channels:
        check_type(elt, (str,), "event_channels")
        if event_stream is None:
            if elt not in stream._info.ch_names:
                raise ValueError(
                    "The event channel(s) must be part of the connected Stream if "
                    f"an 'event_stream' is not provided. '{elt}' was not found."
                )
            if elt in stream._info["bads"]:
                raise ValueError(
                    f"The event channel '{elt}' should not be marked as bad in the "
                    "connected Stream."
                )
            if stream.get_channel_types(picks=elt)[0] != "stim":
                raise ValueError(f"The event channel '{elt}' should be of type 'stim'.")
        elif event_stream is not None:
            if elt not in event_stream._info.ch_names:
                raise ValueError(
                    "If 'event_stream' is provided, the event channel(s) must be "
                    f"part of 'event_stream'. '{elt}' was not found."
                )
            if elt in event_stream._info["bads"]:
                raise ValueError(
                    f"The event channel '{elt}' in the event stream should not be "
                    "marked as bad."
                )
            if (
                event_stream._info["sfreq"] != 0
                and event_stream.get_channel_types(picks=elt)[0] != "stim"
            ):
                raise ValueError(
                    f"The event channel '{elt}' in the event stream should be of type "
                    "'stim' if the event stream is regularly sampled."
                )


def _ensure_event_id(
    event_id: int | dict[str, int] | None, event_stream: BaseStream | None
) -> dict[str, int] | None:
    """Ensure event_ids is a dictionary or None."""
    check_type(event_id, (None, int, dict), "event_id")
    if event_id is None:
        if event_stream is None or event_stream.info["sfreq"] != 0:
            raise ValueError(
                "The 'event_id' must be provided if no irregularly sampled "
                "'event_stream' is provided."
            )
        return None
    if (
        event_id is not None
        and event_stream is not None
        and event_stream.info["sfreq"] == 0
    ):
        warn(
            "The argument 'event_id' should be set to None when events are selected "
            "from an irregularly sampled event stream."
        )
        return None
    raise_ = False
    if isinstance(event_id, int):
        if event_id <= 0:
            raise_ = True
        event_id = {str(event_id): event_id}
    else:
        for key, value in event_id.items():
            check_type(key, (str,), "event_id")
            check_type(value, ("int-like",), "event_id")
            if len(key) == 0:
                raise_ = True
            if isinstance(value, int) and value <= 0:
                raise_ = True
    if raise_:
        raise ValueError(
            "The 'event_id' must be a positive integer or a dictionary mapping "
            "non-empty strings to positive integers."
        )
    return event_id


def _check_baseline(
    baseline: tuple[float | None, float | None] | None,
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
    reject: dict[str, float] | None, flat: dict[str, float] | None, info: Info
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
    reject_tmin: float | None, reject_tmax: float | None, tmin: float, tmax: float
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


def _ensure_detrend_str(detrend: int | str | None) -> str | None:
    """Ensure detrend is an integer."""
    if detrend is None:
        return None
    if isinstance(detrend, str):
        check_value(detrend, ("constant", "linear"), "detrend")
        return detrend
    detrend = ensure_int(detrend, "detrend")
    if detrend not in (0, 1):
        raise ValueError(
            "The detrend argument must be 'constant', 'linear' or their integer "
            "equivalent 0 and 1."
        )
    mapping = {0: "constant", 1: "linear"}
    return mapping[detrend]


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
    min_samples = min_duration * sfreq
    events_list = []
    for d, ch_name in zip(data, event_channels, strict=True):
        events = find_events(
            d[np.newaxis, :],
            first_samp=0,
            verbose="CRITICAL",  # disable MNE's logging
            output=output,
            consecutive=consecutive,
            min_samples=min_samples,
            mask=mask,
            uint_cast=uint_cast,
            mask_type=mask_type,
            initial_event=initial_event,
            ch_name=ch_name,
        )
        # add safety check for spurious events (for ex. from neuromag syst.) by
        # checking the number of low sample events
        n_short_events = np.sum(np.diff(events[:, 0]) < shortest_event)
        if n_short_events > 0:
            warn(
                f"You have {n_short_events} events shorter than the shortest_event. "
                "These are very unusual and you may want to set min_duration to a "
                "larger value e.g. x / raw.info['sfreq']. Where x = 1 sample shorter "
                "than the shortest event length."
            )
        events_list.append(events)
    events = np.concatenate(events_list, axis=0)
    events = _find_unique_events(events)
    return events[np.argsort(events[:, 0])]


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
    # remove events outside of the event_id dictionary
    if event_id is not None:
        sel = np.isin(events[:, 2], list(event_id.values()))
        events = events[sel]
    # get the events position in the stream times after removing events outside of ts
    if ts_events is not None:
        sel = np.where(
            (ts[0] <= ts_events[events[:, 0]]) & (ts_events[events[:, 0]] <= ts[-1])
        )[0]
        events = events[sel]
        events[:, 0] = np.searchsorted(ts, ts_events[events[:, 0]], side="left")
    sel = np.where(0 <= events[:, 0] + tmin_shift)[0]
    # remove events which can't fit an entire epoch and/or are outside of the buffer
    sel = np.where(
        (0 <= events[:, 0] + tmin_shift)
        & (events[:, 0] + tmin_shift + buffer_size <= ts.size)
    )[0]
    events = events[sel]
    # remove events which have already been moved to the buffer
    if last_ts is not None:
        sel = np.where(ts[events[:, 0]] > last_ts)[0]
        events = events[sel]
    return events


def _process_data(
    data: ScalarArray,  # array of shape (n_epochs, n_samples, n_channels)
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
    # start by PTP rejection to limit the number of epochs to baseline and detrend
    if reject is not None or flat is not None:
        # figure out the slice of indices to use for rejection
        if reject_tmin is None:
            reject_imin = None
        else:
            idx = np.nonzero(reject_tmin <= times)[0]
            reject_imin = idx[0]
        if reject_tmax is None:
            reject_imax = None
        else:
            idx = np.nonzero(times <= reject_tmax)[0]
            reject_imax = idx[-1]
        reject_time = slice(reject_imin, reject_imax)
        data_ptp = data[:, reject_time, :]
        if data_ptp.shape[1] != 0:  # check that the slice is not empty
            ptp = np.max(data[:, reject_time, :], axis=1) - np.min(
                data[:, reject_time, :], axis=1
            )  # shape (n_epochs, n_channels)
            if reject is not None:
                for ch_type, threshold in reject.items():
                    idx = ch_idx_by_type[ch_type]
                    ptp_ch = ptp[:, idx]
                    # select the epochs to **keep**
                    sel1 = np.where(np.all(ptp_ch < threshold, axis=1))[0]
            else:
                sel1 = np.arange(data.shape[0])
            if flat is not None:
                for ch_type, threshold in flat.items():
                    idx = ch_idx_by_type[ch_type]
                    ptp_ch = ptp[:, idx]
                    # select the epochs to **keep**
                    sel2 = np.where(np.all(threshold < ptp_ch, axis=1))[0]
            else:
                sel2 = np.arange(data.shape[0])
            sel = np.intersect1d(sel1, sel2)  # select the epochs to **keep**
            data = data[sel, :, :]
        else:
            warn(
                "The rejection time window defined with 'reject_tmin' and "
                "'reject_tmax' yields an empty segment. Skipping rejection."
            )
    if data.shape[0] == 0:
        return data
    # next apply baseline correction
    if baseline is not None:
        if baseline[0] is None:
            baseline_imin = None
        else:
            idx = np.nonzero(baseline[0] <= times)[0]
            baseline_imin = idx[0]
        if baseline[1] is None:
            baseline_imax = None
        else:
            idx = np.nonzero(times <= baseline[1])[0]
            baseline_imax = idx[-1]
        baseline_time = slice(baseline_imin, baseline_imax)
        data_baseline = data[:, baseline_time, :]
        if data_baseline.shape[1] != 0:
            data -= np.mean(data[:, baseline_time, :], axis=1, keepdims=True)
        else:
            warn(
                "The baseline time window defined with 'baseline', 'tmin' and 'tmax' "
                "yields an empty segment. Skipping baseline correction."
            )
    # finally detrend the data
    if detrend_type is not None:
        data = detrend(data, axis=1, type=detrend_type, overwrite_data=True)
    return data


def _remove_empty_elements(
    data: ScalarArray, ts: NDArray[np.float64]
) -> tuple[ScalarArray, NDArray[np.float64]]:
    """Remove empty elements from the data and ts array."""
    n_samples = np.count_nonzero(ts)
    data = data[:, -n_samples:]
    ts = ts[-n_samples:]
    return data, ts
