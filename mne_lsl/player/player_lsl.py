from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import catch_warnings, filterwarnings

import numpy as np
from mne import Annotations
from mne.annotations import _handle_meas_date

from ..lsl import StreamInfo, StreamOutlet, local_clock
from ..utils._checks import check_type
from ..utils._docs import copy_doc, fill_doc
from ..utils._time import high_precision_sleep
from ..utils.logs import logger, warn
from ._base import BasePlayer

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Callable, Optional, Union


@fill_doc
class PlayerLSL(BasePlayer):
    """Class for creating a mock LSL stream.

    Parameters
    ----------
    %(player_fname)s
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the :class:`~mne_lsl.lsl.StreamOutlet`.
    %(n_repeat)s
    name : str | None
        Name of the mock LSL stream. If ``None``, the name ``MNE-LSL-Player`` is used.
    source_id : str
        A unique identifier of the device or source of the data. This information
        improves the system robustness since it allows recipients to recover
        from failure by finding a stream with the same ``source_id`` on the network.
        By default, the source_id is set to ``"MNE-LSL"``.
    annotations : bool | None
        If ``True``, an :class:`~mne_lsl.lsl.StreamOutlet` is created for the
        :class:`~mne.Annotations` of the :class:`~mne.io.Raw` object. If ``False``,
        :class:`~mne.Annotations` are ignored and the :class:`~mne_lsl.lsl.StreamOutlet`
        is not created. If ``None`` (default), the :class:`~mne_lsl.lsl.StreamOutlet` is
        created only if the :class:`~mne.io.Raw` object has :class:`~mne.Annotations` to
        push. See notes for additional information on the :class:`~mne.Annotations`
        timestamps.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended. Once
    the end-of-file is reached, the player loops back to the beginning which can lead to
    a small discontinuity in the data stream.

    Each time a chunk (defined by ``chunk_size``) is pushed on the
    :class:`~mne_lsl.lsl.StreamOutlet`, the last sample of the chunk is attributed the
    current time (as returned by the function :func:`~mne_lsl.lsl.local_clock`). Thus,
    the sample ``chunk[0, :]`` occurred in the  past and the sample ``chunk[-1, :]``
    occurred "now". If :class:`~mne.Annotations` are streamed, the annotations within
    the chunk are pushed on the annotation :class:`~mne_lsl.lsl.StreamOutlet`. The
    :class:`~mne.Annotations` are pushed with a timestamp corrected for the annotation
    onset in regards to the chunk beginning. However, :class:`~mne.Annotations` push is
    *not* delayed until the the annotation timestamp or until the end of the chunk.
    Thus, an :class:`~mne.Annotations` can arrived at the client
    :class:`~mne_lsl.lsl.StreamInlet` "ahead" of time, i.e. earlier than the current
    time (as returned by the function :func:`~mne_lsl.lsl.local_clock`). Thus, it is
    recommended to connect to an annotation stream with the
    :class:`~mne_lsl.lsl.StreamInlet` or :class:`~mne_lsl.stream.StreamLSL` with the
    ``clocksync`` processing flag and to always inspect the timestamps returned for
    every samples.

    .. code-block:: python

        from mne_lsl.lsl import local_clock
        from mne_lsl.player import PlayerLSL as Player
        from mne_lsl.stream import StreamLSL as Stream

        player = Player(..., annotations=True)  # file with annotations
        player.start()
        stream = Stream(bufsize=100, stype="annotations")
        stream.connect(processing_flags=["clocksync"])
        data, ts = stream.get_data()
        print(ts - local_clock())  # positive values are annotations in the "future"

    If :class:`~mne.Annotations` are streamed, the :class:`~mne_lsl.lsl.StreamOutlet`
    name is ``{name}-annotations`` where ``name`` is the name of the
    :class:`~mne_lsl.player.PlayerLSL`. The ``dtype`` is set to ``np.float64`` and each
    unique :class:`~mne.Annotations` description is encoded as a channel. The value
    streamed on a channel correspond to the duration of the :class:`~mne.Annotations`.
    Thus, a sample on this :class:`~mne_lsl.lsl.StreamOutlet` is a one-hot encoded
    vector of the :class:`~mne.Annotations` description/duration.
    """

    def __init__(
        self,
        fname: Union[str, Path],
        chunk_size: int = 10,
        n_repeat: Union[int, float] = np.inf,
        *,
        name: Optional[str] = None,
        source_id: str = "MNE-LSL",
        annotations: Optional[bool] = None,
    ) -> None:
        super().__init__(fname, chunk_size, n_repeat)
        check_type(name, (str, None), "name")
        check_type(source_id, (str,), "source_id")
        check_type(annotations, (bool, None), "annotations")
        self._name = "MNE-LSL-Player" if name is None else name
        self._source_id = source_id
        # look for annotations
        if annotations is None:
            self._annotations = True if len(self._raw.annotations) != 0 else False
        else:
            if annotations and len(self._raw.annotations) == 0:
                warn(
                    f"{self._name}: The raw file has no annotations. The annotations "
                    "will be ignored."
                )
                self._annotations = False
            else:
                self._annotations = annotations
        # create stream info based on raw
        ch_types = self._raw.get_channel_types(unique=True)
        self._sinfo = StreamInfo(
            name=self._name,
            stype=ch_types[0] if len(ch_types) == 1 else "",
            n_channels=len(self._raw.info["ch_names"]),
            sfreq=self._raw.info["sfreq"],
            dtype=np.float64,
            source_id=self._source_id,
        )
        self._sinfo.set_channel_info(self._raw.info)
        logger.debug("%s: set channel info", self._name)
        if self._annotations:
            self._annotations_names = {
                name: idx
                for idx, name in enumerate(
                    sorted(set(self._raw.annotations.description))
                )
            }
            self._sinfo_annotations = StreamInfo(
                name=f"{self._name}-annotations",
                stype="annotations",
                n_channels=len(self._annotations_names),
                sfreq=0.0,
                dtype=np.float64,
                source_id=self._source_id,
            )
            self._sinfo_annotations.set_channel_names(list(self._annotations_names))
            self._sinfo_annotations.set_channel_types("annotations")
            self._sinfo_annotations.set_channel_units("none")
            self._annotations_idx = self._raw.time_as_index(self._raw.annotations.onset)
            self._annotations_idx -= self._raw.first_samp
        else:
            self._sinfo_annotations = None
            self._annotations_idx = None
        # create additional streaming variables
        self._reset_variables()

    @copy_doc(BasePlayer.rename_channels)
    def rename_channels(
        self,
        mapping: Union[dict[str, str], Callable],
        allow_duplicates: bool = False,
        *,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> PlayerLSL:
        super().rename_channels(mapping, allow_duplicates)
        self._sinfo.set_channel_names(self.info["ch_names"])
        return self

    def start(self) -> PlayerLSL:
        """Start streaming data on the LSL :class:`~mne_lsl.lsl.StreamOutlet`.

        Returns
        -------
        player : instance of :class:`~mne_lsl.player.PlayerLSL`
            The player instance modified in-place.
        """
        super().start()
        self._outlet = StreamOutlet(self._sinfo, self._chunk_size)
        self._outlet_annotations = (
            StreamOutlet(self._sinfo_annotations, 1) if self._annotations else None
        )
        self._streaming_delay = self.chunk_size / self.info["sfreq"]
        self._target_timestamp = local_clock()
        self._executor.submit(self._stream)
        logger.debug("%s: Started streaming thread.", self._name)
        return self

    @copy_doc(BasePlayer.set_channel_types)
    def set_channel_types(
        self,
        mapping: dict[str, str],
        *,
        on_unit_change: str = "warn",
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> PlayerLSL:
        super().set_channel_types(
            mapping, on_unit_change=on_unit_change, verbose=verbose
        )
        self._sinfo.set_channel_types(self.get_channel_types(unique=False))
        return self

    @copy_doc(BasePlayer.set_channel_units)
    def set_channel_units(self, mapping: dict[str, Union[str, int]]) -> PlayerLSL:
        super().set_channel_units(mapping)
        ch_units_after = np.array(
            [ch["unit_mul"] for ch in self.info["chs"]], dtype=np.int8
        )
        self._sinfo.set_channel_units(ch_units_after)
        return self

    def stop(self) -> PlayerLSL:
        """Stop streaming data on the LSL :class:`~mne_lsl.lsl.StreamOutlet`.

        Returns
        -------
        player : instance of :class:`~mne_lsl.player.PlayerLSL`
            The player instance modified in-place.
        """
        logger.debug("%s: Stopping.", self._name)
        super().stop()
        self._del_outlets()
        self._reset_variables()
        return self

    def _del_outlets(self) -> None:
        """Attempt to delete outlets."""
        if hasattr(self, "_outlet"):
            try:
                self._outlet._del()
            except Exception:  # pragma: no cover
                pass
        if hasattr(self, "_outlet_annotations"):
            try:
                self._outlet_annotations._del()
            except Exception:  # pragma: no cover
                pass

    @copy_doc(BasePlayer._stream)
    def _stream(self) -> None:
        try:
            # retrieve data and push to the stream outlet
            start = self._start_idx
            if start == 0 and self._n_repeated == 1:
                logger.debug("First _stream ping %s", self._name)
            stop = start + self._chunk_size
            if stop <= self._raw.times.size:
                data = self._raw[:, start:stop][0].T
                self._start_idx += self._chunk_size
            elif self._raw.times.size < stop and self._n_repeated < self._n_repeat:
                logger.debug("End of file reached, looping back to the beginning.")
                stop %= self._raw.times.size
                data = np.vstack([self._raw[:, start:][0].T, self._raw[:, :stop][0].T])
                self._start_idx = stop
                self._n_repeated += 1
            else:
                logger.debug("End of file reached, stopping the player.")
                stop = self._raw.times.size
                data = self._raw[:, start:stop][0].T
                self._end_streaming = True
                if data.size == 0:  # pragma: no cover
                    # rare condition where if chunk_size is equal to 1, the last chunk
                    # will be empty and we should abort at this point.
                    logger.debug("End of file reached with an empty chunk.")
                    if self._chunk_size != 1:  # pragma: no cover
                        warn(
                            f"{self._name}: End of file reached with an empty chunk. "
                            "This should not happen with a chunk_size different from 1."
                        )
                    self._del_outlets()
                    self._reset_variables()
                    return None
            # bump the target LSL timestamp before pushing because the argument
            # 'timestamp' expects the timestamp of the most 'recent' sample, which in
            # this non-real time replay scenario is the timestamp of the last sample in
            # the chunk.
            start_timestamp = self._target_timestamp
            self._target_timestamp += self._streaming_delay
            logger.debug(
                "%s: Pushing chunk %s:%s, timestamp=%s",
                self._name,
                start,
                stop,
                self._target_timestamp,
            )
            if self._chunk_size == 1:  # pragma: no cover
                self._outlet.push_sample(data[0, :], timestamp=self._target_timestamp)
            else:
                self._outlet.push_chunk(data, timestamp=self._target_timestamp)
            self._stream_annotations(start, stop, start_timestamp)
        except Exception as exc:  # pragma: no cover
            logger.error("%s: Stopping due to exception: %s", self._name, exc)
            self._del_outlets()
            self._reset_variables()
        else:
            if self._end_streaming:
                self._del_outlets()
                self._reset_variables()
                return None  # don't schedule another task if we are ending
            # figure out how early or late the thread woke up and compensate the
            # delay for the next thread to remain in the neighbourhood of
            # _target_timestamp for the following wake.
            delta = self._target_timestamp - self._streaming_delay - local_clock()
            delay = max(self._streaming_delay + delta, 0)
            high_precision_sleep(delay)
            try:
                self._executor.submit(self._stream)
            except RuntimeError:  # pragma: no cover
                pass  # shutdown

    def _stream_annotations(
        self, start: int, stop: int, start_timestamp: float
    ) -> None:
        """Push annotations in a chunk."""
        if not self._annotations:
            return None
        # get the annotations in the chunk
        if start < stop:
            mask = (self._annotations_idx >= start) & (self._annotations_idx < stop)
            idx = np.where(mask)[0]
        else:  # start > stop, equality is impossible or chunk_size would be equal to 0.
            mask1 = self._annotations_idx >= start
            mask2 = self._annotations_idx < stop
            idx = np.hstack([np.where(mask1)[0], np.where(mask2)[0]])
        if idx.size == 0:
            return None
        # estimate LSL timestamp of each annotation
        timestamps = (
            start_timestamp
            + (
                self.annotations.onset[idx]
                - self._raw.first_samp / self._raw.info["sfreq"]
            )
            - self._raw.times[start]
        )
        # one-hot encode the description and duration in the channels
        idx_ = np.array(
            [
                self._annotations_names[desc]
                for desc in self.annotations.description[idx]
            ]
        )
        data = np.zeros((timestamps.size, len(self._annotations_names)))
        data[np.arange(timestamps.size), idx_] = self.annotations.duration[idx]
        # push as a chunk all annotations in the [start:stop] range
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="A single sample is pushed. Consider using push_sample().",
                category=RuntimeWarning,
            )
            self._outlet_annotations.push_chunk(data, timestamps)

    def _reset_variables(self) -> None:
        """Reset variables for streaming."""
        logger.debug("Resetting variables %s", self._name)
        super()._reset_variables()
        self._outlet = None
        self._outlet_annotations = None
        self._target_timestamp = None

    # ----------------------------------------------------------------------------------
    def __del__(self):
        """Delete the player and destroy the :class:`~mne_lsl.lsl.StreamOutlet`."""
        try:
            self.stop()
        except Exception:  # pragma: no cover
            pass

    def __repr__(self):
        """Representation of the instance."""
        status = "OFF" if getattr(self, "_outlet", None) is None else "ON"
        if self._fname is not None:
            repr_ = f"<Player: {self.name} | {status} | {self._fname}>"
        else:
            repr_ = f"<Player: {self.name} | {status}>"
        return repr_

    # ----------------------------------------------------------------------------------
    @property
    def annotations(self) -> Annotations:
        """Annotations attached to the raw object, if streamed.

        :type: :class:`~mne.Annotations`
        """
        return (
            self._raw.annotations
            if self._annotations
            else Annotations([], [], [], _handle_meas_date(self.info["meas_date"]))
        )

    @property
    def name(self) -> str:
        """Name of the LSL stream.

        :type: :class:`str`
        """
        return self._name

    @property
    def source_id(self) -> str:
        """Source ID of the LSL stream.

        :type: :class:`str`
        """
        return self._source_id
