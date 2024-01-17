from __future__ import annotations  # c.f. PEP 563, PEP 649

from threading import Timer
from typing import TYPE_CHECKING
from warnings import catch_warnings, filterwarnings, warn

import numpy as np
from mne import Annotations
from mne.annotations import _handle_meas_date

from ..lsl import StreamInfo, StreamOutlet, local_clock
from ..utils._checks import check_type
from ..utils._docs import copy_doc
from ..utils.logs import logger
from ._base import BasePlayer

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Callable, Optional, Union


class PlayerLSL(BasePlayer):
    """Class for creating a mock LSL stream.

    Parameters
    ----------
    fname : path-like
        Path to the file to re-play as a mock LSL stream. MNE-Python must be able to
        load the file with :func:`mne.io.read_raw`.
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the :class:`~mne_lsl.lsl.StreamOutlet`.
        If these chunks are too small then the thread-based timing might not work
        properly.
    name : str | None
        Name of the mock LSL stream. If ``None``, the name ``MNE-LSL-Player`` is used.
    annotations : bool | None
        If ``True``, an :class:`~mne_lsl.lsl.StreamOutlet` is created for the
        :class:`~mne.Annotations` of the :class:`~mne.io.Raw` object. If ``False``,
        :class:`~mne.Annotations` are ignored and the :clas:`~mne_lsl.lsl.StreamOutlet`
        is not created. If ``None`` (default), the :class:`~mne_lsl.lsl.StreamOutlet` is
        created only if the :class:`~mne.io.Raw` object has :class:`~mne.Annotations` to
        push.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended. Once
    the end-of-file is reached, the player loops back to the beginning which can lead to
    a small discontinuity in the data stream.
    """

    def __init__(
        self,
        fname: Union[str, Path],
        chunk_size: int = 64,
        name: Optional[str] = None,
        annotations: Optional[bool] = None,
    ) -> None:
        super().__init__(fname, chunk_size)
        check_type(name, (str, None), "name")
        check_type(annotations, (bool, None), "annotations")
        self._name = "MNE-LSL-Player" if name is None else name
        # look for annotations
        if annotations is None:
            self._annotations = True if len(self._raw.annotations) != 0 else False
        else:
            if annotations and len(self._raw.annotations) == 0:
                warn(
                    f"{self._name}: The raw file has no annotations. The annotations "
                    "will be ignored.",
                    RuntimeWarning,
                    stacklevel=2,
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
            source_id="MNE-LSL",
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
                source_id="MNE-LSL",
            )
            self._sinfo_annotations.set_channel_names(list(self._annotations_names))
            self._sinfo_annotations.set_channel_types("annotations")
            self._sinfo_annotations.set_channel_units("none")
            self._annotations_idx = self._raw.time_as_index(self._raw.annotations.onset)
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
        if self._streaming_thread is not None:
            warn(
                f"{self._name}: The player is already started. "
                "Use Player.stop() to stop streaming.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self
        self._outlet = StreamOutlet(self._sinfo, self._chunk_size)
        self._outlet_annotations = (
            StreamOutlet(self._sinfo_annotations, 1) if self._annotations else None
        )
        self._streaming_delay = self.chunk_size / self.info["sfreq"]
        self._streaming_thread = Timer(0, self._stream)
        self._streaming_thread.daemon = True
        self._target_timestamp = local_clock()
        self._streaming_thread.start()
        logger.debug("%s: Started streaming thread", self._name)
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
        logger.debug("%s: Stopping", self._name)
        super().stop()
        self._outlet = None
        self._outlet_annotations = None
        self._reset_variables()
        return self

    @copy_doc(BasePlayer._stream)
    def _stream(self) -> None:
        try:
            # retrieve data and push to the stream outlet
            start = self._start_idx
            if start == 0:
                logger.debug("First _stream ping %s", self._name)
            stop = start + self._chunk_size
            if stop <= self._raw.times.size:
                data = self._raw[:, start:stop][0].T
                self._start_idx += self._chunk_size
            else:
                stop = self._chunk_size - (self._raw.times.size - start)
                data = np.vstack([self._raw[:, start:][0].T, self._raw[:, :stop][0].T])
                self._start_idx = stop
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
            self._outlet.push_chunk(data, timestamp=self._target_timestamp)
            self._stream_annotations(start, stop, start_timestamp)
        except Exception as exc:
            logger.debug("%s: Stopping due to exception: %s", self._name, exc)
            self._reset_variables()
            return None  # equivalent to an interrupt
        else:
            if self._interrupt:
                return None  # don't recreate the thread if we are interrupting
            else:
                # figure out how early or late the thread woke up and compensate the
                # delay for the next thread to remain in the neighbourhood of
                # _target_timestamp for the following wake.
                delta = self._target_timestamp - self._streaming_delay - local_clock()
                delay = max(self._streaming_delay + delta, 0)
                # recreate the timer thread as it is one-call only
                self._streaming_thread = Timer(delay, self._stream)
                self._streaming_thread.daemon = True
                self._streaming_thread.start()

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
            start_timestamp + self.annotations.onset[idx] - self._raw.times[start]
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
        super().__del__()
        self._outlet = None
        self._outlet_annotations = None

    def __repr__(self):
        """Representation of the instance."""
        if getattr(self, "_outlet", None) is None:
            status = "OFF"
        else:
            status = "ON"
        return f"<Player: {self.name} | {status} | {self._fname}>"

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
