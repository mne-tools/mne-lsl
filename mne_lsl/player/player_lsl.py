from __future__ import annotations  # c.f. PEP 563, PEP 649

from threading import Timer
from typing import TYPE_CHECKING

import numpy as np

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
    name : str | None
        Name of the mock LSL stream. If ``None``, the name ``MNE-LSL-Player`` is used.
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the :class:`~mne_lsl.lsl.StreamOutlet`.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended. Once
    the end-of-file is reached, the player loops back to the beginning which can lead to
    a small discontinuity in the data stream.
    """

    def __init__(
        self, fname: Union[str, Path], name: Optional[str] = None, chunk_size: int = 16
    ) -> None:
        super().__init__(fname, chunk_size)
        check_type(name, (str, None), "name")
        self._name = "MNE-LSL-Player" if name is None else name
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
        # create additional streaming variables
        self._reset_variables()

    @copy_doc(BasePlayer.rename_channels)
    def rename_channels(
        self,
        mapping: Union[dict[str, str], Callable],
        allow_duplicates: bool = False,
        *,
        verbose=None,
    ) -> None:
        super().rename_channels(mapping, allow_duplicates)
        self._sinfo.set_channel_names(self.info["ch_names"])

    def start(self) -> None:
        """Start streaming data on the LSL `~mne_lsl.lsl.StreamOutlet`."""
        if self._streaming_thread is not None:
            logger.warning(
                "The player is already started. Use Player.stop() to stop streaming."
            )
            return None
        self._outlet = StreamOutlet(self._sinfo, self._chunk_size)
        self._streaming_delay = self.chunk_size / self.info["sfreq"]
        self._streaming_thread = Timer(0, self._stream)
        self._streaming_thread.daemon = True
        self._target_timestamp = local_clock()
        self._streaming_thread.start()

    @copy_doc(BasePlayer.set_channel_types)
    def set_channel_types(
        self, mapping: dict[str, str], *, on_unit_change: str = "warn", verbose=None
    ) -> None:
        super().set_channel_types(
            mapping, on_unit_change=on_unit_change, verbose=verbose
        )
        self._sinfo.set_channel_types(self.get_channel_types(unique=False))

    @copy_doc(BasePlayer.set_channel_units)
    def set_channel_units(self, mapping: dict[str, Union[str, int]]) -> None:
        super().set_channel_units(mapping)
        ch_units_after = np.array(
            [ch["unit_mul"] for ch in self.info["chs"]], dtype=np.int8
        )
        self._sinfo.set_channel_units(ch_units_after)

    def stop(self) -> None:
        """Stop streaming data on the LSL :class:`~mne_lsl.lsl.StreamOutlet`."""
        super().stop()
        del self._outlet
        self._reset_variables()

    @copy_doc(BasePlayer._stream)
    def _stream(self) -> None:
        try:
            # retrieve data and push to the stream outlet
            start = self._start_idx
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
            self._target_timestamp += self._streaming_delay
            self._outlet.push_chunk(data, timestamp=self._target_timestamp)
        except Exception:
            self._reset_variables()
            return None  # equivalent to an interrupt
        else:
            # figure out how early or late the thread woke up and compensate the delay
            # for the next thread to remain in the neighbourhood of _target_timestamp
            # for the following wake.
            delta = self._target_timestamp - self._streaming_delay - local_clock()
            delay = self._streaming_delay + delta
            if self._interrupt:
                # don't recreate the thread if we are trying to interrupt streaming
                return None
            else:
                # recreate the timer thread as it is one-call only
                self._streaming_thread = Timer(delay, self._stream)
                self._streaming_thread.daemon = True
                self._streaming_thread.start()

    def _reset_variables(self) -> None:
        """Reset variables for streaming."""
        super()._reset_variables()
        self._outlet = None
        self._target_timestamp = None

    # ----------------------------------------------------------------------------------
    def __del__(self):
        """Delete the player and destroy the :class:`~mne_lsl.lsl.StreamOutlet`."""
        super().__del__()
        try:
            del self._outlet
        except Exception:
            pass

    def __repr__(self):
        """Representation of the instance."""
        if self._outlet is None:
            status = "OFF"
        else:
            status = "ON"
        return f"<Player: {self.name} | {status} | {self._fname}>"

    # ----------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Name of the LSL stream.

        :type: :class:`str`
        """
        return self._name
