from __future__ import annotations  # c.f. PEP 563, PEP 649

from threading import Timer
from typing import TYPE_CHECKING

import numpy as np
from mne.io import read_raw
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.meas_info import ContainsMixin
else:
    from mne.io.meas_info import ContainsMixin

from .lsl import StreamInfo, StreamOutlet, local_clock
from .utils._checks import check_type, ensure_int, ensure_path
from .utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path
    from typing import List, Optional, Union

    from mne import Info


class Player(ContainsMixin):
    """Class for creating a mock LSL stream.

    Parameters
    ----------
    fname : path-like
        Path to the file to re-play as a mock LSL stream. MNE-Python must be able to
        load the file with :func:`mne.io.read_raw`.
    name : str | None
        Name of the mock LSL stream. If ``None``, the name ``BSL-Player`` is used.
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the :class:`~bsl.lsl.StreamOutlet`.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended. Once
    the end-of-file is reached, the player loops back to the beginning which can lead to
    a small discontinuity in the data stream.
    """

    def __init__(
        self, fname: Union[str, Path], name: Optional[str] = None, chunk_size: int = 16
    ) -> None:
        self._fname = ensure_path(fname, must_exist=True)
        check_type(name, (str, None), "name")
        self._name = "BSL-Player" if name is None else name
        self._chunk_size = ensure_int(chunk_size, "chunk_size")
        if self._chunk_size <= 0:
            raise ValueError(
                "The argument 'chunk_size' must be a strictly positive integer. "
                f"{chunk_size} is invalid."
            )

        # load header from the file and create StreamInfo
        self._raw = read_raw(self._fname, preload=True)
        ch_types = self._raw.get_channel_types(unique=True)
        self._sinfo = StreamInfo(
            name=self._name,
            stype=ch_types[0] if len(ch_types) == 1 else "",
            n_channels=len(self._raw.info["ch_names"]),
            sfreq=self._raw.info["sfreq"],
            dtype=np.float64,
            source_id="BSL",
        )
        self._sinfo.set_channel_names(self._raw.info["ch_names"])
        self._sinfo.set_channel_types(self._raw.get_channel_types(unique=False))
        self._sinfo.set_channel_units([ch["unit_mul"] for ch in self._raw.info["chs"]])
        self._outlet = None
        self._start_idx = 0
        self._streaming_delay = None
        self._streaming_thread = None
        self._target_timestamp = None

    def start(self) -> None:
        """Start streaming data on the LSL `~bsl.lsl.StreamOutlet`."""
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

    def stop(self) -> None:
        """Stop streaming data on the LSL :class:`~bsl.lsl.StreamOutlet`."""
        if self._streaming_thread is None:
            raise RuntimeError(
                "The player is not started. Use Player.start() to begin streaming."
            )
        while self._streaming_thread.is_alive():
            self._streaming_thread.cancel()
        del self._outlet
        self._reset_variables()

    def _stream(self) -> None:
        """Push a chunk of data from the raw object to the StreamOutlet.

        Don't use raw.get_data but indexing which is faster.

        >>> [In] %timeit raw[:, 0:16][0]
        >>> 19 µs ± 50.3 ns per loo
        >>> [In] %timeit raw.get_data(start=0, stop=16)
        >>> 1.3 ms ± 1.01 µs per loop
        >>> [In] %timeit np.ascontiguousarray(raw[:, 0:16][0].T)
        >>> 23.7 µs ± 183 ns per loop
        """
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

            # recreate the timer thread as it is one-call only
            self._streaming_thread = Timer(delay, self._stream)
            self._streaming_thread.daemon = True
            self._streaming_thread.start()

    def _reset_variables(self) -> None:
        """Reset variables for streaming."""
        self._outlet = None
        self._start_idx = 0
        self._streaming_delay = None
        self._streaming_thread = None
        self._target_timestamp = None

    # ----------------------------------------------------------------------------------
    def __del__(self):
        """Delete the player and destroy the :class:`~bsl.lsl.StreamOutlet`."""
        if hasattr(self, "_streaming_thread") and self._streaming_thread is not None:
            while self._streaming_thread.is_alive():
                self._streaming_thread.cancel()
        try:
            del self._outlet
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry point."""
        self.start()

    def __exit__(self, exc_type, exc_value, exc_tracebac):
        """Context manager exit point."""
        self.stop()

    def __repr__(self):
        """Representation of the instance."""
        if self._outlet is None:
            status = "OFF"
        else:
            status = "ON"
        return f"<Player: {self.name} | {status} | {self._fname}>"

    # ----------------------------------------------------------------------------------
    @property
    def ch_names(self) -> List[str]:
        """Name of the channels.

        :type: :class:`list` of :class:`str`
        """
        return self.info.ch_names

    @property
    def chunk_size(self) -> int:
        """Number of samples in a chunk.

        :type: :class:`int`
        """
        return self._chunk_size

    @property
    def fname(self) -> Path:
        """Path to file played.

        :type: :class:`~pathlib.Path`
        """
        return self._fname

    @property
    def info(self) -> Info:
        """Info of the LSL stream.

        :type: :class:`~mne.Info`
        """
        return self._raw.info

    @property
    def name(self) -> str:
        """Name of the LSL stream.

        :type: :class:`str`
        """
        return self._name
