from __future__ import annotations  # c.f. PEP 563, PEP 649

from threading import Timer
from typing import TYPE_CHECKING

from mne.io import read_raw
from mne.utils import check_version

if check_version("mne", "1.5"):
    from mne.io.meas_info import ContainsMixin
elif check_version("mne", "1.6"):
    from mne._fiff.meas_info import ContainsMixin
else:
    from mne.io.meas_info import ContainsMixin

from .lsl import StreamInfo
from .utils._checks import check_type, ensure_int, ensure_path

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
        load the file with `mne.io.read_raw`.
    name : str | None
        Name of the mock LSL stream. If None, the name ``BSL-Player`` is used.
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the `~bsl.lsl.StreamOutlet`.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended.
    """

    def __init__(
        self, fname: Union[str, Path], name: Optional[str] = None, chunk_size: int = 1
    ) -> None:
        self._fname = ensure_path(fname, "fname")
        check_type(name, (str, None), "name")
        self._name = "BSL-Player" if name is None else name
        self._chunk_size = ensure_int(chunk_size, "chunk_size")

        # load header from the file and create StreamInfo
        self._raw = read_raw(self._fname, preload=True)
        ch_types = self._raw.get_channel_types(unique=True)
        self._sinfo = StreamInfo(
            name=self._name,
            stype=ch_types[0] if len(ch_types) == 1 else "",
            n_channels=len(self._raw.info["ch_names"]),
            sfreq=self._raw.info["sfreq"],
            dtype="float64",
            source_id="BSL",
        )
        self._sinfo.set_channel_names(self._raw.info["ch_names"])
        self._sinfo.set_channel_types(self._raw.get_channel_types(unique=False))
        # TODO: set the channel units
        self._streaming_thread = None

    def start(self):
        self._streaming_delay = self.chunk_size / self.info["sfreq"]
        self._streaming_thread = Timer(self._streaming_delay, self._stream)
        self.start()

    def stop(self):
        pass

    def _stream(self):
        # recreate the timer thread as it is one-call only
        self._streaming_thread = Timer(self._streaming_delay, self._stream)
        self._streaming_thread.start()
        # TODO: compensate self._streaming_delay for _stream execution time

    # ----------------------------------------------------------------------------------
    def __enter__(self):
        """Context manager entry point."""
        self.start()

    def __exit__(self, exc_type, exc_value, exc_tracebac):
        """Context manager exit point."""
        self.stop()

    def __repr__(self):
        """Representation of the instance."""
        status = "ON" if self._state.value == 1 else "OFF"
        return f"<Player: {self.name} | {status} | {self._fname}>"

    # ----------------------------------------------------------------------------------
    @property
    def ch_names(self) -> List[str]:
        """Name of the channels.

        :type: `list` of `str`
        """
        return self.info.ch_names

    @property
    def chunk_size(self) -> int:
        """Number of samples in a chunk.

        :type: `ìnt`
        """
        return self._chunk_size

    @property
    def fname(self) -> Path:
        """Path to file played.

        :type: `~pathlib.Path`
        """
        return self._fname

    @property
    def info(self) -> Info:
        """Info of the LSL stream.

        :type: `~mne.Info`
        """
        return self._raw.info

    @property
    def name(self) -> str:
        """Name of the LSL stream.

        :type: `str`
        """
        return self._name
