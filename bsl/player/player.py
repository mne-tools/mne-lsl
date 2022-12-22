from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

from mne import create_info

from ..lsl import StreamInfo, StreamOutlet
from ..utils._checks import _check_type, _ensure_int, _ensure_path
from ..utils._docs import copy_doc


class MarkersPlayer:
    def __init__(
        self,
        stream: Dict[str, Any],
    ):
        self._stream_info = stream["info"]
        self._stream_data = stream["time_series"]
        self._stream_timestamps = stream["time_stamps"]

        # sanity-check
        assert self._stream_data.ndim == 1
        assert self._stream_timestamps.ndim == 1
        assert self._stream_data.size == self._stream_timestamps.size

        # load description
        if self._stream_info["desc"][0] == None:
            self._info = create_info(self.n_channels, 1, "stim")
            with self._info.unlock():
                self._info["sfreq"] = 0.
        else:
            desc = self._stream_info["desc"][0]

            self._info["device_info"] = dict()
            self._info["device_info"]["model"] = desc["manufacturer"][0]

    @staticmethod
    def _create_info(desc, n_channels):
        if desc is None:
            info = create_info(n_channels, 1, "stim")
            with info.unlock():
                info["sfreq"] = 0.
            return info

        channels = desc["channels"][0]["channel"]
        assert len(channels) == n_channels







    def _create_sinfo(self):
        self._sinfo = StreamInfo(
            self.name,
            self.stype,
            self.n_channels,
            self.sfreq,
            self.dtype,
            self.source_id,
        )

    @copy_doc(StreamInfo.dtype)
    @property
    def dtype(self) -> str:
        return self._stream_info["channel_format"][0]

    @property
    def effective_sfreq(self) -> float:
        """The estimated real sampling rate in Hz."""
        return self._stream_info["effective_srate"]

    @copy_doc(StreamInfo.name)
    @property
    def name(self) -> str:
        return self._stream_info["name"][0]

    @copy_doc(StreamInfo.n_channels)
    @property
    def n_channels(self) -> int:
        return int(self._stream_info["channel_count"][0])

    @copy_doc(StreamInfo.sfreq)
    @property
    def sfreq(self) -> float:
        return float(self._stream_info["nominal_srate"][0])

    @copy_doc(StreamInfo.source_id)
    @property
    def source_id(self) -> str:
        return self._stream_info["source_id"][0]

    @copy_doc(StreamInfo.stype)
    @property
    def stype(self) -> str:
        return self._stream_info["type"][0]


class DataPlayer:
    pass
