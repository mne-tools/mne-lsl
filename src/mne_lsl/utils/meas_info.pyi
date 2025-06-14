from typing import Any

from _typeshed import Incomplete
from mne import Info

from ..lsl.stream_info import _BaseStreamInfo as _BaseStreamInfo
from ._checks import check_type as check_type
from ._checks import check_value as check_value
from ._checks import ensure_int as ensure_int
from .logs import logger as logger
from .logs import warn as warn

_CH_TYPES_DICT: Incomplete
_STIM_TYPES: tuple[str, ...]
_HUMAN_UNITS: dict[int, dict[str, int]]

def create_info(
    n_channels: int,
    sfreq: float,
    stype: str,
    desc: _BaseStreamInfo | dict[str, Any] | None,
) -> Info:
    """Create a minimal :class:`mne.Info` object from an LSL stream attributes.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    sfreq : float
        Sampling frequency in Hz. ``0`` corresponds to an irregular sampling rate.
    stype : str
        Type of the stream. This type will be used as a default for all channels with
        an unknown type. If the ``stype`` provided is not among the MNE-known channel
        types, defaults to ``'misc'``.
    desc : StreamInfo | dict | None
        If provided, dictionary or :class:`~mne_lsl.lsl.StreamInfo` containing the
        channel information. A `~mne_lsl.lsl.StreamInfo` contains the number of
        channels, sampling frequency and stream type, which will be checked against the
        provided arguments ``n_channels``, ``sfreq`` and ``stype``.

    Returns
    -------
    info : Info
        MNE :class:`~mne.Info` object corresponding.

    Notes
    -----
    If the argument ``desc`` is not aligned with ``n_channels``, it is ignored and an
    :class:`mne.Info` with the number of channels definbed in ``n_channels`` is created.
    """

def _create_default_info(n_channels: int, sfreq: float, stype: str) -> Info:
    """Create a default Info object.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    sfreq : float
        Sampling frequency in Hz. ``0`` corresponds to an irregular sampling rate.
    stype : str
        Type of the stream.

    Returns
    -------
    info : Info
        MNE :class:`~mne.Info` object corresponding.
    """

def _read_desc_sinfo(
    n_channels: int, stype: str, desc: _BaseStreamInfo
) -> tuple[list[str], list[str], list[int], str | None]:
    """Read channel information from a StreamInfo.

    If the StreamInfo is retrieved by resolve_streams, the description will be empty.
    An inlet should be created and the inlet StreamInfo should be used to retrieve the
    channel description.
    """

def _read_desc_dict(
    n_channels: int, stype: str, desc: dict[str, Any]
) -> tuple[list[str], list[str], list[int], str | None]:
    """Read channel information from a description dictionary.

    A dictionary is returned from loading an XDF file.
    """

def _get_ch_types_and_units(
    channels: list[dict[str, Any]], stype: str
) -> tuple[list[str], list[int]]:
    """Get the channel types and units from a stream description."""

def _safe_get(channel, item, default) -> str:
    """Retrieve element from a stream description safely."""

def _set_channel_units(info: Info, mapping: dict[str, str | int]) -> None:
    """Set the channel unit multiplication factor."""
