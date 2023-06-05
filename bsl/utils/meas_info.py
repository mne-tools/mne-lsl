# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from typing import TYPE_CHECKING

from mne import create_info as mne_create_info
from mne.io.constants import _ch_unit_mul_named
from mne.io.pick import get_channel_type_constants

from ._checks import check_type, ensure_int
from .logs import logger
from ..lsl.stream_info import _BaseStreamInfo

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple

    from mne import Info


_CH_TYPES_DICT = get_channel_type_constants(include_defaults=True)
_STIM_TYPES = (
    "marker",
    "markers",
    "stim",
)
_EEG_UNITS = {
    "v": _ch_unit_mul_named[0],
    "volt": _ch_unit_mul_named[0],
    "volts": _ch_unit_mul_named[0],
    "mv": _ch_unit_mul_named[-3],
    "millivolt": _ch_unit_mul_named[-3],
    "millivolts": _ch_unit_mul_named[-3],
    "uv": _ch_unit_mul_named[-6],
    "microvolt": _ch_unit_mul_named[-6],
    "microvolts": _ch_unit_mul_named[-6],
}


def create_info(
    n_channels: int,
    sfreq: float,
    stype: str,
    desc: Optional[_BaseStreamInfo, Dict[str, Any]],
) -> Info:
    """Create an `mne.Info` object from a stream attributes.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    sfreq : float
        Sampling frequency in Hz. ``0`` corresponds to an irregular sampling rate.
    desc : StreamInfo | dict | None
        If provided, dictionary or `~bsl.lsl.StreamInfo` containing the channel
        information.

    Returns
    -------
    info : Info
        MNE info object corresponding.
    """
    n_channels = ensure_int(n_channels, "n_channels")
    check_type(sfreq, ("numeric",), "sfreq")
    check_type(stype, (str,), "stype")
    if sfreq < 0:
        raise ValueError(
            "The sampling frequency provided must be a positive number. "
            f"Provided '{sfreq}' can not be interpreted as a sampling "
            "frequency in Hz."
        )
    check_type(desc, (_BaseStreamInfo, dict, None), "desc")

    # try to identify the main channel type
    stype = stype.lower().strip()
    stype = "stim" if stype in _STIM_TYPES else stype
    stype = stype if stype in _CH_TYPES_DICT else "misc"

    # attempt to create the info depending on the provided description
    try:
        if isinstance(desc, dict):
            ch_names, ch_types, ch_units, manufacturer = _read_ch_dict(
                n_channels, stype, desc
            )
        elif isinstance(desc, _BaseStreamInfo):
            ch_names, ch_types, ch_units, manufacturer = _read_ch_sinfo(
                n_channels, stype, desc
            )

        info = mne_create_info(ch_names, 1, ch_types)
        with info._unlock():
            info["sfreq"] = sfreq
            for ch, ch_unit in zip(info["chs"], ch_units):
                ch["unit_mul"] = ch_unit
        # add manufacturer information if available
        info["device_info"] = dict()
        if isinstance(manufacturer, str):
            info["device_info"]["model"] = manufacturer
        elif (
            isinstance(manufacturer, list)
            and len(manufacturer) == 1
            and isinstance(manufacturer[0], str)
        ):
            info["device_info"]["model"] = manufacturer[0]
    except Exception:
        logger.warning(
            "Something went wrong while reading the channel description. Defaulting to "
            "channel IDs and MNE-compatible stream type."
        )
        info = mne_create_info(n_channels, 1.0, stype)
        info["device_info"] = dict()
        with info._unlock():
            info["sfreq"] = sfreq
        info["device_info"] = dict()

    info._check_consistency()
    return info


# --------------------- Functions to read from a description sinfo ---------------------
def _read_ch_sinfo(
    n_channels: int, stype: str, desc: str
) -> Tuple[List[str], List[str], List[int], Optional[str]]:
    """Read channel information from a StreamInfo.

    If the StreamInfo is retrieved by resolve_streams, the description will be empty.
    An inlet should be created and the inlet StreamInfo should be used to retrive the
    channel description.
    """
    if n_channels != desc.n_channels:
        raise RuntimeError(
            "The number of channels expected and the number of channels in the "
            "StreamInfo differ."
        )
    if stype != desc.stype:
        raise RuntimeError(
            "The stream type expected and the stream type in the StreamInfo differ."
        )

    ch_names = [ch_name for ch_name in desc.get_channel_names() if ch_name is not None]
    assert len(ch_names) == n_channels
    assert all(isinstance(elt, str) and len(elt) != 0 for elt in ch_names)

    try:
        ch_types = list()
        for ch_type in desc.get_channel_types():
            ch_type = ch_type.lower().strip()
            ch_type = stype if ch_type is None else ch_type
            ch_type = "stim" if ch_type in _STIM_TYPES else ch_type
            ch_type = ch_type if ch_type in _CH_TYPES_DICT else stype
            ch_types.append(ch_type)
    except Exception:
        ch_types = [stype] * n_channels
    assert len(ch_types) == n_channels
    assert all(isinstance(elt, str) and len(elt) != 0 for elt in ch_types)

    try:
        ch_units = list()
        for ch_type, ch_unit in zip(ch_types, desc.get_channel_units()):
            ch_unit = ch_unit.lower().strip()
            ch_unit = _ch_unit_mul_named[0] if ch_unit is None else ch_unit
            if ch_type == "eeg" and ch_unit in _EEG_UNITS:
                ch_unit = _EEG_UNITS[ch_unit]
            if isinstance(ch_unit, (str, int)):  # we failed to identify the unit
                ch_unit = _ch_unit_mul_named[0]
            ch_units.append(ch_unit)
    except Exception:
        ch_units = [_ch_unit_mul_named[0]] * n_channels
    assert len(ch_units) == n_channels

    # TODO: retrieve manufacturer from StreamInfo.desc
    manufacturer = None
    return ch_names, ch_types, ch_units, manufacturer


# --------------------- Functions to read from a description dict ----------------------
def _read_ch_dict(
    n_channels: int, stype: str, desc: Dict[str, Any]
) -> Tuple[List[str], List[str], List[int], Optional[str]]:
    """Read channel information from a description dictionary.

    A dictionary is returned from loading an XDF file.
    """
    channels = desc["channels"][0]["channel"]
    assert len(channels) == n_channels
    ch_names = [ch["label"] for ch in channels]
    ch_names = [
        ch[0] if (isinstance(ch, list) and len(ch) == 1) else ch for ch in ch_names
    ]
    assert all(isinstance(elt, str) and len(elt) != 0 for elt in ch_names)
    ch_types, ch_units = _get_ch_types_and_units(channels, stype)
    manufacturer = desc.get("manufacturer", None)
    return ch_names, ch_types, ch_units, manufacturer


def _get_ch_types_and_units(
    channels: List[Dict[str, Any]],
    stype: str,
) -> Tuple[List[str], List[int]]:
    """Get the channel types and units from a stream description."""
    ch_types = list()
    ch_units = list()
    for ch in channels:
        ch_type = _safe_get(ch, "type", stype)
        ch_type = "stim" if ch_type in _STIM_TYPES else ch_type
        ch_type = ch_type if ch_type in _CH_TYPES_DICT else stype

        ch_unit = _safe_get(ch, "unit", _ch_unit_mul_named[0])
        if ch_type == "eeg" and ch_unit in _EEG_UNITS:
            ch_unit = _EEG_UNITS[ch_unit]
        if isinstance(ch_unit, (str, int)):  # we failed to identify the unit
            ch_unit = _ch_unit_mul_named[0]

        ch_types.append(ch_type)
        ch_units.append(ch_unit)
    return ch_types, ch_units


def _safe_get(channel, item, default) -> str:
    """Retrieve element from a stream description safely."""
    # does it exist?
    if item not in channel:
        return default
    elt = channel[item]
    # is it nested?
    if isinstance(elt, list):
        if len(elt) != 1:
            return default
        elt = elt[0]
    # is it a string?
    if not isinstance(elt, str):
        return default
    # is it an empty string?
    if len(elt) == 0:
        return default
    elt = elt.lower().strip()  # ensure format
    return elt
