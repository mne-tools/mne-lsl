from __future__ import annotations  # c.f. PEP 563, PEP 649

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from mne import create_info as mne_create_info
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF, _ch_unit_mul_named
    from mne._fiff.pick import get_channel_type_constants
else:
    from mne.io.constants import FIFF, _ch_unit_mul_named
    from mne.io.pick import get_channel_type_constants

from ._checks import check_type, check_value, ensure_int
from .logs import logger

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional, Tuple, Union

    from mne import Info

    from ..lsl.stream_info import _BaseStreamInfo


_CH_TYPES_DICT = get_channel_type_constants(include_defaults=True)
_STIM_TYPES = (
    "marker",
    "markers",
    "stim",
)
_HUMAN_UNITS = {
    FIFF.FIFF_UNIT_V: {
        "v": _ch_unit_mul_named[0],
        "volt": _ch_unit_mul_named[0],
        "volts": _ch_unit_mul_named[0],
        "mv": _ch_unit_mul_named[-3],
        "millivolt": _ch_unit_mul_named[-3],
        "millivolts": _ch_unit_mul_named[-3],
        "uv": _ch_unit_mul_named[-6],
        "microvolt": _ch_unit_mul_named[-6],
        "microvolts": _ch_unit_mul_named[-6],
    },
}


def create_info(
    n_channels: int,
    sfreq: float,
    stype: str,
    desc: Optional[_BaseStreamInfo, Dict[str, Any]],
) -> Info:
    """Create a minimal `mne.Info` object from an LSL stream attributes.

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
        If provided, dictionary or `~mne_lsl.lsl.StreamInfo` containing the channel
        information. A `~mne_lsl.lsl.StreamInfo` contains the number of channels,
        sampling frequency and stream type, which will be checked against the provided
        arguments ``n_channels``, ``sfreq`` and ``stype``.

    Returns
    -------
    info : Info
        MNE `~mne.Info` object corresponding.

    Notes
    -----
    If the argument ``desc`` is not aligned with ``n_channels``, it is ignored and an
    `mne.Info` with the number of channels definbed in ``n_channels`` is created.
    """
    from ..lsl.stream_info import _BaseStreamInfo

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
            ch_names, ch_types, ch_units, manufacturer = _read_desc_dict(
                n_channels, stype, desc
            )
        elif isinstance(desc, _BaseStreamInfo):
            ch_names, ch_types, ch_units, manufacturer = _read_desc_sinfo(
                n_channels, stype, desc
            )

        info = mne_create_info(ch_names, sfreq if sfreq != 0 else 1, ch_types)
        with info._unlock():
            if sfreq == 0:
                info["sfreq"] = sfreq
                info["lowpass"] = 0.0
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
def _read_desc_sinfo(
    n_channels: int, stype: str, desc: _BaseStreamInfo
) -> Tuple[List[str], List[str], List[int], Optional[str]]:
    """Read channel information from a StreamInfo.

    If the StreamInfo is retrieved by resolve_streams, the description will be empty.
    An inlet should be created and the inlet StreamInfo should be used to retrieve the
    channel description.
    """
    if n_channels != desc.n_channels:
        raise RuntimeError(
            "The number of channels expected and the number of channels in the "
            "StreamInfo differ."
        )

    ch_names = [ch_name for ch_name in desc.get_channel_names() if ch_name is not None]
    assert len(ch_names) == n_channels
    assert all(isinstance(elt, str) and len(elt) != 0 for elt in ch_names)

    try:
        ch_types = list()
        for ch_type in desc.get_channel_types():
            ch_type = ch_type.lower().strip()
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
            fiff_unit = _CH_TYPES_DICT[ch_type]["unit"]
            if fiff_unit in _HUMAN_UNITS:
                ch_unit = _HUMAN_UNITS[fiff_unit].get(ch_unit, ch_unit)
            if isinstance(ch_unit, str):
                # try to convert the str to an integer to get the multiplication factor
                try:
                    decimal = Decimal(ch_unit)
                    if int(decimal) == decimal and int(decimal) in _ch_unit_mul_named:
                        ch_unit = int(decimal)
                    else:
                        ch_unit = _ch_unit_mul_named[0]
                except InvalidOperation:
                    ch_unit = _ch_unit_mul_named[0]
            ch_units.append(ch_unit)
    except Exception:
        ch_units = [_ch_unit_mul_named[0]] * n_channels
    assert len(ch_units) == n_channels

    # TODO: retrieve manufacturer from StreamInfo.desc
    manufacturer = None
    return ch_names, ch_types, ch_units, manufacturer


# --------------------- Functions to read from a description dict ----------------------
def _read_desc_dict(
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
        fiff_unit = _CH_TYPES_DICT[ch_type]["unit"]
        if fiff_unit in _HUMAN_UNITS:
            ch_unit = _HUMAN_UNITS[fiff_unit].get(ch_unit, ch_unit)
        if isinstance(ch_unit, str):
            # try to convert the str to an integer to get the multiplication factor
            try:
                decimal = Decimal(ch_unit)
                if int(decimal) == decimal and int(decimal) in _ch_unit_mul_named:
                    ch_unit = int(decimal)
                else:
                    ch_unit = _ch_unit_mul_named[0]
            except InvalidOperation:
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


# ----------------------------- Functions to edit an Info ------------------------------
def _set_channel_units(info: Info, mapping: Dict[str, Union[str, int]]) -> None:
    """Set the channel unit multiplication factor."""
    check_type(mapping, (dict,), "mapping")
    mapping_idx = dict()  # to avoid overwriting the input dictionary
    for ch, unit in mapping.items():
        check_type(ch, (str,), "ch")
        check_value(ch, info.ch_names, "ch")

        # handle channels which are not suppose to have a unit
        idx = info.ch_names.index(ch)
        fiff_unit = info["chs"][idx]["unit"]
        if fiff_unit == FIFF.FIFF_UNIT_NONE:
            raise ValueError(
                f"The channel {ch} type unit is N/A. If you want to set the unit of "
                "this channel, first change its type to one which supports a unit."
            )

        check_type(unit, (str, "int-like"), "unit")
        # convert the unit to a known value
        if isinstance(unit, str):
            if fiff_unit in _HUMAN_UNITS and unit in _HUMAN_UNITS[fiff_unit]:
                mapping_idx[idx] = _HUMAN_UNITS[fiff_unit][unit]
                continue
            else:
                raise ValueError(
                    f"The human-readable unit {unit} for the channel {ch} "
                    f"({info['chs'][idx]['unit']} is unknown to MNE-LSL. Please "
                    "contact the developers on GitHub if you want to add support for "
                    "this unit."
                )
        elif isinstance(unit, int):
            check_value(unit, _ch_unit_mul_named, "unit")
            mapping_idx[idx] = _ch_unit_mul_named[unit]

    # now that the units look safe, set them in Info
    for ch_idx, unit in mapping_idx.items():
        info["chs"][ch_idx]["unit_mul"] = unit
