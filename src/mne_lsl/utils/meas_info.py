from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

import numpy as np
from mne import Info, Projection
from mne import create_info as mne_create_info
from mne._fiff._digitization import DigPoint
from mne._fiff.constants import (
    FIFF,
    _ch_coil_type_named,
    _ch_kind_named,
    _ch_unit_mul_named,
    _coord_frame_named,
    _dig_cardinal_named,
    _dig_kind_named,
)
from mne._fiff.pick import get_channel_type_constants

from ._checks import check_type, check_value, ensure_int
from .logs import logger, warn

if TYPE_CHECKING:
    from typing import Any

    from ..lsl._utils import XMLElement
    from ..lsl.stream_info import _BaseStreamInfo


_CH_TYPES_DICT = get_channel_type_constants(include_defaults=True)
_STIM_TYPES: tuple[str, ...] = (
    "marker",
    "markers",
    "stim",
)
_HUMAN_UNITS: dict[int, dict[str, int]] = {
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
# mapping between the mne-lsl channel attribute names and the XML element names
_MAPPING_LSL = {
    "ch_name": "label",
    "ch_type": "type",
    "ch_unit": "unit",
}
# fmt: off
_LOC_NAMES = (
    "R0x", "R0y", "R0z",
    "Exx", "Exy", "Exz",
    "Eyx", "Eyy", "Eyz",
    "Ezx", "Ezy", "Ezz",
)
# fmt: on


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

    The ``desc`` provided as a :class:`~mne_lsl.lsl.StreamInfo` (live stream) and as a
    :class:`dict` (loaded from an XDF file) describe the same tree. Each is wrapped in a
    node exposing the ``XMLElement`` read API and parsed by the same functions, so the
    channel locations, projectors, digitization and filters are recovered the same way
    from either source.
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

    if desc is None:
        logger.info("No description provided. Creating a default Info object.")
        return _create_default_info(n_channels, sfreq, stype)

    # wrap both sources in a node exposing the same read API
    desc_node = desc.desc if isinstance(desc, _BaseStreamInfo) else _DictElement(desc)

    try:
        ch_names, ch_types, ch_units, manufacturer = _read_desc(
            desc_node, n_channels, stype
        )
        info = mne_create_info(ch_names, sfreq if sfreq != 0 else 1, ch_types)
        with info._unlock():
            if sfreq == 0:
                info["sfreq"] = sfreq
                info["lowpass"] = 0.0
                info["highpass"] = 0.0
            for ch, ch_unit in zip(info["chs"], ch_units, strict=True):
                ch["unit_mul"] = ch_unit
        # add manufacturer information if available
        info["device_info"] = dict()
        if isinstance(manufacturer, str):
            info["device_info"]["model"] = manufacturer
        info._check_consistency()
    except Exception:
        warn(
            "Something went wrong while reading the channel description. Defaulting to "
            "channel IDs and MNE-compatible stream type."
        )
        return _create_default_info(n_channels, sfreq, stype)

    # the base info is valid; complete it with the richer FIFF information present in
    # the description. A failure here is not fatal to the names/types/units above.
    _add_channel_metadata(info, desc_node, n_channels, sfreq)
    return info


def _create_default_info(
    n_channels: int,
    sfreq: float,
    stype: str,
) -> Info:
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
    info = mne_create_info(n_channels, 1.0, stype)
    info["device_info"] = dict()
    with info._unlock():
        info["sfreq"] = sfreq
    info["device_info"] = dict()
    return info


# ------------------- Adapter presenting a dict as an XMLElement -----------------------
class _DictElement:
    """Read-only view of a pyxdf description :class:`dict` as an ``XMLElement``.

    ``pyxdf`` represents an XML element as a :class:`dict` mapping each child tag to the
    :class:`list` of that tag's occurrences; the text body of a leaf is stored as that
    tag's value, either as a one-element list (``{"label": ["Fz"]}``) or as the bare
    value (``{"label": "Fz"}``). This class exposes the subset of the
    :class:`~mne_lsl.lsl._utils.XMLElement` navigation API used to read a stream
    description, so a description loaded from an XDF file is parsed by the exact same
    functions as a live :class:`~mne_lsl.lsl.StreamInfo`.
    """

    __slots__ = ("_index", "_node", "_siblings")

    def __init__(self, node: Any, siblings: list | None = None, index: int = 0) -> None:
        # _node is a dict (element), a str (text body), or None (empty / missing node).
        self._node = node
        self._siblings = siblings  # list this node belongs to, to walk next_sibling()
        self._index = index

    def child(self, name: str) -> _DictElement:
        """Get the first child with the given tag, or an empty node."""
        if not isinstance(self._node, dict) or name not in self._node:
            return _DictElement(None)
        values = self._node[name]
        if isinstance(values, list):
            if len(values) == 0:
                return _DictElement(None)
            return _DictElement(values[0], siblings=values, index=0)
        return _DictElement(values, siblings=[values], index=0)

    def first_child(self) -> _DictElement:
        """Get the first child, i.e. the text body for a leaf element."""
        # a text body is stored directly as the node value, so the leaf is its own
        # first child from the point of view of value().
        return self if isinstance(self._node, str) else _DictElement(None)

    def next_sibling(self) -> _DictElement:
        """Get the next occurrence of this tag among its siblings."""
        index = self._index + 1
        if self._siblings is None or index >= len(self._siblings):
            return _DictElement(None)
        return _DictElement(self._siblings[index], siblings=self._siblings, index=index)

    def empty(self) -> bool:
        """Return True if the node is missing or empty."""  # noqa: D401
        return self._node is None

    def value(self) -> str:
        """Get the text value of the node, or an empty string."""
        return self._node if isinstance(self._node, str) else ""

    def child_value(self, name: str) -> str:
        """Get the text value of the first child with the given tag."""
        return self.child(name).first_child().value()


# ----------------------------- Functions to read a description ------------------------
def _read_desc(
    desc_node: XMLElement | _DictElement, n_channels: int, stype: str
) -> tuple[list[str], list[str], list[int], str | None]:
    """Read the channel names, types, units and manufacturer from a description node.

    The node exposes the :class:`~mne_lsl.lsl._utils.XMLElement` read API, so this reads
    identically from a live ``StreamInfo`` description and from a dictionary loaded from
    an XDF file.
    """
    ch_names = _read_channel_field(desc_node, "ch_name")
    if ch_names is None or len(ch_names) != n_channels:
        raise RuntimeError(
            "The stream description does not contain a valid channel name for each of "
            f"the {n_channels} channels."
        )
    if any(name is None or len(name) == 0 for name in ch_names):
        raise RuntimeError("The stream description contains an empty channel name.")

    raw_types = _read_channel_field(desc_node, "ch_type")
    raw_units = _read_channel_field(desc_node, "ch_unit")
    ch_types = list()
    ch_units = list()
    for k in range(n_channels):
        raw_type = raw_types[k] if raw_types is not None else None
        raw_unit = raw_units[k] if raw_units is not None else None
        ch_type, ch_unit = _normalize_type_and_unit(raw_type, raw_unit, stype)
        ch_types.append(ch_type)
        ch_units.append(ch_unit)

    manufacturer = desc_node.child_value("manufacturer") or None
    return ch_names, ch_types, ch_units, manufacturer


def _normalize_type_and_unit(
    ch_type: str | None, ch_unit: str | None, stype: str
) -> tuple[str, int]:
    """Normalize one channel's type and unit to MNE-compatible values."""
    ch_type = stype if ch_type is None else ch_type.lower().strip()
    ch_type = "stim" if ch_type in _STIM_TYPES else ch_type
    ch_type = ch_type if ch_type in _CH_TYPES_DICT else stype

    ch_unit = _ch_unit_mul_named[0] if ch_unit is None else ch_unit.lower().strip()
    # some channel types (e.g. 'chpi', 'gof') carry no unit in the FIFF constants
    fiff_unit = _CH_TYPES_DICT[ch_type].get("unit")
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
    return ch_type, ch_unit


def _read_channel_field(
    desc_node: XMLElement | _DictElement, name: str
) -> list[str | None] | None:
    """Read one 'channels/channel/<name>' field for every channel in the description.

    Return a value per channel found in the description, ``None`` for the channels
    missing this field, and ``None`` if the field is absent for every channel.
    Validating the length against the expected channel count is left to the caller.
    """
    channels = desc_node.child("channels")
    if channels.empty():
        return None
    name = _MAPPING_LSL.get(name, name)

    ch_infos = list()
    ch = channels.child("channel")
    while not ch.empty():
        ch_info = ch.child(name).first_child().value()
        ch_infos.append(ch_info if len(ch_info) != 0 else None)
        ch = ch.next_sibling()

    if all(ch_info is None for ch_info in ch_infos):
        return None
    return ch_infos


def _add_channel_metadata(
    info: Info, desc_node: XMLElement | _DictElement, n_channels: int, sfreq: float
) -> None:
    """Complete an Info in-place with the FIFF information found in the description.

    This recovers the integer channel codes, the channel locations, the filters, the
    SSP projectors and the digitization, from the same description tree whether it comes
    from a live ``StreamInfo`` or from a dictionary loaded from an XDF file.
    """
    # integer channel codes
    kinds = _read_channel_field(desc_node, "kind")
    coil_types = _read_channel_field(desc_node, "coil_type")
    coord_frames = _read_channel_field(desc_node, "coord_frame")
    range_cals = _read_channel_field(desc_node, "range_cal")
    locs = _read_locs(desc_node, n_channels)

    with info._unlock(update_redundant=True):
        for var, name, fiff_named in (
            (kinds, "kind", _ch_kind_named),
            (coil_types, "coil_type", _ch_coil_type_named),
            (coord_frames, "coord_frame", _coord_frame_named),
        ):
            if var is None:
                continue
            for k, value in enumerate(var):
                value = _get_fiff_int_named(value, name, fiff_named)
                if value is not None:
                    info["chs"][k][name] = value

        if range_cals is not None:
            for k, range_cal in enumerate(range_cals):
                if range_cal is not None:
                    try:
                        info["chs"][k]["range"] = 1.0
                        info["chs"][k]["cal"] = float(range_cal)
                    except ValueError:
                        warn(f"Could not cast 'range_cal' factor {range_cal} to float.")
        for k, loc in enumerate(locs):
            info["chs"][k]["loc"] = loc

    # filters
    filters = desc_node.child("filters")
    if not filters.empty() and sfreq != 0:
        highpass = filters.child("highpass").first_child().value()
        lowpass = filters.child("lowpass").first_child().value()
        with info._unlock():
            for name, value in zip(
                ("highpass", "lowpass"), (highpass, lowpass), strict=True
            ):
                if len(value) != 0:
                    try:
                        info[name] = float(value)
                    except ValueError:
                        warn(f"Could not cast '{name}' {value} to float.")
    elif not filters.empty() and sfreq == 0:
        warn(
            "Node 'filters' found in the description of an irregularly sampled stream. "
            "This is inconsistent and will be skipped."
        )

    projs = _read_projectors(desc_node)
    dig = _read_digitization(desc_node)
    with info._unlock(update_redundant=True, check_after=True):
        info["projs"] = projs
        if len(dig) != 0:
            info["dig"] = dig


def _read_locs(desc_node: XMLElement | _DictElement, n_channels: int) -> np.ndarray:
    """Read the channel locations from the description node."""
    locs = list()
    channels = desc_node.child("channels")
    ch = channels.child("channel")
    while not ch.empty():
        loc_array = list()
        loc = ch.child("loc")
        for loc_name in _LOC_NAMES:
            try:
                value = float(loc.child(loc_name).first_child().value())
            except ValueError:
                value = np.nan
            loc_array.append(value)
        locs.append(loc_array)
        ch = ch.next_sibling()
    if len(locs) == 0:
        return np.array([[np.nan] * 12] * n_channels)
    return np.array(locs)


def _read_projectors(desc_node: XMLElement | _DictElement) -> list[Projection]:
    """Read the SSP projectors from the description node."""
    projs = list()
    projectors = desc_node.child("projectors")
    projector = projectors.child("projector")
    while not projector.empty():
        desc = projector.child("desc").first_child().value()
        if len(desc) == 0:
            warn("An SSP projector without description was found. Skipping.")
            projector = projector.next_sibling()
            continue
        kind = projector.child("kind").first_child().value()
        try:
            kind = int(kind)
        except ValueError:
            warn(f"Could not cast the SSP kind {kind} to integer.")
            projector = projector.next_sibling()
            continue

        ch_names = list()
        ch_datas = list()
        data = projector.child("data")
        ch = data.child("channel")
        while not ch.empty():
            ch_name = ch.child("label").first_child().value()
            if len(ch_name) == 0:
                warn(
                    "SSP projector has an empty-channel label. The channel will be "
                    "skipped."
                )
                ch = ch.next_sibling()
                continue
            ch_data = ch.child("data").first_child().value()
            try:
                ch_data = float(ch_data)
            except ValueError:
                warn(
                    f"Could not cast the SSP value {ch_data} for channel {ch_name} to "
                    "float.",
                )
                ch = ch.next_sibling()
                continue
            ch_names.append(ch_name)
            ch_datas.append(ch_data)
            ch = ch.next_sibling()

        assert len(ch_names) == len(ch_datas)  # sanity-check
        proj_data = {
            "nrow": 1,
            "ncol": len(ch_names),
            "row_names": None,
            "col_names": ch_names,
            "data": np.array(ch_datas).reshape(1, -1),
        }
        projs.append(Projection(data=proj_data, desc=desc, kind=kind))
        projector = projector.next_sibling()
    return projs


def _read_digitization(desc_node: XMLElement | _DictElement) -> list[DigPoint]:
    """Read the digitization points from the description node."""
    dig = desc_node.child("dig")
    dig_points = list()
    point = dig.child("point")
    while not point.empty():
        kind = point.child("kind").first_child().value()
        kind = _get_fiff_int_named(kind, "dig_kind", _dig_kind_named)
        if kind is None:
            point = point.next_sibling()
            continue
        ident = point.child("ident").first_child().value()
        if kind == FIFF.FIFFV_POINT_CARDINAL:
            ident = _get_fiff_int_named(ident, "dig_ident", _dig_cardinal_named)
        else:
            try:
                ident = int(ident)
            except ValueError:
                warn(f"Could not cast 'ident' {ident} to integer.")
                point = point.next_sibling()
                continue
        loc = point.child("loc")
        r = [loc.child(pos).first_child().value() for pos in ("X", "Y", "Z")]
        if ident is None or any(len(elt) == 0 for elt in r):
            point = point.next_sibling()
            continue
        try:
            r = np.array([float(elt) for elt in r], dtype=np.float32)
        except ValueError:
            warn(f"Could not cast dig point location {r} to float.")
            point = point.next_sibling()
            continue
        dig_points.append(
            DigPoint(kind=kind, ident=ident, r=r, coord_frame=FIFF.FIFFV_COORD_HEAD)
        )
        point = point.next_sibling()
    return dig_points


def _get_fiff_int_named(
    value: str | None, name: str, mapping: dict[int, int]
) -> int | None:
    """Try to retrieve the FIFF integer code from the str representation."""
    if value is None:
        return None
    try:
        idx = int(value)
        value = mapping[idx]
        return value
    except ValueError:
        warn(f"Could not cast '{name}' {value} to integer.")
    except KeyError:
        warn(
            f"Could not convert '{name}' {int(value)} to a known FIFF "
            "code: {tuple(mapping.keys())}.",
        )
    return None


# ----------------------------- Functions to edit an Info ------------------------------
def _set_channel_units(info: Info, mapping: dict[str, str | int]) -> None:
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
