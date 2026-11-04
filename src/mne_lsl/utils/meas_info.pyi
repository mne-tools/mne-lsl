from typing import Any

import numpy as np
from _typeshed import Incomplete
from mne import Info, Projection
from mne._fiff._digitization import DigPoint

from ..lsl._utils import XMLElement as XMLElement
from ..lsl.stream_info import _BaseStreamInfo as _BaseStreamInfo
from ._checks import check_type as check_type
from ._checks import check_value as check_value
from ._checks import ensure_int as ensure_int
from .logs import logger as logger
from .logs import warn as warn

_CH_TYPES_DICT: Incomplete
_STIM_TYPES: tuple[str, ...]
_HUMAN_UNITS: dict[int, dict[str, int]]
_MAPPING_LSL: Incomplete
_LOC_NAMES: Incomplete

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

    __slots__: Incomplete
    _node: Incomplete
    _siblings: Incomplete
    _index: Incomplete

    def __init__(
        self, node: Any, siblings: list | None = None, index: int = 0
    ) -> None: ...
    def child(self, name: str) -> _DictElement:
        """Get the first child with the given tag, or an empty node."""

    def first_child(self) -> _DictElement:
        """Get the first child, i.e. the text body for a leaf element."""

    def next_sibling(self) -> _DictElement:
        """Get the next occurrence of this tag among its siblings."""

    def empty(self) -> bool:
        """Return True if the node is missing or empty."""

    def value(self) -> str:
        """Get the text value of the node, or an empty string."""

    def child_value(self, name: str) -> str:
        """Get the text value of the first child with the given tag."""

def _read_desc(
    desc_node: XMLElement | _DictElement, n_channels: int, stype: str
) -> tuple[list[str], list[str], list[int], str | None]:
    """Read the channel names, types, units and manufacturer from a description node.

    The node exposes the :class:`~mne_lsl.lsl._utils.XMLElement` read API, so this reads
    identically from a live ``StreamInfo`` description and from a dictionary loaded from
    an XDF file.
    """

def _normalize_type_and_unit(
    ch_type: str | None, ch_unit: str | None, stype: str
) -> tuple[str, int]:
    """Normalize one channel's type and unit to MNE-compatible values."""

def _read_channel_field(
    desc_node: XMLElement | _DictElement, name: str
) -> list[str | None] | None:
    """Read one 'channels/channel/<name>' field for every channel in the description.

    Return a value per channel found in the description, ``None`` for the channels
    missing this field, and ``None`` if the field is absent for every channel.
    Validating the length against the expected channel count is left to the caller.
    """

def _add_channel_metadata(
    info: Info, desc_node: XMLElement | _DictElement, n_channels: int, sfreq: float
) -> None:
    """Complete an Info in-place with the FIFF information found in the description.

    This recovers the integer channel codes, the channel locations, the filters, the
    SSP projectors and the digitization, from the same description tree whether it comes
    from a live ``StreamInfo`` or from a dictionary loaded from an XDF file.
    """

def _read_locs(desc_node: XMLElement | _DictElement, n_channels: int) -> np.ndarray:
    """Read the channel locations from the description node."""

def _read_projectors(desc_node: XMLElement | _DictElement) -> list[Projection]:
    """Read the SSP projectors from the description node."""

def _read_digitization(desc_node: XMLElement | _DictElement) -> list[DigPoint]:
    """Read the digitization points from the description node."""

def _get_fiff_int_named(
    value: str | None, name: str, mapping: dict[int, int]
) -> int | None:
    """Try to retrieve the FIFF integer code from the str representation."""

def _set_channel_units(info: Info, mapping: dict[str, str | int]) -> None:
    """Set the channel unit multiplication factor."""
