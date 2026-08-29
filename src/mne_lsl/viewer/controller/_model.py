"""Channel model backing the Channels page and the trace display.

Single source of truth for the channel presentation: name, type, unit, bad state,
visibility and display order live here and nowhere else. Metadata edits route to the
stream operations (:meth:`~mne_lsl.stream.BaseStream.rename_channels`,
:meth:`~mne_lsl.stream.BaseStream.set_channel_types`,
:meth:`~mne_lsl.stream.BaseStream.set_channel_units` and ``stream.info['bads']``), while
visibility and display order are viewer presentation state which the trace display
mirrors.

Three index spaces meet here, and the naming keeps them apart:

- **acquisition index**, ``Channel.acq_index``, the channel identity, which
  ``info.ch_names``, the buffer and ``picks`` speak, and which seeds the trace color. It
  never changes for the life of a model;
- **presentation index**, the model row, hidden channels included. Every Qt index, the
  delegate, the selection and the context menu speak this one, and a bare ``row`` in
  this module always means it;
- **visible-row index**, what the trace display stacks. It exists only inside
  :meth:`ChannelModel.visible_acq_indices`, which is the single translation site.

The five metadata fields of a :class:`Channel` are a **cache** of ``stream.info``,
overwritten wholesale after every write; the stream stays the only writer of metadata.
Reading ``info`` per row per paint instead would cost a ``get_channel_units`` call per
row, while the whole read is 95 µs at 256 channels -- and re-reading after every write
is what makes the cache unable to diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from mne._fiff.constants import FIFF, _ch_unit_mul_named
from mne._fiff.meas_info import _unit2human
from qtpy.QtCore import QAbstractListModel, QModelIndex, Qt, Signal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from qtpy.QtCore import QObject

    from ...stream import BaseStream

# Choices offered by the inspector combo boxes and the context-menu submenus.
CH_TYPES = ["eeg", "eog", "ecg", "emg", "stim", "misc"]

# The unit of a channel is two FIFF fields: 'unit', the physical-unit *kind*
# (FIFF_UNIT_*), and 'unit_mul', the power-of-ten *multiplier* (FIFF_UNITM_*). MNE
# derives the kind from the channel type, thus the Type control writes a kind and the
# Unit control writes the multiplier only.
#
# This is the only hand-maintained unit table: per kind the viewer can produce, its
# symbol and the multiplier ladder the Unit control offers, ascending. One table rather
# than two keyed by the same seven kinds, which would have to be kept in sync by hand.
#
# The symbols deliberately differ from MNE's own private kind-to-symbol table on two
# entries: '°C' rather than 'C', which would read as Coulomb, and 'V/m²', which that
# table does not carry at all even though 'csd' produces the kind.
#
# The ladders are deliberately a shortlist of the 17 multipliers FIFF names: nobody
# picks exavolts, and the two rungs a user would expect between mV and µV -- 100 µV and
# 10 µV, i.e. -4 and -5 -- are simply not FIFF multipliers and cannot be written at all.
_KINDS: dict[int, tuple[str, list[int]]] = {
    int(FIFF.FIFF_UNIT_V): ("V", [-6, -3, -2, -1, 0]),
    int(FIFF.FIFF_UNIT_T): ("T", [-15, -12, 0]),
    int(FIFF.FIFF_UNIT_T_M): ("T/m", [-15, -12, 0]),
    int(FIFF.FIFF_UNIT_MOL): ("M", [-6, 0]),
    int(FIFF.FIFF_UNIT_V_M2): ("V/m²", [-6, -3, 0]),
    int(FIFF.FIFF_UNIT_CEL): ("°C", [0]),
    int(FIFF.FIFF_UNIT_S): ("S", [0]),
}

# SI prefix of every power of ten which is a multiple of three, over the range FIFF
# names a multiplier for.
_SI_PREFIX: dict[int, str] = {
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
}

# Shown for a channel whose kind is not a physical quantity, e.g. a misc channel. It is
# read-only: it is never offered by 'unit_choices' and never accepted by 'set_unit',
# because writing a multiplier onto such a channel raises -- giving it a unit is a Type
# change.
_NO_UNIT = "(none)"

# The offered ladders, filtered through the multipliers MNE will actually accept: it
# validates 'unit_mul' against the 17 named FIFF constants, so a rung which is not one
# of them would raise from inside a Qt slot. Filtering here rather than testing the
# table makes an unwritable rung impossible to offer instead of merely detectable.
_MULTIPLIERS: dict[int, list[int]] = {
    kind: [mul for mul in muls if mul in _ch_unit_mul_named]
    for kind, (_symbol, muls) in _KINDS.items()
}

# Custom roles read by the row delegate and by the inspector. The list has a single
# column, thus the index of a row carries the entire channel.
_UR = int(Qt.ItemDataRole.UserRole)
NameRole = _UR + 1
TypeRole = _UR + 2
UnitRole = _UR + 3
VisibleRole = _UR + 4
BadRole = _UR + 5


def unit_label(kind: int, mul: int) -> str:
    """Return the human label of a ``(kind, multiplier)`` unit pair.

    Parameters
    ----------
    kind : int
        FIFF unit kind, ``FIFF_UNIT_*``.
    mul : int
        FIFF unit multiplier, ``FIFF_UNITM_*``, i.e. the power of ten.

    Returns
    -------
    label : str
        The human label, e.g. ``'µV'``.

    Notes
    -----
    The label is built rather than looked up, so that it covers every multiplier a
    stream may declare and not only the ones the Unit control offers: the largest
    multiple of three at or below ``mul`` selects the SI prefix and the remainder is a
    leading coefficient, e.g. ``-1`` on Volts reads ``'100 mV'`` and ``-6`` ``'µV'``.

    This is read from a paint path and therefore never raises: an unregistered kind or a
    multiplier past the FIFF range falls back to ``'(none)'``.
    """
    registered = _KINDS.get(int(kind))
    if registered is None:
        return _NO_UNIT
    symbol = registered[0]
    # floor division, thus correct for a negative multiplier
    power = 3 * (int(mul) // 3)
    prefix = _SI_PREFIX.get(power)
    if prefix is None:
        return _NO_UNIT
    coefficient = 10 ** (int(mul) - power)
    if coefficient == 1:
        return f"{prefix}{symbol}"
    return f"{coefficient} {prefix}{symbol}"


# Every offered label and its pair, both generated from the ladders above so that
# 'unit_label(*unit_pair(label)) == label' holds by construction.
_LABEL_TO_PAIR: dict[str, tuple[int, int]] = {
    unit_label(kind, mul): (kind, mul)
    for kind, muls in _MULTIPLIERS.items()
    for mul in muls
}
UNIT_LABELS = list(_LABEL_TO_PAIR)


def unit_pair(label: str) -> tuple[int, int]:
    """Return the ``(kind, multiplier)`` pair of a unit ``label``.

    Parameters
    ----------
    label : str
        Human unit label, e.g. ``'µV'``.

    Returns
    -------
    kind : int
        FIFF unit kind, ``FIFF_UNIT_*``.
    mul : int
        FIFF unit multiplier, ``FIFF_UNITM_*``.

    Notes
    -----
    An unknown label resolves to ``(FIFF_UNIT_NONE, FIFF_UNITM_NONE)`` rather than
    raising, as this is called from Qt slots; :meth:`ChannelModel.set_unit` is where an
    unknown label is refused.
    """
    return _LABEL_TO_PAIR.get(label, (int(FIFF.FIFF_UNIT_NONE), 0))


def unit_choices(kind: int | None) -> list[str]:
    """Return the unit labels offered for a unit ``kind``.

    Parameters
    ----------
    kind : int | None
        FIFF unit kind, ``FIFF_UNIT_*``, or ``None`` for a selection spanning several
        kinds.

    Returns
    -------
    labels : list of str
        The curated multiplier ladder of that kind, empty when there is none.

    Notes
    -----
    Keyed on the kind and not on the channel type, because the kind is what constrains
    the answer: two channels of different types which share a kind, e.g. an eeg and an
    ecg channel, must offer the same ladder, so a mixed-type selection keeps a working
    Unit control. Every returned label shares one kind, which is what makes it
    impossible for the Unit control to write a kind.

    An empty list is what disables the control: a channel whose kind is not a physical
    quantity, e.g. a misc channel, acquires one through the Type control.
    """
    if kind is None:
        return []
    return [unit_label(kind, mul) for mul in _MULTIPLIERS.get(int(kind), [])]


class Original(NamedTuple):
    """Acquisition values of one channel, restored by a metadata reset.

    Parameters
    ----------
    name : str
        Channel name the stream declared.
    ch_type : str
        Channel type the stream declared.
    unit_kind : int
        Physical unit kind, ``FIFF_UNIT_*``.
    unit_mul : int
        Decimal power-of-ten multiplier, ``FIFF_UNITM_*``, forced to ``0`` when
        ``unit_kind`` is not a physical quantity.
    bad : bool
        Whether the channel was declared bad.

    Notes
    -----
    Named rather than a bare tuple because it is read at six sites: swapping the two
    adjacent integer fields, i.e. reading the kind as the multiplier, type-checks and
    passes the happy path.

    ``unit_mul`` is normalized on capture because a multiplier cannot be written back
    onto a channel whose kind is not a physical quantity -- MNE refuses it outright.
    Recording the acquisition multiplier of such a channel would make this a baseline
    the reset cannot restore, silently, and nothing on screen would show the loss, since
    both the acquisition and the restored value render as a unit-less label.
    """

    name: str
    ch_type: str
    unit_kind: int
    unit_mul: int
    bad: bool


@dataclass
class Channel:
    """Editable state of one channel and its original stream values.

    Parameters
    ----------
    name : str
        Current channel name.
    ch_type : str
        Current channel type.
    unit_kind : int
        Physical unit kind, ``FIFF_UNIT_*``.
    unit_mul : int
        Decimal power-of-ten multiplier, ``FIFF_UNITM_*``.
    visible : bool
        Whether the channel is drawn by the trace display, i.e. presentation state.
    bad : bool
        Whether the channel is marked bad, mirroring ``stream.info['bads']``.
        Independent of the visibility: a bad channel stays visible, dimmed and struck
        through, unless it is also hidden.
    acq_index : int
        Index of the channel in acquisition order.
    orig : Original
        Acquisition values, restored by a reset.

    Notes
    -----
    Exactly three fields are model state -- ``visible``, ``acq_index`` and ``orig``. The
    other five cache ``stream.info`` and are overwritten wholesale after every write.
    Every field but ``acq_index`` therefore has a placeholder default: the metadata is
    filled in by the first stream read and ``orig`` captured from it.
    """

    acq_index: int
    name: str = ""
    ch_type: str = ""
    unit_kind: int = int(FIFF.FIFF_UNIT_NONE)
    unit_mul: int = 0
    visible: bool = True
    bad: bool = False
    orig: Original = Original("", "", int(FIFF.FIFF_UNIT_NONE), 0, False)

    @property
    def unit(self) -> str:
        """Human unit label of the ``(kind, multiplier)`` pair."""
        return unit_label(self.unit_kind, self.unit_mul)

    @property
    def original(self) -> str:
        """One-line acquisition value, e.g. ``'ECG ecg mV good'``.

        Notes
        -----
        Formatted once here rather than at each of the two sites which show it -- the
        row tooltip and the inspector's originals line -- as two spellings of the same
        four fields read as two different quantities.
        """
        state = "bad" if self.orig.bad else "good"
        label = unit_label(self.orig.unit_kind, self.orig.unit_mul)
        return f"{self.orig.name} {self.orig.ch_type} {label} {state}"


class ChannelModel(QAbstractListModel):
    """List model holding the channel metadata, visibility and display order.

    Parameters
    ----------
    stream : BaseStream
        The connected stream the metadata is read from and written back to.
    parent : QObject | None
        Parent object.

    Attributes
    ----------
    layout_changed : Signal
        Emitted once per user action which changed the display order or the visibility.
    metadata_changed : Signal
        Emitted once per user action which changed a name, type, unit or bad state, i.e.
        for an identity-preserving change.

    Notes
    -----
    The two signals are the entire cross-widget contract: whoever owns both this model
    and a trace display pushes :meth:`ChannelModel.visible_acq_indices` on the first and
    re-reads the metadata on the second. The edge is one-way, so no signal blocking is
    needed at the boundary.

    A bulk mutator emits **one** spanning ``dataChanged`` and **one** coarse signal,
    never one per row: at 256 channels a per-row emission costs the page ~95 ms against
    ~2 ms, since it is the page handler which runs 256 times rather than Qt.

    Every metadata write is a silent no-op over a disconnected stream, as the
    presentation mutators already were: the stream is the only writer of metadata, so
    there is nothing to write to, and reading ``info`` off a disconnected stream raises.
    The inspector's Bad, Type, Unit and Reset controls stay live while a stream is down,
    so this is a reachable path and not a defensive check.
    """

    layout_changed = Signal()
    metadata_changed = Signal()

    def __init__(self, stream: BaseStream, parent: QObject | None = None) -> None:
        """Initialize the model from the metadata of a connected stream."""
        super().__init__(parent)
        self._stream = stream
        self._rows: list[Channel] = []
        self._build()

    def _build(self) -> None:
        """Build one channel per acquisition channel, in acquisition order.

        Notes
        -----
        A disconnected stream yields zero rows rather than raising, mirroring the trace
        display; :meth:`ChannelModel.refresh` is the way back once it is connected.
        """
        self._rows = []
        if not self._stream.connected:
            return
        self._rows = [
            Channel(acq_index=acq) for acq in range(len(self._stream.info.ch_names))
        ]
        self._read_stream()
        for channel in self._rows:
            # the acquisition baseline, captured once and never written again: this is
            # what a metadata reset restores. The multiplier of a channel whose kind is
            # not a physical quantity is dropped rather than recorded, as it cannot be
            # written back -- see 'Original'.
            unit_less = channel.unit_kind == int(FIFF.FIFF_UNIT_NONE)
            channel.orig = Original(
                channel.name,
                channel.ch_type,
                channel.unit_kind,
                0 if unit_less else channel.unit_mul,
                channel.bad,
            )

    def _read_stream(self) -> None:
        """Refresh the cached metadata of every channel from the stream.

        Raises
        ------
        ValueError
            If the stream no longer holds one channel per row, i.e. the channel *set*
            changed behind the model. Indexing the three lists by the acquisition index
            would otherwise raise ``IndexError`` from whichever mutator happened to run
            next, after its write had already landed.

        Notes
        -----
        The one place the stream metadata is read. The picks are an explicit integer
        range rather than the ``None`` default: measured at the model level, resolving
        ``None`` costs ~10x more per call, and for the units it used to silently drop
        the bad channels and misalign the result.

        The names come from ``info['ch_names']``, which is already the deduplicated list
        -- ``Cz`` twice becomes ``Cz-0`` / ``Cz-1`` -- and costs 1.4 µs. Reading them
        back from the LSL stream description instead would re-emit the duplicate-name
        warning on every metadata change.
        """
        if not self._stream.connected:
            return
        info = self._stream.info
        names = list(info.ch_names)
        if len(names) != len(self._rows):
            raise ValueError(
                f"The stream holds {len(names)} channels against the model's "
                f"{len(self._rows)}; the channel model is stale and must be refreshed."
            )
        bads = set(info["bads"])
        picks = list(range(len(names)))
        types = self._stream.get_channel_types(picks=picks)
        units = self._stream.get_channel_units(picks=picks)
        for channel in self._rows:
            acq = channel.acq_index
            channel.name = names[acq]
            channel.ch_type = types[acq]
            channel.unit_kind = int(units[acq][0])
            channel.unit_mul = int(units[acq][1])
            channel.bad = names[acq] in bads

    # -- introspection -----------------------------------------------------------------
    def channel(self, row: int) -> Channel:
        """Return the channel at display ``row``.

        Parameters
        ----------
        row : int
            Display row index.

        Returns
        -------
        channel : Channel
            The channel record.

        Raises
        ------
        ValueError
            If ``row`` is not a display row. A negative row is refused rather than
            wrapped: Python list indexing would hand back the *last* channel, so a
            caller reading row ``-1`` would silently inspect the wrong one.
        """
        if not 0 <= row < len(self._rows):
            raise ValueError(
                f"The display row must be in [0, {len(self._rows)}), got {row}."
            )
        return self._rows[row]

    @property
    def n_channels(self) -> int:
        """Total number of acquisition channels of the stream.

        Notes
        -----
        Equal to :meth:`ChannelModel.rowCount` at all times: hiding and reordering never
        change the row count. It carries the same meaning as the total channel count of
        the trace display, so a status bar can cross-check the two.
        """
        return len(self._rows)

    # -- Qt model interface ------------------------------------------------------------
    def rowCount(self, parent: QModelIndex | None = None) -> int:
        """Return the number of channels, ``0`` for a valid parent."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Return the item flags of ``index``; editing goes through the inspector.

        Notes
        -----
        Deliberately neither editable nor drag- nor drop-enabled: the metadata is edited
        through the inspector and the context menu, and reordering is command-driven.
        """
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        """Return the channel data of ``index`` under ``role``."""
        if not index.isValid():
            return None
        channel = self._rows[index.row()]
        if role == NameRole:
            return channel.name
        if role == TypeRole:
            return channel.ch_type
        if role == UnitRole:
            return channel.unit
        if role == VisibleRole:
            return channel.visible
        if role == BadRole:
            return channel.bad
        if role == Qt.ItemDataRole.DisplayRole:
            return channel.name  # keyboard search and the accessible name
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"acquisition value: {channel.original}"
        return None

    def setData(
        self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        """Apply a per-row visibility or bad edit, e.g. from the eye toggle.

        Notes
        -----
        Both edits are routed through the bulk mutators rather than writing the cached
        field, so that the single-row and the bulk path share one writer -- of
        ``stream.info['bads']`` for the bad state, and of the visibility, whose funnel
        also carries the row bounds check. The emission is identical either way.
        """
        if not index.isValid():
            return False
        row = index.row()
        if role == VisibleRole:
            self.set_visible([row], bool(value))
            return True
        if role == BadRole:
            self.set_bad([row], bool(value))
            return True
        return False

    # -- bulk edits, applied to an entire selection ------------------------------------
    def set_visible(self, rows: Iterable[int], value: bool) -> None:
        """Set the visibility of ``rows`` to ``value``.

        Parameters
        ----------
        rows : iterable of int
            Display rows. An empty iterable is a silent no-op, so that an empty
            selection does not make the trace display rebuild its layout for nothing.
        value : bool
            Whether the channels are drawn.

        Raises
        ------
        ValueError
            If a row is not a display row.

        Notes
        -----
        The channels are resolved before any of them is written, so a refused row leaves
        the visibility untouched. This is the one field with no stream round-trip to
        correct it: a half-applied hide which emitted nothing would leave
        :meth:`ChannelModel.visible_acq_indices` and the display diverged permanently.
        """
        rows = _sorted_rows(rows, len(self._rows))
        if not rows:
            return
        channels = [self._rows[row] for row in rows]
        for channel in channels:
            channel.visible = bool(value)
        self._emit_layout(rows[0], rows[-1])

    def set_bad(self, rows: Iterable[int], value: bool) -> None:
        """Mark ``rows`` bad or good, writing to ``stream.info['bads']``.

        Parameters
        ----------
        rows : iterable of int
            Display rows. An empty iterable is a silent no-op.
        value : bool
            Whether the channels are marked bad.

        Raises
        ------
        ValueError
            If a row is not a display row, or if a channel of ``rows`` is absent from
            the stream, i.e. the model is stale.

        Notes
        -----
        The bad state is independent of the visibility, thus this never touches it: a
        bad channel stays drawn, dimmed and struck through, unless it is also hidden.

        The new list is the current one plus or minus the affected names, so a channel
        which was already bad and is outside ``rows`` stays bad. Assigning an unknown
        name raises, which is what surfaces a stale model as an exception rather than as
        a silent no-op.
        """
        rows = _sorted_rows(rows, len(self._rows))
        if not rows or not self._stream.connected:
            return
        names = set(self._names(rows))
        bads = set(self._stream.info["bads"])
        try:
            self._write_bads(bads | names if value else bads - names)
        finally:
            self._read_stream()
        self._emit_metadata(rows[0], rows[-1])

    def set_type(self, rows: Iterable[int], value: str) -> None:
        """Set the channel type of ``rows`` to ``value``.

        Parameters
        ----------
        rows : iterable of int
            Display rows. An empty iterable is a silent no-op.
        value : str
            The new channel type, one of :data:`CH_TYPES`.

        Raises
        ------
        ValueError
            If ``value`` is not an offered channel type, if a row is not a display row,
            if a channel of ``rows`` is absent from the stream, i.e. the model is stale,
            or if a channel declares a unit ``set_channel_types`` cannot name.

        Notes
        -----
        ``value`` is validated before the empty-row shortcut, so an unknown type is
        refused whatever the selection is rather than only when something is selected.

        Every other check runs before the write because ``set_channel_types`` is **not**
        all-or-nothing across its mapping: it mutates the valid channels and only then
        raises on an invalid one, so a stale model would half-apply the edit. The unit
        check is the same failure in a different disguise -- MNE looks the *current*
        unit up in a table which is missing two of the kinds it can itself produce, and
        raises from inside that same loop.

        A type change which changes the unit kind also resets the multiplier to zero,
        and the warning MNE emits for it is suppressed: that reset is the intended
        behaviour here and the row shows the resulting unit in the same repaint, because
        the metadata is re-read before the emission.
        """
        if value not in CH_TYPES:
            raise ValueError(
                f"The channel type must be one of {CH_TYPES}, got {value!r}."
            )
        rows = _sorted_rows(rows, len(self._rows))
        if not rows or not self._stream.connected:
            return
        unnamed = sorted(
            {
                self._rows[row].name
                for row in rows
                if self._rows[row].unit_kind not in _unit2human
            }
        )
        if unnamed:
            raise ValueError(
                f"The channels {unnamed} declare a physical unit the type write path "
                f"cannot name, so their type cannot be changed."
            )
        names = self._names(rows)
        try:
            self._stream.set_channel_types(
                dict.fromkeys(names, value), on_unit_change="ignore"
            )
        finally:
            self._read_stream()
        self._emit_metadata(rows[0], rows[-1])

    def set_unit(self, rows: Iterable[int], label: str) -> None:
        """Set the unit of ``rows`` from a human ``label``, i.e. its multiplier.

        Parameters
        ----------
        rows : iterable of int
            Display rows. An empty iterable is a silent no-op.
        label : str
            One of the labels :func:`unit_choices` offers for the channels' kind.

        Raises
        ------
        ValueError
            If ``label`` is not an offered label, if it belongs to a different unit kind
            than a channel of ``rows``, if a row is not a display row, or if a channel
            is absent from the stream.

        Notes
        -----
        Only the multiplier is written: MNE derives the unit kind from the channel type,
        so the Type control is what changes a kind. The check is enforced here and not
        only greyed out in the inspector, because the context menu reaches this method
        directly.

        The multiplier is written as an integer rather than as a human string, since the
        string path knows three labels while the integer path accepts every multiplier
        FIFF names.
        """
        rows = _sorted_rows(rows, len(self._rows))
        if not rows or not self._stream.connected:
            return
        if label not in _LABEL_TO_PAIR:
            raise ValueError(
                f"Unknown unit {label!r}; the offered units are {UNIT_LABELS}."
            )
        kind, mul = _LABEL_TO_PAIR[label]
        wrong = sorted(
            {self._rows[row].name for row in rows if self._rows[row].unit_kind != kind}
        )
        if wrong:
            raise ValueError(
                f"The unit {label!r} is not a unit of the channels {wrong}; change "
                f"their type first, as the unit kind follows the channel type."
            )
        names = self._names(rows)
        try:
            self._stream.set_channel_units(dict.fromkeys(names, mul))
        finally:
            self._read_stream()
        self._emit_metadata(rows[0], rows[-1])

    def rename(self, row: int, name: str) -> None:
        """Rename the single channel at ``row``.

        Parameters
        ----------
        row : int
            Display row index; renaming is a single-channel operation.
        name : str
            The new name, stripped of its surrounding whitespace.

        Raises
        ------
        ValueError
            If ``row`` is not a display row, or if ``name`` is blank, unprintable or
            already in use.

        Notes
        -----
        This is a trust boundary: ``rename_channels`` accepts both ``''`` and ``'  '``
        and leaves the channel nameless, and its ``allow_duplicates`` escape hatch
        mangles *both* colliding names and leaves ``info['bads']`` holding a name the
        info no longer has -- so it is never used. Renaming to the current name returns
        early, and ``info['bads']`` follows a rename on its own, thus there is nothing
        to remap.

        Printability is checked and not only blankness, because ``str.strip`` removes
        only what :meth:`str.isspace` matches: a zero-width space, a byte-order mark or
        an embedded newline all survive it, and a channel named with one paints nothing,
        cannot be searched for and is only reachable again through the context menu on a
        row which looks empty.
        """
        channel = self.channel(row)
        name = name.strip()
        if not name or not name.isprintable():
            raise ValueError(
                f"A channel name must be non-empty and printable, got {name!r}."
            )
        if name == channel.name:
            return
        if not self._stream.connected:
            return
        if name in set(self._stream.info.ch_names):
            raise ValueError(f"The channel name {name!r} is already in use.")
        try:
            self._stream.rename_channels({channel.name: name})
        finally:
            self._read_stream()
        self._emit_metadata(row, row)

    def rename_many(self, names: Mapping[int, str]) -> None:
        """Rename several channels at once, in one write.

        Parameters
        ----------
        names : dict of int to str
            New name per display row. A row already carrying its requested name is
            dropped, and an empty result is a no-op.

        Raises
        ------
        ValueError
            If a row is not a display row, if a name is blank or unprintable, or if the
            names the request would leave behind are not unique.

        Notes
        -----
        Renaming one channel at a time cannot express a **permutation**. The underlying
        operation refuses a target that is still held, so ``{'A': 'B', 'B': 'A'}``
        applied row by row fails on its first write, while the same mapping in one call
        succeeds -- the library resolves the whole set at once. A restore therefore has
        to group, or a saved configuration which merely reorders names loses the writes
        that collide, silently and one at a time.

        Uniqueness is checked against the names the request *leaves*, not against the
        names in use now, which is what allows a swap while still refusing a collision
        with a channel nobody is renaming. Everything else matches the single-channel
        path: names are stripped, blank and unprintable are refused,
        ``allow_duplicates`` is never used, and ``info['bads']`` follows a rename by
        itself.
        """
        rows = _sorted_rows(names, len(self._rows))
        if not rows or not self._stream.connected:
            return
        mapping: dict[str, str] = {}
        for row in rows:
            channel = self.channel(row)
            value = names[row].strip()
            if not value or not value.isprintable():
                raise ValueError(
                    f"A channel name must be non-empty and printable, got {value!r}."
                )
            if value != channel.name:
                mapping[channel.name] = value
        if not mapping:
            return
        current = list(self._stream.info["ch_names"])
        final = [mapping.get(name, name) for name in current]
        if len(set(final)) != len(final):
            raise ValueError("The requested channel names are not unique.")
        try:
            self._stream.rename_channels(mapping)
        finally:
            self._read_stream()
        self._emit_metadata(rows[0], rows[-1])

    def reset_metadata(self, rows: Iterable[int]) -> list[str]:
        """Restore the name, type, unit and bad state of ``rows`` to stream values.

        Parameters
        ----------
        rows : iterable of int
            Display rows. An empty iterable is a silent no-op.

        Returns
        -------
        skipped : list of str
            Current names of the channels whose acquisition *name* could not be restored
            because another channel holds it. Everything else was restored for them.

        Raises
        ------
        ValueError
            If a row is not a display row, or if a channel of ``rows`` is absent from
            the stream, i.e. the model is stale.

        Notes
        -----
        The write order is fixed: types, then units, then names, then bads. A type
        change resets the multiplier, thus the units follow the types; every mapping is
        keyed by the current name, thus the renames come last of the three; and the bads
        must be applied after the renames or they would hold names the info no longer
        has.

        A row whose original name is currently held by *another* channel is skipped for
        the rename step alone, the rest of its reset still applied, so that resetting a
        swapped pair of names cannot raise. Reset is the documented escape hatch out of
        a confusing metadata state, thus the skip is reported rather than silent: a
        caller which dropped it would leave the row named ``'b'`` while its own
        inspector still shows ``orig: a``.

        The visibility and the display order are presentation state and are left
        untouched.
        """
        rows = _sorted_rows(rows, len(self._rows))
        if not rows or not self._stream.connected:
            return []
        channels = [self._rows[row] for row in rows]
        self._names(rows)  # validated up front, as 'set_channel_types' half-applies
        taken = {c.name for c in self._rows} - {c.name for c in channels}
        skipped = [
            c.name for c in channels if c.name != c.orig.name and c.orig.name in taken
        ]
        try:
            types = {
                c.name: c.orig.ch_type for c in channels if c.ch_type != c.orig.ch_type
            }
            if types:
                self._stream.set_channel_types(types, on_unit_change="ignore")
            # written unconditionally rather than only on a difference: the type write
            # above has just reset the multiplier of every channel whose kind changed. A
            # kind which is not a physical quantity is excluded, as MNE refuses a
            # multiplier on it -- which is why 'Original' never records one.
            units = {
                c.name: c.orig.unit_mul
                for c in channels
                if c.orig.unit_kind != int(FIFF.FIFF_UNIT_NONE)
            }
            if units:
                self._stream.set_channel_units(units)
            names = {
                c.name: c.orig.name
                for c in channels
                if c.name != c.orig.name and c.orig.name not in taken
            }
            if names:
                self._stream.rename_channels(names)
            # keyed by the *restored* names, read straight off the info by acquisition
            # index: the cache still holds the pre-rename ones, and refreshing it here
            # only to read the names back would be a second full metadata read -- 95 µs
            # at 256 channels -- and a second place which writes the cache.
            info = self._stream.info
            restored = info.ch_names
            bads = set(info["bads"])
            for channel in channels:
                name = restored[channel.acq_index]
                if channel.orig.bad:
                    bads.add(name)
                else:
                    bads.discard(name)
            self._write_bads(bads)
        finally:
            self._read_stream()
        self._emit_metadata(rows[0], rows[-1])
        return skipped

    def refresh(self) -> None:
        """Re-read the stream after it was changed from outside the model.

        Notes
        -----
        The entry point for a configuration load, a reconnect, or an operation which
        changed the channel *set* such as adding a reference channel. A change of the
        channel count is structural: every channel is rebuilt, the order goes back to
        acquisition order and everything becomes visible again, because the acquisition
        indices a layout is built from would otherwise be stale -- which is exactly what
        the display's own layout guard refuses. A change which preserved the count is an
        ordinary metadata refresh and keeps the order and the visibility.

        The metadata signal is emitted **before** the layout one: a listener which owns
        both this model and a trace display validates a pushed layout against the
        channel count it last read, so pushing the grown layout first would have it
        refused against the stale count and leave the display stuck on the previous
        channel set. A shrunk set is safe either way, so metadata-first is correct
        unconditionally.
        """
        if not self._stream.connected:
            return
        if len(self._stream.info.ch_names) == len(self._rows):
            # no empty-model guard: the rows are empty only while the stream is
            # disconnected, which the check above already returned on, and a connected
            # stream never reports zero channels.
            self._read_stream()
            self._emit_metadata(0, len(self._rows) - 1)
            return
        self.beginResetModel()
        self._build()
        self.endResetModel()
        self.metadata_changed.emit()
        self.layout_changed.emit()

    # -- presentation order, mirrored by the trace display -----------------------------
    def order_by(self, kind: str) -> None:
        """Reorder every channel with a deterministic command.

        Parameters
        ----------
        kind : str
            One of ``'acquisition'``, ``'type'`` or ``'alphabetical'``.

        Raises
        ------
        ValueError
            If ``kind`` is not one of the three commands. Refusing it is what prevents
            an unknown command from silently keeping the previous order.

        Notes
        -----
        Every key is tie-broken by the acquisition index, so that the order inside a
        type group is stable and two channels whose names differ only by case -- or the
        ``Cz-0`` / ``Cz-1`` pair a duplicate name produces -- have a deterministic
        order. A type outside :data:`CH_TYPES` sorts last.
        """
        if kind == "acquisition":
            rows = sorted(self._rows, key=lambda c: c.acq_index)
        elif kind == "type":
            order = {ch_type: i for i, ch_type in enumerate(CH_TYPES)}
            rows = sorted(
                self._rows,
                key=lambda c: (order.get(c.ch_type, len(order)), c.acq_index),
            )
        elif kind == "alphabetical":
            rows = sorted(self._rows, key=lambda c: (c.name.casefold(), c.acq_index))
        else:
            raise ValueError(
                "The ordering must be one of 'acquisition', 'type' or 'alphabetical', "
                f"got {kind!r}."
            )
        self._apply_order(rows)

    def set_order(self, acq_indices: Sequence[int]) -> None:
        """Reorder the channels into an explicit acquisition-index order.

        Parameters
        ----------
        acq_indices : sequence of int
            The acquisition indices of every channel, in the display order they must
            take. Over an empty model, ``[]`` is the only permutation and is a no-op.

        Raises
        ------
        ValueError
            If ``acq_indices`` is not a permutation of the model's own acquisition
            indices, i.e. if it repeats one, omits one, or names one the model does not
            hold.

        Notes
        -----
        The counterpart of :meth:`presentation_order`, and the entry point of a restored
        configuration. A partial order is refused rather than tolerated: applying one
        would leave the omitted channels out of the row list altogether, hence out of
        :meth:`visible_acq_indices` -- undrawable *and* unreachable from the Channels
        page, with nothing on screen to explain the loss.

        Acquisition indices and never names, symmetrically with
        :meth:`presentation_order`: the model's ordering vocabulary is the acquisition
        index, and translating a saved name back to one belongs to whoever holds the
        saved names.
        """
        by_index = {channel.acq_index: channel for channel in self._rows}
        wanted = [int(index) for index in acq_indices]
        if sorted(wanted) != sorted(by_index):
            raise ValueError(
                f"The display order must be a permutation of the {len(by_index)} "
                f"acquisition indices of the model, got {wanted}."
            )
        self._apply_order([by_index[index] for index in wanted])

    def acquisition_names(self) -> list[str]:
        """Return the names the stream declared, in acquisition order.

        Returns
        -------
        names : list of str
            ``Channel.orig.name`` of every channel, ordered by acquisition index.

        Notes
        -----
        The availability contract of a saved configuration and the key set of every
        channel-keyed block it carries, thus spelled once, here, where the acquisition
        baseline lives. The *original* names and never the edited ones: a contract
        holding a renamed channel could match no stream, and the acquisition order and
        never the presentation one: a contract which reshuffled between two saves of one
        workspace would compare unequal to itself.
        """
        return [
            channel.orig.name
            for channel in sorted(self._rows, key=lambda c: c.acq_index)
        ]

    def _apply_order(self, rows: list[Channel]) -> None:
        """Replace the row order, remapping the persistent indexes.

        Notes
        -----
        The remap is what makes the selection follow the *channels* across a reorder
        rather than staying on the row numbers. The channels themselves are reordered
        and never rebuilt, so no metadata and no visibility can change here.

        The position lookup is built only when there is something to remap: it is a dict
        over every channel and, with nothing selected, it was 60% of the cost of a
        reorder.
        """
        self.layoutAboutToBeChanged.emit()
        persistent = self.persistentIndexList()
        previous = self._rows
        self._rows = rows
        if persistent:
            position = {id(channel): row for row, channel in enumerate(rows)}
            for index in persistent:
                channel = previous[index.row()]
                self.changePersistentIndex(index, self.index(position[id(channel)], 0))
        self.layoutChanged.emit()
        self.layout_changed.emit()

    def presentation_order(self) -> list[int]:
        """Return the acquisition indices of the channels, in display order."""
        return [channel.acq_index for channel in self._rows]

    def hidden_channels(self) -> list[int]:
        """Return the acquisition indices of the channels which are hidden."""
        return [channel.acq_index for channel in self._rows if not channel.visible]

    def visible_acq_indices(self) -> list[int]:
        """Return the acquisition indices of the visible channels, in display order.

        Returns
        -------
        rows : list of int
            What the trace display draws, in the order it draws it, and what it passes
            to ``get_data`` as its picks.

        Notes
        -----
        The single translation site between the presentation order and what the display
        stacks. The result is order-preserving, duplicate-free and holds acquisition
        indices -- never row numbers -- which is exactly what the display's own guard
        checks, so the two halves of the contract agree by construction.
        """
        return [channel.acq_index for channel in self._rows if channel.visible]

    # -- emission ----------------------------------------------------------------------
    def _emit_layout(self, lo: int, hi: int) -> None:
        """Emit one ``dataChanged`` spanning ``lo``..``hi``, then ``layout_changed``."""
        self.dataChanged.emit(self.index(lo, 0), self.index(hi, 0))
        self.layout_changed.emit()

    def _emit_metadata(self, lo: int, hi: int) -> None:
        """Emit one ``dataChanged`` over ``lo``..``hi``, then ``metadata_changed``."""
        self.dataChanged.emit(self.index(lo, 0), self.index(hi, 0))
        self.metadata_changed.emit()

    def _names(self, rows: list[int]) -> list[str]:
        """Return the current names of ``rows``, checked against the stream.

        Raises
        ------
        ValueError
            If a name is absent from the stream, i.e. the model is stale because the
            stream was edited or reconnected behind its back.
        """
        known = set(self._stream.info.ch_names)
        names = [self._rows[row].name for row in rows]
        missing = sorted(set(names) - known)
        if missing:
            raise ValueError(
                f"The channels {missing} are absent from the stream; the channel model "
                f"is stale and must be refreshed before it is edited."
            )
        return names

    def _write_bads(self, bads: set[str]) -> None:
        """Write ``bads`` to ``stream.info['bads']``, without emitting.

        Notes
        -----
        The single writer of the bad list, so that the bulk edit and the reset share one
        error behaviour rather than one surfacing MNE's raw message and the other the
        model's. Filtering through ``info.ch_names`` keeps the list in acquisition
        order.
        """
        info = self._stream.info
        info["bads"] = [name for name in info.ch_names if name in bads]


def _sorted_rows(rows: Iterable[int], count: int) -> list[int]:
    """Return ``rows`` as ascending, deduplicated, in-range display rows.

    Parameters
    ----------
    rows : iterable of int
        Display rows, in any order, possibly with duplicates.
    count : int
        Number of display rows, i.e. the exclusive upper bound.

    Returns
    -------
    rows : list of int
        The rows, ascending and unique, so that a mutator can span its ``dataChanged``
        from the first to the last and write every channel exactly once.

    Raises
    ------
    ValueError
        If a row is outside ``[0, count)``. The single funnel of every bulk mutator, and
        the lower bound matters as much as the upper one: Python list indexing wraps, so
        ``-1`` would edit the *last* channel instead of raising, and the emission would
        span an invalid index. These rows come from a selection, from a context menu
        and, from a later phase, from a restored configuration file.
    """
    checked = sorted({int(row) for row in rows})
    if checked and not (0 <= checked[0] and checked[-1] < count):
        raise ValueError(
            f"Every display row must be in [0, {count}), got "
            f"{checked[0]}..{checked[-1]}."
        )
    return checked
