from __future__ import annotations

import pkgutil
import warnings
from typing import TYPE_CHECKING

import pytest
from mne._fiff.constants import FIFF, _ch_unit_mul_named
from qtpy.QtCore import QItemSelectionModel, Qt

import mne_lsl.viewer.controller
from mne_lsl.stream import StreamLSL
from mne_lsl.viewer.controller import (
    CH_TYPES,
    UNIT_LABELS,
    BadRole,
    ChannelModel,
    NameRole,
    TypeRole,
    UnitRole,
    VisibleRole,
    _channels,
    _events,
    _model,
    _processing,
    unit_choices,
    unit_label,
    unit_pair,
)
from mne_lsl.viewer.controller._model import _KINDS, _MULTIPLIERS, _NO_UNIT
from mne_lsl.viewer.display import TraceDisplay
from mne_lsl.viewer.theme import trace_color

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from qtpy.QtWidgets import QApplication

    from tests.viewer.controller.conftest import Emissions

_V = int(FIFF.FIFF_UNIT_V)
_NONE = int(FIFF.FIFF_UNIT_NONE)

# Names no module of 'controller/' may reach: the trace display it must stay independent
# of, and the low-level LSL layer which only 'backend/' may name.
_FORBIDDEN = ("display", "TraceDisplay", "lsl", "StreamLSL")
# Every module of 'controller/', for the import-rule scan. The set is asserted against
# 'pkgutil' below, so a new module cannot silently escape the check.
_MODULES = {
    "_channels": _channels,
    "_events": _events,
    "_model": _model,
    "_processing": _processing,
}


def _row_of(model: ChannelModel, ch_type: str) -> int:
    """Return the display row of the first channel of ``ch_type``."""
    rows = (
        row for row in range(model.rowCount()) if model.channel(row).ch_type == ch_type
    )
    return next(rows)


def _stream_types(stream: StreamLSL) -> list[str]:
    """Return the channel types the stream itself reports, in acquisition order."""
    return list(stream.get_channel_types(picks=list(range(stream.info["nchan"]))))


def _stream_units(stream: StreamLSL) -> list[tuple[int, int]]:
    """Return the ``(kind, multiplier)`` pairs the stream itself reports."""
    return stream.get_channel_units(picks=list(range(stream.info["nchan"])))


# -- index spaces and the translation site ---------------------------------------------
def test_construction_is_acquisition_order(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    rows_of: Callable[[ChannelModel], list],
) -> None:
    """Test that a fresh model is in acquisition order with every channel visible."""
    assert model.n_channels == 8
    assert model.rowCount() == 8
    assert model.presentation_order() == list(range(8))
    assert rows_of(model) == list(mixed_stream.info.ch_names)
    assert all(model.channel(row).visible for row in range(8))
    assert model.hidden_channels() == []


def test_visible_acq_indices_is_acquisition_order_initially(
    model: ChannelModel,
) -> None:
    """Test that everything is drawn, in acquisition order, before any edit.

    The starting state of the whole layout contract, and what makes the two index spaces
    coincide at t=0 -- which is exactly why the tests below deliberately reorder or hide
    before asserting anything about it.
    """
    assert model.visible_acq_indices() == list(range(8))


def test_visible_acq_indices_skips_hidden(model: ChannelModel) -> None:
    """Test that a hidden channel leaves the layout."""
    model.set_visible([1, 3], False)
    assert model.visible_acq_indices() == [0, 2, 4, 5, 6, 7]


def test_visible_acq_indices_follows_the_presentation_order(
    model: ChannelModel,
) -> None:
    """Test that the layout is in display order, not in acquisition order."""
    model.order_by("alphabetical")
    expected = [model.channel(row).acq_index for row in range(model.rowCount())]
    assert model.visible_acq_indices() == expected
    assert expected != list(range(8))  # the reorder really moved something


def test_visible_acq_indices_returns_acquisition_not_presentation(
    model: ChannelModel,
) -> None:
    """Test that the layout holds acquisition indices and never row numbers.

    Returning the loop counter instead of the channel's own index is the single most
    dangerous confusion of this phase: it looks right for a fresh model, because the two
    spaces coincide, and silently draws the wrong channels after any reorder or hide.
    """
    model.order_by("alphabetical")
    model.set_visible([0], False)
    layout = model.visible_acq_indices()
    # the row numbers of the visible rows are 1..7; the acquisition indices are not.
    assert layout != list(range(1, model.rowCount()))
    assert layout == [
        model.channel(row).acq_index for row in range(1, model.rowCount())
    ]


def test_visible_acq_indices_is_a_valid_layout(
    app: QApplication, model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that the display accepts what the model produces, in all three regimes.

    This is the one Phase D test of the cross-widget contract: the display refuses a
    layout holding an out-of-range index, and the model's guarantee is that it never
    produces one.
    """
    widget = TraceDisplay(mixed_stream)
    try:
        widget.set_channel_layout(model.visible_acq_indices())
        assert widget.n_rows == 8
        model.set_visible([1, 3], False)
        widget.set_channel_layout(model.visible_acq_indices())
        assert widget.n_rows == 6
        model.set_visible(range(model.rowCount()), False)
        assert model.visible_acq_indices() == []
        widget.set_channel_layout(model.visible_acq_indices())
        assert widget.n_rows == 0
    finally:
        widget.stop()
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_presentation_order_includes_hidden(model: ChannelModel) -> None:
    """Test that hiding a channel does not remove it from the presentation order."""
    before = model.presentation_order()
    model.set_visible([2, 5], False)
    assert model.presentation_order() == before
    assert len(before) == 8


def test_hidden_channels(model: ChannelModel) -> None:
    """Test that the hidden channels are the complement, in display order."""
    model.order_by("alphabetical")
    model.set_visible([0, 2], False)
    hidden = [model.channel(row).acq_index for row in (0, 2)]
    assert model.hidden_channels() == hidden
    assert set(hidden) | set(model.visible_acq_indices()) == set(range(8))


def test_n_channels_is_invariant(model: ChannelModel) -> None:
    """Test that neither hiding nor reordering changes the channel count."""
    model.set_visible([0, 1, 2], False)
    model.order_by("type")
    assert model.n_channels == 8
    assert model.rowCount() == 8


def test_row_count_is_zero_for_a_valid_parent(model: ChannelModel) -> None:
    """Test that the flat list reports no child rows, as a list model must."""
    assert model.rowCount(model.index(0, 0)) == 0


# -- ordering --------------------------------------------------------------------------
def test_order_by_acquisition(model: ChannelModel) -> None:
    """Test that the acquisition command restores the acquisition order."""
    model.order_by("alphabetical")
    model.order_by("acquisition")
    assert model.presentation_order() == list(range(8))


def test_order_by_type_groups_and_is_stable_inside_a_type(
    model: ChannelModel,
) -> None:
    """Test that channels group by type, acquisition-ordered inside each group."""
    model.order_by("alphabetical")
    model.order_by("type")
    order = {ch_type: index for index, ch_type in enumerate(CH_TYPES)}
    keys = [order[model.channel(row).ch_type] for row in range(8)]
    assert keys == sorted(keys)
    eeg = [
        model.channel(row).acq_index
        for row in range(8)
        if model.channel(row).ch_type == "eeg"
    ]
    assert eeg == sorted(eeg)


def test_order_by_type_puts_an_unknown_type_last(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a type outside the offered list sorts last rather than first."""
    # written through the stream: the model refuses a type it does not offer, and this
    # pins what happens when a *stream* declares one anyway.
    mixed_stream.set_channel_types({"Fp1": "seeg"}, on_unit_change="ignore")
    model.refresh()
    model.order_by("type")
    assert model.channel(7).ch_type == "seeg"


def test_order_by_alphabetical_is_case_insensitive(
    model: ChannelModel, rows_of: Callable[[ChannelModel], list]
) -> None:
    """Test that the alphabetical order ignores the case.

    A case-sensitive sort puts every upper-case name before every lower-case one, which
    reads as no order at all for a mixed-case montage.
    """
    model.rename(0, "aFp1")
    model.order_by("alphabetical")
    names = rows_of(model)
    assert names == sorted(names, key=str.casefold)
    assert names != sorted(names)  # a case-sensitive sort would differ


def test_order_by_alphabetical_is_deterministic_for_duplicate_names(
    app: QApplication,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the deduplicated names of a duplicate pair keep a stable order.

    Two channels published under one name reach the info as 'Cz-0' / 'Cz-1'. The pair is
    already in that order before the sort and 'sorted' is stable, so this pins the
    *outcome* rather than the tie-break; the tie-break is pinned by the test below,
    where the previous sort left the tied pair in the opposite order.
    """
    with pytest.warns(RuntimeWarning, match="Channel names are not unique"):
        stream, _ = lsl_stream(
            n_channels=4, ch_names=["Cz", "Cz", "Fp1", "STI"], n_stim=1
        )
    model = ChannelModel(stream)
    try:
        assert [model.channel(row).name for row in range(4)] == [
            "Cz-0",
            "Cz-1",
            "Fp1",
            "STI",
        ]
        model.order_by("alphabetical")
        names = [model.channel(row).name for row in range(4)]
        assert names.index("Cz-0") < names.index("Cz-1")
    finally:
        model.deleteLater()
        app.processEvents()


def test_order_by_alphabetical_breaks_a_tie_on_the_acquisition_index(
    model: ChannelModel, rows_of: Callable[[ChannelModel], list]
) -> None:
    """Test that two names equal up to their case land in acquisition order.

    Without the tie-break the sort is merely stable, so the pair keeps whichever order
    the *previous* command left it in -- the same two channels would come out in a
    different order depending on how the user got there.
    """
    model.rename(0, "zz")
    model.rename(5, "aa")
    model.order_by("alphabetical")
    names = rows_of(model)
    assert names.index("aa") < names.index("zz")  # channel 5 now precedes channel 0
    model.rename(names.index("aa"), "ZZ")  # a tie with 'zz', still ahead of it
    model.order_by("alphabetical")
    tied = [
        model.channel(row).acq_index
        for row in range(8)
        if model.channel(row).name.casefold() == "zz"
    ]
    assert tied == [0, 5]  # acquisition order, not the order the previous sort left
    assert rows_of(model)[-2:] == ["zz", "ZZ"]


def test_order_by_rejects_an_unknown_kind(model: ChannelModel) -> None:
    """Test that an unknown ordering raises instead of silently keeping the order."""
    model.order_by("alphabetical")
    before = model.presentation_order()
    with pytest.raises(ValueError, match="ordering must be one of"):
        model.order_by("montage")
    assert model.presentation_order() == before


def test_order_by_preserves_selection_identity(model: ChannelModel) -> None:
    """Test that the same *channels* stay selected across every ordering command."""
    selection = QItemSelectionModel(model)
    for row in (0, 5, 6):
        selection.select(model.index(row, 0), QItemSelectionModel.SelectionFlag.Select)
    expected = {model.channel(row).acq_index for row in (0, 5, 6)}
    for kind in ("alphabetical", "type", "acquisition"):
        model.order_by(kind)
        selected = {
            model.channel(index.row()).acq_index
            for index in selection.selectedIndexes()
        }
        assert selected == expected, kind


def test_order_by_preserves_visibility_and_metadata(model: ChannelModel) -> None:
    """Test that a reorder moves channels and rebuilds none of them."""
    model.set_visible([2], False)
    model.rename(0, "Renamed")
    hidden = model.channel(2).acq_index
    model.order_by("alphabetical")
    by_acq = {model.channel(row).acq_index: model.channel(row) for row in range(8)}
    assert by_acq[hidden].visible is False
    assert by_acq[0].name == "Renamed"
    assert sum(not channel.visible for channel in by_acq.values()) == 1


def test_order_by_emits_layout_changed_once(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that a reorder emits one layout signal and no metadata signal."""
    log = emissions(model)
    model.order_by("type")
    assert (log.layout, log.metadata) == (1, 0)
    assert log.layout_qt == 1
    assert (
        log.data == []
    )  # 'layoutChanged' is the reorder notification, not 'dataChanged'


def test_order_by_does_not_renumber_the_acquisition_index(model: ChannelModel) -> None:
    """Test that a reorder moves the channels and renumbers none of them.

    The acquisition index is the channel identity and seeds its trace colour, so a drift
    toward renumbering it per row -- which looks right for a fresh model, where the two
    spaces coincide -- would recolour the whole display on every reorder. Asserted on
    the mapping itself and not on the colour, as comparing 'trace_color' to itself is
    true of any pure function.
    """
    before = {model.channel(row).name: model.channel(row).acq_index for row in range(8)}
    assert sorted(before.values()) == list(range(8))
    for kind in ("alphabetical", "type", "acquisition"):
        model.order_by(kind)
        after = {
            model.channel(row).name: model.channel(row).acq_index for row in range(8)
        }
        assert after == before, kind
    # the colour is a function of that index alone, thus pinning the index pins the pen
    assert trace_color(before["Fp1"], "dark") == trace_color(0, "dark")


# -- an explicit order, i.e. a restored one --------------------------------------------
def test_set_order_applies_a_permutation(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that an explicit order is applied verbatim, announced once, and remapped.

    Four properties over one reorder, because they are one behaviour and one stream: the
    requested order is what the model holds, the display's own layout follows it, the
    move is announced with a single layout notification and no metadata one, and a
    persistent index follows its channel rather than its row number.

    A 'set_order' which sorted, ignored its argument or forwarded to
    'order_by("acquisition")' would silently discard the order of every configuration
    loaded. A dropped emission would leave the display drawing the previous order for
    the rest of the session. A bypassed remap would leave the highlight on the row
    numbers, so the inspector would edit a channel other than the one shown as selected.
    """
    wanted = [3, 0, 7, 1, 6, 2, 5, 4]
    selection = QItemSelectionModel(model)
    for row in (0, 5, 6):
        selection.select(model.index(row, 0), QItemSelectionModel.SelectionFlag.Select)
    expected = {model.channel(row).acq_index for row in (0, 5, 6)}
    log = emissions(model)
    model.set_order(wanted)
    assert model.presentation_order() == wanted
    assert model.visible_acq_indices() == wanted
    assert (log.layout, log.metadata) == (1, 0)
    assert log.layout_qt == 1
    selected = {
        model.channel(index.row()).acq_index for index in selection.selectedIndexes()
    }
    assert selected == expected
    # a hidden channel drops out of the display layout but keeps its place in the order.
    model.set_visible([2], False)
    assert model.presentation_order() == wanted
    assert model.visible_acq_indices() == [3, 0, 1, 6, 2, 5, 4]


def test_set_order_rejects_anything_but_a_permutation(model: ChannelModel) -> None:
    """Test that a non-permutation raises and leaves the order untouched.

    Tolerating a partial or duplicated order would drop the omitted channels out of the
    row list, hence out of 'visible_acq_indices' -- undrawable and unreachable from the
    page, with nothing on screen to explain the loss. An unknown index has to be refused
    here as well, or it raises 'KeyError' from inside whichever Qt slot ran the restore.

    The four cases share one test rather than a parametrization on purpose: the model
    fixture is function-scoped, so a parametrization would pay one stream connection per
    case, and every case leaves the model unchanged by construction.
    """
    before = model.presentation_order()
    orders = (
        [3, 0, 7, 1, 6, 2, 5],  # partial
        [3, 3, 0, 7, 1, 6, 2, 5],  # a duplicate, hence one channel omitted
        [3, 0, 7, 1, 6, 2, 5, 99],  # an index the model does not hold
        [],  # empty over a non-empty model
    )
    for order in orders:
        with pytest.raises(ValueError, match="must be a permutation"):
            model.set_order(order)
        assert model.presentation_order() == before, order


def test_set_order_over_a_disconnected_model() -> None:
    """Test that an empty model accepts the empty order and refuses anything else.

    A model built over a stream which is not connected holds no row, and the empty list
    is the only permutation of nothing: refusing it would make the check divide by the
    row count or index its first element. No connection is made here at all.
    """
    empty = ChannelModel(StreamLSL(2.0, name="absent", stype="eeg", source_id="absent"))
    empty.set_order([])
    assert empty.presentation_order() == []
    assert empty.acquisition_names() == []
    with pytest.raises(ValueError, match="must be a permutation"):
        empty.set_order([0])


def test_acquisition_names_is_acquisition_order_and_original(
    model: ChannelModel,
) -> None:
    """Test that the contract holds the device's names, in acquisition order.

    Reading 'Channel.name' instead of 'Channel.orig.name' puts the *edited* names into
    the availability contract, so the configuration would match no stream and be
    permanently unavailable; reading the presentation order would make the contract
    reshuffle between two saves of one unchanged workspace.
    """
    declared = model.acquisition_names()
    assert declared == [model.channel(row).name for row in range(model.rowCount())]
    model.rename(0, "Renamed")
    model.set_order([3, 0, 7, 1, 6, 2, 5, 4])
    assert model.acquisition_names() == declared


# -- visibility ------------------------------------------------------------------------
def test_set_visible_bulk_emits_one_data_changed(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that a bulk hide emits one spanning 'dataChanged' and one layout signal.

    A per-row emission makes the page handler run once per row, which is the measured
    ~95 ms 'hide all' regression at 256 channels against ~2 ms for one span.
    """
    log = emissions(model)
    model.set_visible(range(8), False)
    assert log.data == [(0, 7)]
    assert (log.layout, log.metadata) == (1, 0)


def test_set_visible_empty_rows_is_silent(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that an empty selection emits nothing at all.

    An empty layout push would make the display rebuild its layout for nothing.
    """
    log = emissions(model)
    model.set_visible([], False)
    assert (log.data, log.layout, log.metadata) == ([], 0, 0)


def test_set_visible_leaves_other_rows_alone(model: ChannelModel) -> None:
    """Test that only the requested rows change visibility."""
    model.set_visible([1, 2], False)
    assert [model.channel(row).visible for row in range(8)] == [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_set_visible_does_not_touch_bad(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that hiding a channel does not mark it bad; the two are independent."""
    model.set_visible(range(8), False)
    assert mixed_stream.info["bads"] == []
    assert not any(model.channel(row).bad for row in range(8))


def test_set_data_visible_role_toggles_and_emits_layout(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test the eye-glyph path: it toggles the row and pushes a new layout.

    Without the layout signal the traces silently stop following the eye.
    """
    log = emissions(model)
    assert model.setData(model.index(2, 0), False, VisibleRole) is True
    assert model.channel(2).visible is False
    assert model.data(model.index(2, 0), VisibleRole) is False
    assert log.data == [(2, 2)]
    assert (log.layout, log.metadata) == (1, 0)
    assert 2 not in model.visible_acq_indices()


def test_set_data_bad_role_routes_through_set_bad(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a single-row bad edit reaches the stream, not only the cache."""
    assert model.setData(model.index(1, 0), True, BadRole) is True
    assert mixed_stream.info["bads"] == ["Fp2"]
    assert model.data(model.index(1, 0), BadRole) is True


def test_set_data_rejects_an_unknown_role(model: ChannelModel) -> None:
    """Test that an unsupported role is refused and mutates nothing."""
    before = model.channel(0).name
    assert model.setData(model.index(0, 0), "x", NameRole) is False
    assert model.setData(model.index(0, 0), "x", Qt.ItemDataRole.EditRole) is False
    assert model.channel(0).name == before


def test_set_data_on_an_invalid_index(model: ChannelModel) -> None:
    """Test that an invalid index is refused rather than indexing the row list."""
    assert model.setData(model.index(999, 0), True, VisibleRole) is False
    assert model.data(model.index(999, 0), NameRole) is None


def test_data_unhandled_role_is_none(model: ChannelModel) -> None:
    """Test that a role the model does not serve returns nothing."""
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DecorationRole) is None


def test_data_display_and_tooltip_roles(model: ChannelModel) -> None:
    """Test that the display role is the name and the tooltip is the acquisition value.

    The display role is what the view's keyboard search and the accessible name read.
    """
    index = model.index(5, 0)
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "ECG"
    assert model.data(index, NameRole) == "ECG"
    assert model.data(index, TypeRole) == "ecg"
    assert model.data(index, UnitRole) == "mV"
    tooltip = model.data(index, Qt.ItemDataRole.ToolTipRole)
    assert tooltip == "acquisition value: ECG ecg mV good"
    assert model.channel(5).original == "ECG ecg mV good"


def test_flags_are_neither_editable_nor_draggable(model: ChannelModel) -> None:
    """Test that rows are selectable only: editing and drag-reorder were removed."""
    flags = model.flags(model.index(0, 0))
    assert flags & Qt.ItemFlag.ItemIsEnabled
    assert flags & Qt.ItemFlag.ItemIsSelectable
    assert not flags & Qt.ItemFlag.ItemIsEditable
    assert not flags & Qt.ItemFlag.ItemIsDragEnabled
    assert not flags & Qt.ItemFlag.ItemIsDropEnabled
    assert model.flags(model.index(999, 0)) == Qt.ItemFlag.NoItemFlags


def test_hide_all_then_visible_acq_indices_is_empty(model: ChannelModel) -> None:
    """Test that hiding everything yields an empty layout rather than a stale one."""
    model.set_visible(range(8), False)
    assert model.visible_acq_indices() == []
    assert model.hidden_channels() == list(range(8))


# -- metadata writes -------------------------------------------------------------------
def test_set_type_writes_the_stream(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that a type edit reaches the stream and signals a *metadata* change.

    Which of the two coarse signals is emitted is the whole cross-widget contract: on
    the layout signal the display rebuilds its stacking, on the metadata one it re-reads
    names, types and units. A type edit which announced itself as a layout change would
    leave the trace colour, the per-type gain and the axis label stale for the life of
    the stream.
    """
    log = emissions(model)
    model.set_type([0, 1], "eog")
    assert _stream_types(mixed_stream)[:2] == ["eog", "eog"]
    assert [model.channel(row).ch_type for row in (0, 1)] == ["eog", "eog"]
    assert log.data == [(0, 1)]
    assert (log.layout, log.metadata) == (0, 1)


def test_set_type_does_not_warn(model: ChannelModel) -> None:
    """Test that a kind-changing bulk type edit is silent.

    MNE warns when a type change changes the unit, at default verbosity and through the
    mne-lsl wrapper. The viewer does it deliberately and shows the resulting unit in the
    row, so the warning is noise -- and warnings are errors in this suite, which would
    fail every type-change test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.set_type([0, 1, 2], "misc")  # Volts -> not applicable


def test_set_type_resets_the_unit_multiplier_and_the_row_shows_it(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that the multiplier reset by a type change is visible in the same read.

    Re-reading the stream before the write instead of after would leave the row showing
    the previous unit until the next unrelated edit.
    """
    assert model.channel(0).unit == "µV"
    model.set_type([0], "misc")
    assert model.channel(0).unit_mul == 0
    assert model.channel(0).unit_kind == _NONE
    assert model.data(model.index(0, 0), UnitRole) == _NO_UNIT
    assert _stream_units(mixed_stream)[0] == (_NONE, 0)


def test_set_type_rejects_an_unknown_type(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a type outside the offered list is refused before any write."""
    with pytest.raises(ValueError, match="channel type must be one of"):
        model.set_type([0], "banana")
    assert _stream_types(mixed_stream)[0] == "eeg"


def test_set_type_validates_names_before_writing(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a stale model raises without half-applying the edit.

    'set_channel_types' mutates the valid channels of its mapping and only *then* raises
    on the invalid one, so without the up-front check one channel would be silently
    rewritten by a failing call.
    """
    mixed_stream.rename_channels({"Fp1": "Renamed"})  # behind the model's back
    with pytest.raises(ValueError, match="absent from the stream"):
        model.set_type([0, 1], "eog")
    assert _stream_types(mixed_stream)[:2] == ["eeg", "eeg"]


def test_set_type_empty_rows_is_silent(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that an empty selection performs no write and emits nothing."""
    log = emissions(model)
    model.set_type([], "eog")
    assert _stream_types(mixed_stream)[0] == "eeg"
    assert (log.data, log.layout, log.metadata) == ([], 0, 0)


def test_set_type_validates_the_type_before_the_empty_shortcut(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that an unknown type is refused even with nothing selected.

    Returning early on an empty row list before validating makes the guard exist only
    while a channel happens to be selected, so 'set_type([], "banana")' is accepted
    silently -- and a caller which relies on the refusal never hears about the typo.
    """
    log = emissions(model)
    with pytest.raises(ValueError, match="channel type must be one of"):
        model.set_type([], "banana")
    assert (log.data, log.layout, log.metadata) == ([], 0, 0)


def test_set_type_refuses_a_channel_whose_unit_has_no_name(
    app: QApplication,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a channel whose unit kind MNE cannot name is refused before the write.

    'set_channel_types' looks the *current* unit up in a table missing two of the kinds
    it can itself produce, and raises from inside its mutation loop -- after the earlier
    channels of the same mapping have already been rewritten. A stream declaring a 'csd'
    channel reaches that state, and the viewer labels its unit 'V/m²', so it is
    displayable but not re-typeable; the check is what keeps the edit all-or-nothing
    instead of rewriting the healthy channel beside it.
    """
    stream, _ = lsl_stream(
        n_channels=3,
        n_stim=1,
        ch_names=["CSD", "EEG", "STI"],
        ch_types=["csd", "eeg", "stim"],
        ch_units=["uv", "uv", "none"],
    )
    model = ChannelModel(stream)
    try:
        assert model.channel(0).unit == "V/m²"
        with pytest.raises(ValueError, match="cannot name"):
            model.set_type([0, 1], "eog")
        assert _stream_types(stream) == ["csd", "eeg", "stim"]
        assert model.channel(1).ch_type == "eeg"  # not half-applied
        model.set_type([1], "eog")  # the healthy channel is still editable
        assert _stream_types(stream)[1] == "eog"
    finally:
        model.deleteLater()
        app.processEvents()


def test_set_unit_writes_the_multiplier(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that a unit edit reaches the stream and signals a *metadata* change.

    Same contract as the type edit: the display's amplitude gain is unit-aware and is
    re-read on the metadata signal only, so announcing a unit change as a layout change
    would leave every trace drawn at the previous multiplier's scale.
    """
    log = emissions(model)
    model.set_unit([0, 1], "mV")
    assert _stream_units(mixed_stream)[:2] == [(_V, -3), (_V, -3)]
    assert model.channel(0).unit == "mV"
    assert log.data == [(0, 1)]
    assert (log.layout, log.metadata) == (0, 1)


def test_set_unit_uses_the_integer_path(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a multiplier outside the three human strings is writable.

    The human-string path knows only volts, millivolts and microvolts; the integer path
    accepts every multiplier FIFF names, which is what makes the ladder writable.
    """
    model.set_unit([0], "10 mV")
    assert _stream_units(mixed_stream)[0] == (_V, -2)
    model.set_unit([0], "100 mV")
    assert _stream_units(mixed_stream)[0] == (_V, -1)


def test_set_unit_rejects_a_kind_change(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a unit of another kind is refused, pointing at the Type control.

    The check lives in the model and not only in the greyed-out combo, because the
    context menu reaches this method directly.
    """
    misc = _row_of(model, "misc")
    with pytest.raises(ValueError, match="change their type first"):
        model.set_unit([misc], "µV")
    assert _stream_units(mixed_stream)[misc] == (_NONE, 0)


def test_set_unit_on_a_stim_channel_is_allowed(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a stim channel's multiplier is writable.

    A stim channel's kind is Volts, an MNE quirk, so it takes the Volt ladder like any
    other Volt channel rather than a special-cased list.
    """
    stim = _row_of(model, "stim")
    assert model.channel(stim).unit_kind == _V
    model.set_unit([stim], "µV")
    assert _stream_units(mixed_stream)[stim] == (_V, -6)


def test_set_unit_rejects_an_unknown_label(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that an unknown label raises rather than silently writing nothing."""
    for label in ("banana", _NO_UNIT, "100 µV"):
        with pytest.raises(ValueError, match="Unknown unit"):
            model.set_unit([0], label)
    assert _stream_units(mixed_stream)[0] == (_V, -6)


def test_set_unit_empty_rows_is_silent(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that an empty selection is a no-op, even for an unknown label."""
    log = emissions(model)
    model.set_unit([], "banana")
    assert (log.data, log.metadata) == ([], 0)


def test_bulk_mutators_reject_an_out_of_range_row(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a row outside the display range is refused before anything is written.

    The lower bound matters as much as the upper one: Python list indexing wraps, so
    without it row -1 hides, renames, re-types, re-units or resets the *last* channel
    instead of raising, and the emission spans an index the view rejects. Rows reach
    these methods from a selection, from a context menu and, from a later phase, from a
    restored configuration file.

    Looped rather than parametrized: every case is refused before any write, so a single
    stream covers them all.
    """
    calls = (
        lambda row: model.set_visible([row], False),
        lambda row: model.set_bad([row], True),
        lambda row: model.set_type([row], "eog"),
        lambda row: model.set_unit([row], "mV"),
        lambda row: model.reset_metadata([row]),
        lambda row: model.channel(row),
        lambda row: model.rename(row, "Renamed"),
    )
    for row in (-1, -8, 8, 999):
        for call in calls:
            with pytest.raises(ValueError, match="display row must be in"):
                call(row)
    # the last channel, which a wrapped -1 would have hit, is untouched in every field
    channel = model.channel(7)
    assert (channel.name, channel.ch_type, channel.visible) == ("MISC", "misc", True)
    assert mixed_stream.info["bads"] == []
    assert list(mixed_stream.info.ch_names)[7] == "MISC"


def test_bulk_mutators_reject_a_mixed_valid_and_invalid_row_list(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that one bad row refuses the edit rather than applying the good rows."""
    with pytest.raises(ValueError, match="display row must be in"):
        model.set_visible([0, 1, 99], False)
    assert model.visible_acq_indices() == list(range(8))
    with pytest.raises(ValueError, match="display row must be in"):
        model.set_bad([0, -1], True)
    assert mixed_stream.info["bads"] == []


def test_set_bad_writes_info_bads(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that a bad edit writes the stream's bad list, the single source of truth.

    A cache-only write would leave the trace display drawing the channel with its normal
    pen and no 'X ' axis prefix, since it reads the info.
    """
    log = emissions(model)
    model.set_bad([1, 3], True)
    assert set(mixed_stream.info["bads"]) == {"Fp2", "Pz"}
    assert [model.channel(row).bad for row in (1, 3)] == [True, True]
    assert log.data == [(1, 3)]
    assert (log.layout, log.metadata) == (0, 1)


def test_set_bad_preserves_unrelated_bads(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a bad channel outside the edited rows stays bad."""
    model.set_bad([0], True)
    model.set_bad([4], True)
    assert set(mixed_stream.info["bads"]) == {"Fp1", "EOG"}
    model.set_bad([4], False)
    assert mixed_stream.info["bads"] == ["Fp1"]


def test_set_bad_does_not_hide(model: ChannelModel) -> None:
    """Test that marking a channel bad leaves it visible; the two are independent."""
    model.set_bad(range(8), True)
    assert model.visible_acq_indices() == list(range(8))


def test_set_bad_false_clears(model: ChannelModel, mixed_stream: StreamLSL) -> None:
    """Test that unmarking removes the channel from the bad list."""
    model.set_bad([2], True)
    model.set_bad([2], False)
    assert mixed_stream.info["bads"] == []
    assert model.channel(2).bad is False


def test_set_bad_empty_rows_is_silent(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that an empty selection performs no bad write."""
    log = emissions(model)
    model.set_bad([], True)
    assert (log.data, log.metadata) == ([], 0)


def test_rename_writes_the_stream(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that a rename reaches the stream and emits once for the row."""
    log = emissions(model)
    model.rename(0, "  Renamed  ")
    assert mixed_stream.info.ch_names[0] == "Renamed"
    assert model.channel(0).name == "Renamed"
    assert log.data == [(0, 0)]
    assert (log.layout, log.metadata) == (0, 1)


def test_rename_many_swaps_two_names(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that a permutation of names is applied, which one write per row cannot do.

    The underlying operation refuses a target still held by another channel, so renaming
    row by row fails on the first half of a swap. This is what makes a grouped write
    necessary rather than merely faster: restoring a saved configuration that only
    exchanges two names would otherwise lose the first one, silently.

    Fails if ``rename_many`` reverts to calling the single-channel path in a loop, and
    fails if it stops emitting once for the span.
    """
    first, second = model.channel(0).name, model.channel(1).name
    log = emissions(model)
    model.rename_many({0: second, 1: first})
    assert mixed_stream.info.ch_names[:2] == [second, first]
    assert (model.channel(0).name, model.channel(1).name) == (second, first)
    assert log.data == [(0, 1)]
    assert (log.layout, log.metadata) == (0, 1)


def test_rename_many_rejects_a_collision_with_an_untouched_channel(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a grouped rename still refuses to duplicate a name it does not touch.

    Uniqueness is checked against the names the request *leaves*, which is what allows a
    swap while still refusing a genuine collision. Fails if the check is dropped, or if
    it is written against the names in use now, which would also refuse the swap above.
    """
    third = model.channel(2).name
    before = list(mixed_stream.info.ch_names)
    with pytest.raises(ValueError, match="not unique"):
        model.rename_many({0: third})
    assert mixed_stream.info.ch_names == before


def test_rename_rejects_blank_and_unprintable(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a blank or unprintable name is refused.

    'rename_channels' accepts both an empty and a whitespace-only name and leaves the
    channel nameless, so this guard is the only thing standing between the user and an
    unaddressable channel.

    Printability matters as much as blankness, and 'str.strip' does not cover it: it
    removes only what 'str.isspace' matches, so a zero-width space, a byte-order mark, a
    Mongolian vowel separator, an embedded newline and an embedded NUL all survive it. A
    channel named with one paints nothing and cannot be searched for.

    Looped rather than parametrized: every case is read-only, and each parameter would
    cost another stream connection.
    """
    for name in ("", "   ", "\t", "​", "﻿", "᠎", "x\ny", "x\x00y"):
        with pytest.raises(ValueError, match="non-empty and printable"):
            model.rename(0, name)
    assert mixed_stream.info.ch_names[0] == "Fp1"
    # and it must not refuse a legitimate name: a unit symbol, a digit suffix, a space
    for name in ("µV", "Cz-0", "a b"):
        model.rename(0, name)
    assert mixed_stream.info.ch_names[0] == "a b"


def test_rename_rejects_a_duplicate(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that renaming onto another channel's name is refused.

    MNE's own escape hatch for this mangles *both* names and leaves the bad list holding
    a name the info no longer has, so it is never used.
    """
    model.set_bad([1], True)
    with pytest.raises(ValueError, match="already in use"):
        model.rename(0, "Fp2")
    assert mixed_stream.info.ch_names[:2] == ["Fp1", "Fp2"]
    assert mixed_stream.info["bads"] == ["Fp2"]


def test_rename_same_name_is_silent(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that renaming a channel to its current name emits nothing."""
    log = emissions(model)
    model.rename(0, "Fp1")
    assert (log.data, log.metadata) == ([], 0)


def test_rename_keeps_the_bad_state(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a bad channel stays bad through a rename.

    MNE remaps the bad list itself, thus a hand-rolled remap on top of it would be the
    bug rather than the fix.
    """
    model.set_bad([0], True)
    model.rename(0, "Renamed")
    assert mixed_stream.info["bads"] == ["Renamed"]
    assert model.channel(0).bad is True


def test_rename_never_allows_duplicates(
    model: ChannelModel, mixed_stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the rename is issued without MNE's duplicate escape hatch.

    Asserted on the call itself because the model's own guard makes the flag unreachable
    from the outside, and what it does when it *is* reached is unrecoverable: it mangles
    both colliding names and leaves the bad list holding a name the info no longer has.
    """
    calls: list[tuple[tuple, dict]] = []
    original = mixed_stream.rename_channels

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(mixed_stream, "rename_channels", _spy)
    model.rename(0, "Renamed")
    assert calls == [(({"Fp1": "Renamed"},), {})]


def test_rename_on_a_deduplicated_name(
    app: QApplication,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a deduplicated name renames without touching its twin.

    The mapping is keyed on the info's own names, so 'Cz-0' is addressable; keying it on
    the raw stream description would make both channels permanently unrenameable.
    """
    with pytest.warns(RuntimeWarning, match="Channel names are not unique"):
        stream, _ = lsl_stream(
            n_channels=4, ch_names=["Cz", "Cz", "Fp1", "STI"], n_stim=1
        )
    model = ChannelModel(stream)
    try:
        model.rename(0, "CzA")
        assert list(stream.info.ch_names) == ["CzA", "Cz-1", "Fp1", "STI"]
    finally:
        model.deleteLater()
        app.processEvents()


def test_a_failed_write_leaves_the_cache_agreeing_with_the_stream(
    model: ChannelModel, mixed_stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a stream write which raised half-way still refreshes the cache.

    'set_channel_types' is not all-or-nothing across its mapping, so a raise can leave
    the stream changed. With the re-read skipped on the failure path the cached mirror
    keeps the pre-write values for the rest of the session -- nothing in the viewer
    calls 'refresh()' -- and Reset, keyed on the cached names, then fails too.
    """
    original = mixed_stream.set_channel_types

    def _half_apply(mapping, **kwargs):
        """Apply the first key of ``mapping``, then raise as MNE does."""
        original({next(iter(mapping)): mapping[next(iter(mapping))]}, **kwargs)
        raise ValueError("half-applied")

    monkeypatch.setattr(mixed_stream, "set_channel_types", _half_apply)
    with pytest.raises(ValueError, match="half-applied"):
        model.set_type([0, 1], "eog")
    assert _stream_types(mixed_stream)[:2] == ["eog", "eeg"]
    assert [model.channel(row).ch_type for row in (0, 1)] == ["eog", "eeg"]


def test_reset_restores_every_field(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a reset restores the name, type, unit and bad state, in the stream."""
    model.set_type([0], "misc")
    model.set_unit([5], "V")
    model.rename(1, "Renamed")
    model.set_bad([2], True)
    model.reset_metadata(range(8))
    assert list(mixed_stream.info.ch_names) == [
        "Fp1",
        "Fp2",
        "Cz",
        "Pz",
        "EOG",
        "ECG",
        "STI",
        "MISC",
    ]
    assert _stream_types(mixed_stream) == ["eeg"] * 4 + ["eog", "ecg", "stim", "misc"]
    assert _stream_units(mixed_stream)[0] == (_V, -6)
    assert _stream_units(mixed_stream)[5] == (_V, -3)
    assert mixed_stream.info["bads"] == []
    for row in range(8):
        channel = model.channel(row)
        assert (
            channel.name,
            channel.ch_type,
            channel.unit_kind,
            channel.unit_mul,
            channel.bad,
        ) == channel.orig


def test_reset_order_is_types_units_renames_bads(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a reset of all four fields at once restores all four.

    The write order is forced by the API: a type change resets the multiplier, so units
    must follow types -- and writing the multiplier first would raise outright, as the
    channel is a unit-less misc channel until its type is restored.
    """
    model.set_type([0], "misc")
    model.rename(0, "Renamed")
    model.set_bad([0], True)
    assert model.channel(0).unit == _NO_UNIT
    model.reset_metadata([0])
    channel = model.channel(0)
    assert (channel.name, channel.ch_type, channel.unit, channel.bad) == (
        "Fp1",
        "eeg",
        "µV",
        False,
    )
    assert _stream_units(mixed_stream)[0] == (_V, -6)


def test_reset_restores_a_bad_channel_to_bad(
    app: QApplication,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a channel which was bad at acquisition is bad again after a reset."""
    stream, _ = lsl_stream(n_channels=4, n_stim=1)
    stream.info["bads"] = ["ch0"]
    model = ChannelModel(stream)
    try:
        assert model.channel(0).orig[4] is True
        model.set_bad([0], False)
        assert stream.info["bads"] == []
        model.reset_metadata([0])
        assert stream.info["bads"] == ["ch0"]
        assert model.channel(0).bad is True
    finally:
        model.deleteLater()
        app.processEvents()


def test_reset_skips_a_rename_onto_a_taken_name(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a reset whose original name is held by another channel still applies.

    Resetting a swapped pair of names would otherwise raise and leave nothing restored;
    the rename alone is skipped and every other field is still put back.
    """
    model.rename(0, "spare")
    model.rename(1, "Fp1")  # channel 1 now holds channel 0's original name
    model.set_type([0], "eog")
    assert model.reset_metadata([0]) == ["spare"]  # and it says which
    assert model.channel(0).ch_type == "eeg"
    assert model.channel(0).name == "spare"  # the rename step was skipped
    assert list(mixed_stream.info.ch_names)[:2] == ["spare", "Fp1"]


def test_reset_reports_nothing_when_it_restored_everything(
    model: ChannelModel,
) -> None:
    """Test that a reset which applied in full reports no skipped channel.

    Reset is the escape hatch out of a confusing metadata state, so the caller has to be
    able to tell the two apart: a silent decline leaves the row named 'spare' under an
    inspector still reading 'orig: Fp1'.
    """
    model.rename(0, "spare")
    model.set_type([1], "eog")
    assert model.reset_metadata([0, 1]) == []
    assert [model.channel(row).name for row in (0, 1)] == ["Fp1", "Fp2"]


def test_reset_restores_a_channel_which_had_no_unit(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a unit-less channel given a type and a unit resets to no unit at all.

    The one reset sequence where the units mapping must be built from the acquisition
    baseline and not from the cache: by the time the units are written the type write
    has already put the channel back to a kind which is not a physical quantity, and MNE
    refuses a multiplier on such a channel outright. Reading the cache -- which still
    holds the pre-type-write kind -- would include the channel and make the reset raise.
    """
    misc = _row_of(model, "misc")
    assert model.channel(misc).orig.unit_kind == _NONE
    model.set_type([misc], "eeg")
    model.set_unit([misc], "mV")
    assert _stream_units(mixed_stream)[misc] == (_V, -3)
    assert model.reset_metadata([misc]) == []
    assert model.channel(misc).ch_type == "misc"
    assert model.channel(misc).unit == _NO_UNIT
    assert _stream_units(mixed_stream)[misc] == (_NONE, 0)


def test_orig_never_claims_a_multiplier_it_cannot_restore(
    app: QApplication,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the baseline of a unit-less channel records no multiplier.

    A stream may declare its units as bare exponents, and the multiplier is then written
    independently of the kind the channel type implies, so a misc channel declared '-6'
    reaches the info as kind-none with a -6 multiplier. That state cannot be written
    back: MNE refuses a multiplier on a kind-none channel. Recording it would make Reset
    claim a baseline it can never restore, and both the acquisition and the restored
    value render as the same unit-less label, so nothing would show the 1e6 discrepancy.
    """
    stream, _ = lsl_stream(
        n_channels=2,
        n_stim=1,
        ch_names=["MISC", "STI"],
        ch_types=["misc", "stim"],
        ch_units=["-6", "none"],
    )
    model = ChannelModel(stream)
    try:
        assert model.channel(0).unit_kind == _NONE
        assert model.channel(0).unit_mul == -6  # what the stream really declared
        assert model.channel(0).orig.unit_kind == _NONE
        assert model.channel(0).orig.unit_mul == 0  # not claimed: it cannot be restored
        assert model.reset_metadata([0]) == []  # and the reset itself does not raise
    finally:
        model.deleteLater()
        app.processEvents()


def test_orig_is_a_named_tuple(model: ChannelModel) -> None:
    """Test that the acquisition baseline is read by name as well as by position.

    The kind and the multiplier are adjacent integers, so swapping the two -- which the
    reset path did -- type-checks and passes the happy path.
    """
    orig = model.channel(5).orig
    assert (orig.name, orig.ch_type, orig.bad) == ("ECG", "ecg", False)
    assert (orig.unit_kind, orig.unit_mul) == (_V, -3)
    assert orig == ("ECG", "ecg", _V, -3, False)  # positional reads still work


def test_reset_does_not_change_visibility_or_order(model: ChannelModel) -> None:
    """Test that a metadata reset leaves the presentation state alone."""
    model.order_by("alphabetical")
    model.set_visible([0, 1], False)
    order = model.presentation_order()
    hidden = model.hidden_channels()
    model.set_type(range(8), "misc")
    model.reset_metadata(range(8))
    assert model.presentation_order() == order
    assert model.hidden_channels() == hidden


def test_reset_emits_once(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that a reset of four fields over three rows emits once, not per field."""
    model.set_type([0, 1, 2], "eog")
    model.set_bad([0, 1, 2], True)
    log = emissions(model)
    model.reset_metadata([0, 1, 2])
    assert log.data == [(0, 2)]
    assert (log.layout, log.metadata) == (0, 1)


def test_reset_empty_rows_is_silent(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that resetting an empty selection performs no write."""
    log = emissions(model)
    model.reset_metadata([])
    assert (log.data, log.metadata) == ([], 0)


def test_orig_is_never_written(model: ChannelModel) -> None:
    """Test that the acquisition baseline survives every mutator.

    A refresh which rebuilt 'orig' from the current values would make Reset a no-op
    forever, silently.
    """
    before = [model.channel(row).orig for row in range(8)]
    model.set_type([0], "eog")
    model.set_unit([1], "mV")
    model.rename(2, "Renamed")
    model.set_bad([3], True)
    model.set_visible([4], False)
    model.order_by("alphabetical")
    model.refresh()
    by_acq = {model.channel(row).acq_index: model.channel(row) for row in range(8)}
    assert [by_acq[acq].orig for acq in range(8)] == before


# -- refresh and the stream boundary ---------------------------------------------------
def test_refresh_picks_up_an_external_edit(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a change made behind the model's back is visible after a refresh."""
    mixed_stream.rename_channels({"Fp1": "Renamed"})
    mixed_stream.set_channel_types({"Fp2": "eog"}, on_unit_change="ignore")
    mixed_stream.info["bads"] = ["Cz"]
    model.refresh()
    assert model.channel(0).name == "Renamed"
    assert model.channel(1).ch_type == "eog"
    assert model.channel(2).bad is True


def test_refresh_keeps_order_and_visibility_when_the_count_is_unchanged(
    model: ChannelModel, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that an identity-preserving refresh does not reset the model."""
    model.order_by("alphabetical")
    model.set_visible([0], False)
    order = model.presentation_order()
    hidden = model.hidden_channels()
    log = emissions(model)
    model.refresh()
    assert model.presentation_order() == order
    assert model.hidden_channels() == hidden
    assert log.reset == 0
    assert log.data == [(0, 7)]
    assert (log.layout, log.metadata) == (0, 1)


def test_refresh_rebuilds_on_a_structural_change(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that a changed channel count rebuilds the model in acquisition order.

    Keeping the old rows would leave the layout holding acquisition indices which no
    longer mean the same channel, which is exactly what the display's guard refuses.
    """
    model.order_by("alphabetical")
    model.set_visible([0, 1], False)
    log = emissions(model)
    mixed_stream.add_reference_channels("REF")
    model.refresh()
    assert log.reset == 1
    assert model.n_channels == 9
    assert model.presentation_order() == list(range(9))
    assert model.visible_acq_indices() == list(range(9))
    assert model.channel(8).name == "REF"
    assert (log.layout, log.metadata) == (1, 1)


def test_refresh_emits_both_signals_on_a_reset(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that a structural refresh pushes both a layout and a metadata signal.

    Only one would leave the display holding a layout built against the old channel set.
    """
    log = emissions(model)
    mixed_stream.add_reference_channels("REF")
    model.refresh()
    assert (log.layout, log.metadata) == (1, 1)


def test_refresh_emits_metadata_before_layout(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test the order of the two coarse signals on a structural refresh."""
    order: list[str] = []
    model.metadata_changed.connect(lambda: order.append("metadata"))
    model.layout_changed.connect(lambda: order.append("layout"))
    mixed_stream.add_reference_channels("REF")
    model.refresh()
    assert order == ["metadata", "layout"]


def test_refresh_signal_order_lets_a_display_grow(
    app: QApplication, model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a grown channel set reaches a display wired the mandated way.

    The reason the signal order is fixed. The display validates a pushed layout against
    the channel count it last read, so pushing the grown layout first has it refused
    against the stale count -- and the refusal is raised inside a slot, where it is only
    logged, leaving the display stuck on the previous channel set for good.
    """
    widget = TraceDisplay(mixed_stream)
    try:
        widget.set_channel_layout(model.visible_acq_indices())
        model.metadata_changed.connect(widget.refresh_metadata)
        model.layout_changed.connect(
            lambda: widget.set_channel_layout(model.visible_acq_indices())
        )
        mixed_stream.add_reference_channels("REF")
        model.refresh()
        assert widget.n_channels == 9
        assert widget.n_rows == 9
    finally:
        widget.stop()
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_metadata_writes_are_a_no_op_on_a_disconnected_stream(
    model: ChannelModel,
    mixed_stream: StreamLSL,
    emissions: Callable[[ChannelModel], Emissions],
) -> None:
    """Test that every metadata mutator tolerates a stream which went away.

    Reachable from the inspector's Bad, Type, Unit and Reset controls: 'refresh()' comes
    back early while the stream is down, so the panel stays live and every one of those
    raised an unguarded 'RuntimeError' out of a Qt slot. Visibility and ordering already
    tolerated it, and this is what makes all of them agree.
    """
    log = emissions(model)
    mixed_stream.disconnect()
    model.set_bad([0], True)
    model.set_type([0], "eog")
    model.set_unit([0], "mV")
    model.rename(0, "Renamed")
    assert model.reset_metadata([0]) == []
    assert (log.data, log.layout, log.metadata) == ([], 0, 0)
    assert model.channel(0).name == "Fp1"  # the last known metadata is kept
    # presentation state still works: it needs no stream at all
    model.set_visible([0], False)
    model.order_by("alphabetical")
    assert (log.layout, log.metadata) == (2, 0)


def test_read_stream_refuses_a_changed_channel_count(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that a stream whose channel set changed surfaces as an explicit refusal.

    The cached mirror is indexed by acquisition index, so a stream which grew or shrank
    behind the model would otherwise raise 'IndexError' from whichever mutator ran next
    -- after that mutator's own write had already landed.
    """
    mixed_stream.add_reference_channels("REF")
    with pytest.raises(ValueError, match="must be refreshed"):
        model._read_stream()
    with pytest.raises(ValueError, match="must be refreshed"):
        model.set_bad([0], True)
    model.refresh()  # the rebuild is the way back
    model._read_stream()
    assert model.n_channels == 9


def test_model_over_a_disconnected_stream_is_empty(
    app: QApplication, mixed_stream: StreamLSL
) -> None:
    """Test that a disconnected stream yields an empty model rather than raising."""
    mixed_stream.disconnect()
    model = ChannelModel(mixed_stream)
    try:
        assert model.rowCount() == 0
        assert model.n_channels == 0
        assert model.visible_acq_indices() == []
        model.refresh()  # must not raise either
        assert model.rowCount() == 0
    finally:
        model.deleteLater()
        app.processEvents()


def test_read_stream_on_a_disconnected_stream_keeps_the_cache(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that the metadata read is a safe no-op once the stream is gone.

    Reading ``info`` off a disconnected stream raises, and this is the primitive a
    reconnect path calls, so it has to leave the last known metadata in place instead of
    taking the model down with the stream.
    """
    before = [model.channel(row).name for row in range(8)]
    mixed_stream.disconnect()
    model._read_stream()  # must not raise
    assert [model.channel(row).name for row in range(8)] == before


def test_metadata_read_uses_explicit_picks(
    app: QApplication, mixed_stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the metadata read always passes explicit integer picks.

    'picks=None' costs ~10x more to resolve at the call level and, historically, made
    the unit list drop the bad channels and misalign with the channel order.
    """
    calls: list[dict] = []

    def _spy(name: str) -> Callable:
        original = getattr(mixed_stream, name)

        def _wrapper(*args, **kwargs):
            calls.append({"name": name, "args": args, "kwargs": kwargs})
            return original(*args, **kwargs)

        return _wrapper

    for name in ("get_channel_types", "get_channel_units"):
        monkeypatch.setattr(mixed_stream, name, _spy(name))
    model = ChannelModel(mixed_stream)
    try:
        model.set_bad([0], True)  # a second read, through a mutator
        assert len(calls) == 4
        for call in calls:
            assert call["args"] == ()
            picks = call["kwargs"]["picks"]
            assert picks == list(range(8))
            assert all(isinstance(pick, int) for pick in picks)
    finally:
        model.deleteLater()
        app.processEvents()


def test_model_never_calls_get_channel_info(
    module_scan: Callable[[ModuleType], tuple[set[str], set[str]]],
) -> None:
    """Test that the model never reads the LSL stream description for its names.

    Both accessors are hazards: one re-emits the duplicate-name warning, which is an
    error in this suite, and the other returns the raw, undeduplicated names.
    """
    _imports, identifiers = module_scan(_model)
    assert identifiers.isdisjoint({"get_channel_info", "get_channel_names"})


def test_no_display_import(
    module_scan: Callable[[ModuleType], tuple[set[str], set[str]]],
) -> None:
    """Test that no module of 'controller/' reaches the display or the LSL layer.

    A source-level check, as importing any viewer module necessarily imports 'mne_lsl'
    and therefore 'mne_lsl.lsl'.
    """
    found = {
        module.name
        for module in pkgutil.iter_modules(mne_lsl.viewer.controller.__path__)
    }
    assert found == set(_MODULES)  # a new module must join the scan
    for name, module in _MODULES.items():
        imports, identifiers = module_scan(module)
        segments = {segment for path in imports for segment in path.split(".")}
        assert (segments | identifiers).isdisjoint(_FORBIDDEN), name


# -- unit registry ---------------------------------------------------------------------
def test_unit_label_round_trips() -> None:
    """Test that every offered pair is labelled by the label it resolves back from.

    Driven from the ladders rather than from 'UNIT_LABELS', which is built by keying a
    dict on 'unit_label' and would therefore satisfy the round trip by construction --
    including if 'unit_label' collapsed every pair onto one string.
    """
    assert UNIT_LABELS
    for kind, muls in _MULTIPLIERS.items():
        for mul in muls:
            label = unit_label(kind, mul)
            assert label != _NO_UNIT, (kind, mul)
            assert unit_pair(label) == (kind, mul), label


def test_unit_labels_are_unique() -> None:
    """Test that no two offered pairs collapse onto one label.

    A collision would make one of the two unreachable from the combo, silently. Counted
    against the ladders and not against 'set(UNIT_LABELS)': the list is 'list(dict)', so
    its own uniqueness is a theorem rather than a property of the labelling.
    """
    assert len(UNIT_LABELS) == sum(len(muls) for muls in _MULTIPLIERS.values())


@pytest.mark.parametrize(
    ("mul", "expected"),
    [
        (0, "V"),
        (-1, "100 mV"),
        (-2, "10 mV"),
        (-3, "mV"),
        (-6, "µV"),
        (-9, "nV"),
        (1, "10 V"),
        (3, "kV"),
    ],
)
def test_unit_label_builds_the_coefficient(mul: int, expected: str) -> None:
    """Test that a multiplier which is not a multiple of three reads as a coefficient.

    The label is generated rather than tabulated, so it covers every multiplier a stream
    may declare and not only the ones the control offers.
    """
    assert unit_label(_V, mul) == expected


@pytest.mark.parametrize(("kind", "mul"), [(_NONE, 0), (999, -6), (_V, 40)])
def test_unit_label_falls_back_to_none(kind: int, mul: int) -> None:
    """Test that an unlabelable pair reads '(none)' instead of raising.

    This is read from a paint path, where an exception would be at best logged.
    """
    assert unit_label(kind, mul) == _NO_UNIT


def test_unit_pair_unknown_label() -> None:
    """Test that an unknown label resolves to the unit-less pair rather than raising."""
    assert unit_pair("banana") == (_NONE, 0)
    assert unit_pair(_NO_UNIT) == (_NONE, 0)


def test_unit_choices_share_one_kind() -> None:
    """Test that every label offered for a kind belongs to that kind.

    This is what makes it structurally impossible for the Unit control to write a kind.
    """
    for kind in _MULTIPLIERS:
        labels = unit_choices(kind)
        assert labels
        assert {unit_pair(label)[0] for label in labels} == {kind}


def test_unit_ladders_hold_only_writable_multipliers() -> None:
    """Test that no offered rung is a multiplier MNE would refuse.

    MNE validates 'unit_mul' against its 17 named FIFF constants, so a plausible ladder
    entry which is not one of them -- '100 µV' and '10 µV', i.e. -4 and -5 -- raises
    from inside a Qt slot. The ladders are filtered through those constants, which makes
    unwritable rung impossible rather than detectable, so it is the raw table which has
    to be checked here.
    """
    assert set(_MULTIPLIERS) == set(_KINDS)
    for kind, (_symbol, raw) in _KINDS.items():
        assert _MULTIPLIERS[kind] == [mul for mul in raw if mul in _ch_unit_mul_named]
        assert _MULTIPLIERS[kind]  # a kind offering nothing would disable the control


def test_unit_choices_for_a_unitless_kind_is_empty() -> None:
    """Test that a channel with no physical unit is offered nothing.

    Every label would raise on such a channel; a unit is acquired through the Type
    control instead.
    """
    assert unit_choices(_NONE) == []


def test_unit_choices_for_a_mixed_selection_is_empty() -> None:
    """Test that a selection spanning several kinds is offered nothing."""
    assert unit_choices(None) == []


def test_unit_choices_are_shared_by_types_of_one_kind(model: ChannelModel) -> None:
    """Test that two channel types sharing a kind offer the same ladder.

    An eeg and an ecg channel are both Volt channels, thus a mixed selection of the two
    keeps a working Unit control -- keying the choices on the channel type would have
    disabled it.
    """
    eeg = model.channel(_row_of(model, "eeg"))
    ecg = model.channel(_row_of(model, "ecg"))
    assert eeg.unit_kind == ecg.unit_kind
    assert unit_choices(eeg.unit_kind) == unit_choices(ecg.unit_kind)
    assert unit_choices(eeg.unit_kind)


def test_unit_choices_match_the_real_write_path(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that every offered label can really be written to a channel of that kind.

    This is the test which ties the control's offer to the API's capability: MNE
    validates the multiplier against its named FIFF constants, so a plausible-looking
    ladder entry such as '100 µV' -- which is not one of them -- would raise from inside
    a Qt slot.
    """
    for row in range(model.rowCount()):
        for label in unit_choices(model.channel(row).unit_kind):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                model.set_unit([row], label)
            assert model.channel(row).unit == label
            assert unit_label(*_stream_units(mixed_stream)[row]) == label


def test_ch_types_are_settable_on_a_real_stream(
    model: ChannelModel, mixed_stream: StreamLSL
) -> None:
    """Test that every offered channel type survives a real, warning-free write."""
    for ch_type in CH_TYPES:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.set_type([0], ch_type)
        assert _stream_types(mixed_stream)[0] == ch_type
