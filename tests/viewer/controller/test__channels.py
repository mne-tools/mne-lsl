from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import QEvent, QItemSelectionModel, QPoint, QPointF, QRect, Qt
from qtpy.QtGui import QFont, QFontMetrics, QMouseEvent, QPainter, QPixmap
from qtpy.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QListView,
    QMenu,
    QStyle,
    QStyleOptionViewItem,
    QWidget,
)

from mne_lsl.viewer.controller import ChannelDelegate, ChannelModel, ChannelsPage
from mne_lsl.viewer.controller._channels import MULTIPLE
from mne_lsl.viewer.controller._model import _NO_UNIT

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from qtpy.QtWidgets import QApplication

    from mne_lsl.stream import StreamLSL
    from mne_lsl.viewer.theme import ThemeController
    from tests.viewer.controller.conftest import Emissions

# Row rectangle every delegate assertion is made against, wide enough that the metadata
# cluster and the name both fit.
_ROW = QRect(0, 0, 300, 26)
# Matches a hardcoded '#rrggbb' colour, and deliberately not a bare '#': an object-name
# selector such as '#segControl' is legitimate QSS.
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def _select(page: ChannelsPage, rows: Sequence[int]) -> None:
    """Select exactly ``rows`` in the page's list view."""
    selection = page.view.selectionModel()
    selection.clearSelection()
    for row in rows:
        selection.select(
            page.model.index(row, 0), QItemSelectionModel.SelectionFlag.Select
        )


def _mouse(x: int, kind: QEvent.Type = QEvent.Type.MouseButtonRelease) -> QMouseEvent:
    """Return a left-button mouse event at ``x`` in a row's local coordinates."""
    pos = QPointF(x, 13)
    return QMouseEvent(
        kind,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _deliver(page: ChannelsPage, row: int, event: QMouseEvent) -> bool:
    """Deliver ``event`` to the delegate for ``row`` and return whether it was eaten."""
    option = QStyleOptionViewItem()
    option.rect = _ROW
    return page.view.itemDelegate().editorEvent(
        event, page.model, option, page.model.index(row, 0)
    )


def _layout(page: ChannelsPage, row: int) -> dict:
    """Return the delegate's sub-rectangles for ``row``."""
    return page.view.itemDelegate()._row_layout(
        _ROW, page.model.index(row, 0), page.font()
    )


def _combo_items(combo) -> list[str]:
    """Return the texts a combo box currently offers."""
    return [combo.itemText(index) for index in range(combo.count())]


def _stream_types(stream: StreamLSL) -> list[str]:
    """Return the channel types the stream itself reports, in acquisition order."""
    return list(stream.get_channel_types(picks=list(range(stream.info["nchan"]))))


# -- ownership -------------------------------------------------------------------------
def test_page_does_not_build_a_model(page: ChannelsPage, model: ChannelModel) -> None:
    """Test that the page presents the model it was handed, by identity.

    A page building its own would give the document two sources of truth, and the traces
    would follow neither.
    """
    assert page.model is model
    assert page.view.model() is model


def test_page_does_not_reparent_the_model(
    page: ChannelsPage, model: ChannelModel
) -> None:
    """Test that the model's ownership stays with whoever built it."""
    assert model.parent() is not page


# -- delegate --------------------------------------------------------------------------
def test_eye_click_toggles_visibility(page: ChannelsPage) -> None:
    """Test that a release inside the eye rectangle toggles the row's visibility."""
    assert page.model.channel(0).visible is True
    assert _deliver(page, 0, _mouse(16)) is True
    assert page.model.channel(0).visible is False
    assert _deliver(page, 0, _mouse(16)) is True
    assert page.model.channel(0).visible is True


def test_body_click_does_not_toggle(page: ChannelsPage) -> None:
    """Test that a click on the row body is left to the view.

    Without the rectangle test every selection click would hide a channel.
    """
    assert _deliver(page, 0, _mouse(150)) is False
    assert page.model.channel(0).visible is True


def test_eye_click_swallows_press_and_release(page: ChannelsPage) -> None:
    """Test that the press is eaten as well, so the eye click does not also select."""
    press = _mouse(16, QEvent.Type.MouseButtonPress)
    assert _deliver(page, 0, press) is True
    assert page.model.channel(0).visible is True  # toggled on the release only
    assert _deliver(page, 0, _mouse(16)) is True
    assert page.model.channel(0).visible is False


def test_other_events_are_not_handled(page: ChannelsPage) -> None:
    """Test that a non-click event over the eye is left alone."""
    move = _mouse(16, QEvent.Type.MouseMove)
    assert _deliver(page, 0, move) is False
    assert page.model.channel(0).visible is True


def test_eye_click_on_an_invalid_index_is_declined(page: ChannelsPage) -> None:
    """Test that a click carrying an invalid index is not consumed.

    Consuming it and then calling 'setData' on an invalid index eats the click and does
    nothing at all. The view bails out before it reaches a delegate with such an index,
    so this closes a latent hole rather than a reachable one.
    """
    option = QStyleOptionViewItem()
    option.rect = _ROW
    delegate = page.view.itemDelegate()
    invalid = page.model.index(999, 0)
    assert delegate.editorEvent(_mouse(16), page.model, option, invalid) is False


def test_size_hint_is_the_fixed_row_height(page: ChannelsPage) -> None:
    """Test that the row height is fixed, which is what uniform item sizes need."""
    hint = page.view.itemDelegate().sizeHint(
        QStyleOptionViewItem(), page.model.index(0, 0)
    )
    assert hint.height() == ChannelDelegate._ROW_H
    assert page.view.uniformItemSizes() is True


def test_row_layout_has_no_dead_gap(page: ChannelsPage) -> None:
    """Test that the metadata cluster is right-anchored and the name abuts it.

    This is the acceptance criterion the row layout was reworked for: the fixed-width
    columns it replaces left a wide empty band between the name and the type.
    """
    layout = _layout(page, 0)
    assert layout["unit"].right() >= _ROW.width() - 12
    cluster = min(layout[key].left() for key in ("dot", "type", "unit"))
    assert 0 <= cluster - layout["name"].right() <= 12


def test_row_layout_keeps_a_long_name_off_the_metadata_cluster(
    page: ChannelsPage,
) -> None:
    """Test that a very long name does not push the metadata cluster off the row."""
    page.model.rename(0, "C" * 200)
    layout = _layout(page, 0)
    assert layout["unit"].right() >= _ROW.width() - 12
    assert layout["name"].width() >= 10
    cluster = min(layout[key].left() for key in ("dot", "type", "unit"))
    assert layout["name"].right() <= cluster


def test_paint_elides_a_long_name(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the painted name is elided to the width the layout allowed.

    The elision lives in 'paint' and not in '_row_layout', which never reads the name at
    all, so the layout assertions above cannot see it. Without it a 200-character name
    is truncated with no ellipsis at the clip boundary -- it does *not* overpaint the
    cluster, which is what the layout already guarantees.
    """
    page.model.rename(0, "C" * 200)
    calls: list[tuple[str, int]] = []
    original = QFontMetrics.elidedText

    def _spy(self, text, mode, width, *args):
        calls.append((text, width))
        return original(self, text, mode, width, *args)

    monkeypatch.setattr(QFontMetrics, "elidedText", _spy)
    option = QStyleOptionViewItem()
    option.rect = _ROW
    option.palette = page.palette()
    option.font = page.font()
    pixmap = QPixmap(_ROW.width(), _ROW.height())
    painter = QPainter(pixmap)
    try:
        page.view.itemDelegate().paint(painter, option, page.model.index(0, 0))
    finally:
        painter.end()
    assert len(calls) == 1
    text, width = calls[0]
    assert text == "C" * 200
    assert width == _layout(page, 0)["name"].width()
    assert (
        original(QFontMetrics(option.font), text, Qt.TextElideMode.ElideRight, width)[
            -1
        ]
        == "…"
    )


def test_row_layout_keeps_the_cluster_on_a_narrow_row(page: ChannelsPage) -> None:
    """Test that a row narrower than its metadata cluster still lays the cluster out.

    The cluster is packed right-to-left from the right edge, so without a floor its
    origin goes negative on a narrow row and the type, the unit and the bad marker are
    not drawn at all -- and just above that width they sit under the eye glyph. The page
    carries a minimum width so a dock cannot get there, but the delegate must degrade
    lose three fields.
    """
    page.model.set_bad([0], True)
    eye = _layout(page, 0)["eye"]
    for width in (110, 130, 160):
        layout = page.view.itemDelegate()._row_layout(
            QRect(0, 0, width, 26), page.model.index(0, 0), page.font()
        )
        cluster = min(layout[key].left() for key in ("dot", "type", "unit", "bad"))
        assert cluster > eye.right(), width
        assert layout["name"].width() >= 10, width
    assert page.minimumWidth() >= 200


def test_bad_row_reserves_the_marker(page: ChannelsPage) -> None:
    """Test that the bad marker's rectangle exists only for a bad channel."""
    assert _layout(page, 0)["bad"] is None
    page.model.set_bad([0], True)
    assert _layout(page, 0)["bad"] is not None


def test_delegate_reads_bad_from_the_model(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that the bad rendering follows the stream, not a construction snapshot.

    The bad state is written to the stream from outside the page here, so the only way
    the delegate can see it is by reading the model role on every paint.
    """
    mixed_stream.info["bads"] = ["Fp1"]
    page.model.refresh()
    assert _layout(page, 0)["bad"] is not None


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("state", ["plain", "hidden", "bad"])
def test_delegate_paints_every_state(
    page: ChannelsPage, selected: bool, state: str
) -> None:
    """Test that painting a row succeeds in each of its states.

    A smoke test for the paint path itself: it is the one method whose sub-rectangle
    arithmetic and font swapping cannot be asserted without running it.
    """
    if state == "hidden":
        page.model.set_visible([0], False)
    elif state == "bad":
        page.model.set_bad([0], True)
    option = QStyleOptionViewItem()
    option.rect = _ROW
    option.palette = page.palette()
    option.font = page.font()
    if selected:
        option.state |= QStyle.StateFlag.State_Selected
    pixmap = QPixmap(_ROW.width(), _ROW.height())
    painter = QPainter(pixmap)
    try:
        page.view.itemDelegate().paint(painter, option, page.model.index(0, 0))
    finally:
        painter.end()


def test_only_a_selected_row_is_painted_bold(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the name weight is a selection cue and not a constant.

    The accepted design's selection cue is the accent bar plus the weight plus the
    background. An unconditionally bold name makes the weight distinguish nothing,
    leaving the 3 px bar as the only cue which is not a colour.

    The glyph cache is warmed by a first paint *before* the spy goes in: rasterizing a
    QtAwesome icon paints a glyph from an icon font, so it calls 'setFont' five more
    times, and the recorded list would not start at the name font on the very first row.
    """

    def _paint(selected: bool) -> None:
        """Paint row 0 once, selected or not."""
        option = QStyleOptionViewItem()
        option.rect = _ROW
        option.palette = page.palette()
        option.font = page.font()
        if selected:
            option.state |= QStyle.StateFlag.State_Selected
        pixmap = QPixmap(_ROW.width(), _ROW.height())
        painter = QPainter(pixmap)
        try:
            page.view.itemDelegate().paint(painter, option, page.model.index(0, 0))
        finally:
            painter.end()

    _paint(False)  # warms the pixmap cache
    fonts: list[tuple[bool, bool, bool]] = []
    original = QPainter.setFont

    def _spy(self, font: QFont) -> None:
        fonts.append((font.bold(), font.italic(), font.strikeOut()))
        original(self, font)

    monkeypatch.setattr(QPainter, "setFont", _spy)
    for selected in (False, True):
        fonts.clear()
        _paint(selected)
        # exactly two fonts: the name font, then the metadata font
        assert len(fonts) == 2, (selected, fonts)
        assert fonts[0][0] is selected, selected
        assert fonts[1][0] is False, selected


# -- inspector -------------------------------------------------------------------------
def test_inspector_hidden_without_a_selection(page: ChannelsPage) -> None:
    """Test that the inspector is hidden while nothing is selected."""
    assert page._inspector.isHidden()


def test_inspector_shown_on_selection(page: ChannelsPage) -> None:
    """Test that selecting a channel reveals the inspector."""
    _select(page, [0])
    assert not page._inspector.isHidden()
    page.view.selectionModel().clearSelection()
    assert page._inspector.isHidden()


def test_inspector_reflects_a_single_selection(page: ChannelsPage) -> None:
    """Test that every inspector field mirrors the selected channel."""
    _select(page, [5])
    assert page._insp_header.text() == "1 channel selected"
    assert page._name_edit.text() == "ECG"
    assert page._type_combo.currentText() == "ecg"
    assert page._unit_combo.currentText() == "mV"
    assert page._visible_btn.isChecked() is True
    assert page._bad_btn.isChecked() is False
    assert page._orig_label.text() == "orig: ECG ecg mV good"


def test_inspector_shows_multiple_for_a_mixed_field(page: ChannelsPage) -> None:
    """Test that a field which differs across the selection reads as mixed.

    Showing the first row's value instead would make committing the combo silently
    rewrite every other channel of the selection.
    """
    _select(page, [0, 5])
    assert page._insp_header.text() == "2 channels selected"
    assert page._type_combo.currentText() == MULTIPLE
    assert page._unit_combo.currentText() == MULTIPLE
    assert page._name_edit.placeholderText() == MULTIPLE


def test_selecting_emits_no_model_signal(
    page: ChannelsPage, emissions: Callable[[ChannelModel], Emissions]
) -> None:
    """Test that changing the selection pushes nothing to the trace display.

    The whole signal-blocker contract, and the only form of it which can fail: writing a
    state toggle's value back into it while reflecting fires ``toggled``, and the
    resulting model write is idempotent but still emits -- so every click would make the
    display rebuild its layout and re-read its metadata. Asserted on the model's own
    signals rather than on the stream writes, since a blocked combo never emits
    ``textActivated`` programmatically in the first place.
    """
    page.model.set_visible([1], False)
    page.model.set_bad([2], True)
    log = emissions(page.model)
    for rows in ([1], [2], [0], [0, 1, 2], []):
        _select(page, rows)
    assert (log.layout, log.metadata, log.data) == (0, 0, [])


def test_inspector_summarizes_a_long_selection(page: ChannelsPage) -> None:
    """Test that the acquisition-original line is summarized past a few channels."""
    _select(page, range(8))
    assert page._orig_label.text().endswith("+4 more")


def test_rename_disabled_for_a_multi_selection(page: ChannelsPage) -> None:
    """Test that renaming is a single-channel operation."""
    _select(page, [0, 1])
    assert page._name_edit.isEnabled() is False
    assert page._rename_btn.isEnabled() is False


def test_rename_disabled_for_a_blank_or_taken_name(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that the Rename button refuses a blank or already-used name.

    Disabling removes the failure path entirely instead of reporting it afterwards.
    Looped rather than parametrized: nothing is written, so one page covers every case.
    """
    _select(page, [0])
    for text in ("", "   ", "Fp2"):
        page._name_edit.setText(text)
        assert page._rename_btn.isEnabled() is False, text
        page._rename_selected()  # a programmatic click must be a no-op too
        assert mixed_stream.info.ch_names[0] == "Fp1", text


def test_rename_from_the_inspector_writes(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that the Rename button applies a free name to the single selection."""
    _select(page, [0])
    page._name_edit.setText("Renamed")
    assert page._rename_btn.isEnabled() is True
    page._rename_selected()
    assert mixed_stream.info.ch_names[0] == "Renamed"


def test_unit_combo_disabled_for_a_unitless_channel(page: ChannelsPage) -> None:
    """Test that a channel with no physical unit gets a disabled Unit combo.

    Every label would be refused by the write path; the tooltip points at the control
    which can actually give the channel a unit.
    """
    misc = next(row for row in range(8) if page.model.channel(row).ch_type == "misc")
    _select(page, [misc])
    assert page._unit_combo.isEnabled() is False
    assert "Type" in page._unit_combo.toolTip()
    assert _combo_items(page._unit_combo) == [_NO_UNIT]


def test_unit_combo_offers_the_ladder_for_a_mixed_volt_selection(
    page: ChannelsPage,
) -> None:
    """Test that a mixed selection sharing one kind keeps a working Unit combo.

    An eeg and an ecg channel are both Volt channels; keying the choices on the channel
    type rather than on the kind would have disabled the control for no reason.
    """
    _select(page, [0, 5])
    assert page._unit_combo.isEnabled() is True
    assert "µV" in _combo_items(page._unit_combo)


def test_unit_combo_disabled_for_a_mixed_kind_selection(page: ChannelsPage) -> None:
    """Test that a selection spanning several unit kinds is offered nothing."""
    _select(page, [0, 7])
    assert page._unit_combo.isEnabled() is False


def test_bulk_type_edit_applies_to_the_whole_selection(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that the Type combo writes every selected channel, not only the current."""
    _select(page, [0, 1, 2])
    page._apply_type("eog")
    assert _stream_types(mixed_stream)[:3] == ["eog"] * 3


def test_bulk_unit_edit_applies_to_the_whole_selection(page: ChannelsPage) -> None:
    """Test that the Unit combo writes every selected channel."""
    _select(page, [0, 1, 2])
    page._apply_unit("mV")
    assert [page.model.channel(row).unit for row in (0, 1, 2)] == ["mV"] * 3


def test_apply_type_skips_the_mixed_marker(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that committing the mixed marker writes nothing."""
    _select(page, [0, 5])
    page._apply_type(MULTIPLE)
    assert _stream_types(mixed_stream)[:6] == ["eeg"] * 4 + ["eog", "ecg"]


def test_apply_unit_skips_a_label_off_the_offered_ladder(page: ChannelsPage) -> None:
    """Test that only a label the control actually offers is written.

    The combo also carries the selection's *current* label, which sits off the ladder
    whenever the stream declared an unusual multiplier; re-picking it would otherwise be
    refused by the model from inside a Qt slot. Looped rather than parametrized: every
    case writes nothing, so one page covers all four.
    """
    _select(page, [0])
    for label in (MULTIPLE, _NO_UNIT, "fT", "nV"):
        page._apply_unit(label)
        assert page.model.channel(0).unit == "µV", label


def test_visible_and_bad_toggles_apply_to_the_selection(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that the inspector's two state toggles act on the whole selection."""
    _select(page, [0, 1])
    page._visible_btn.setChecked(False)
    assert page.model.visible_acq_indices() == [2, 3, 4, 5, 6, 7]
    page._bad_btn.setChecked(True)
    assert set(mixed_stream.info["bads"]) == {"Fp1", "Fp2"}


def test_reset_button_restores_the_selection(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that Reset restores the acquisition metadata of the selection."""
    _select(page, [0])
    page._apply_type("misc")
    page._apply_reset()
    assert page.model.channel(0).ch_type == "eeg"
    assert page.model.channel(0).unit == "µV"


def test_a_declined_write_does_not_leave_the_inspector_showing_it(
    page: ChannelsPage, mixed_stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a toggle whose write never landed goes back to what the model holds.

    Qt commits a widget's visual state before it emits -- 'QAbstractButton' sets
    'checked' before 'toggled' -- and a write which was refused emits no 'dataChanged',
    so nothing else puts the widget back. Reached with no tampering at all: the stream
    drops, the panel stays live because 'refresh()' comes back early while disconnected,
    and the user clicks Bad. A toggle reading 'Bad' over an empty bad list is the worse
    of the two failures.
    """
    _select(page, [0])
    mixed_stream.disconnect()
    page._bad_btn.setChecked(True)
    assert page._bad_btn.isChecked() is False
    assert page.model.channel(0).bad is False


def test_a_refused_write_is_reported_and_rolled_back(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a model refusal reaches a dialog instead of only the log.

    The exception policy swallows anything raised in a slot, so without the guard a
    refused inspector edit is invisible: the combo keeps the value it never wrote.
    """
    from qtpy.QtWidgets import QMessageBox

    warned: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warned.append(args[-1])
    )
    _select(page, [0])
    page._type_combo.setCurrentText("eog")  # commits the widget without writing
    monkeypatch.setattr(
        page.model,
        "set_type",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("refused by the model")),
    )
    page._apply_type("eog")
    assert len(warned) == 1
    assert "refused by the model" in warned[0]
    assert page._type_combo.currentText() == "eeg"  # rolled back to the model's value


def test_reset_reports_a_name_it_could_not_restore(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a reset which had to decline a rename says so.

    Reset is the escape hatch out of a confusing metadata state, so a silent decline
    leaves the row named 'spare' under an inspector still reading 'orig: Fp1'.
    """
    from qtpy.QtWidgets import QMessageBox

    warned: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warned.append(args[-1])
    )
    page.model.rename(0, "spare")
    page.model.rename(1, "Fp1")
    _select(page, [0])
    page._apply_reset()
    assert len(warned) == 1
    assert "spare" in warned[0]
    assert page.model.channel(0).name == "spare"
    # and a reset which applied in full says nothing
    warned.clear()
    _select(page, [2])
    page._apply_type("misc")
    page._apply_reset()
    assert warned == []


def test_context_menu_reset_reports_a_name_it_could_not_restore(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the menu's Reset entry reports the same skip as the inspector button.

    The menu path has no inspector state to fall back on, so wiring it straight to the
    model would drop the report entirely.
    """
    from qtpy.QtWidgets import QMessageBox

    warned: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warned.append(args[-1])
    )
    page.model.rename(0, "spare")
    page.model.rename(1, "Fp1")
    _trigger(page, [0], ["Reset"])
    assert len(warned) == 1
    assert "spare" in warned[0]


def test_bulk_edit_costs_one_handler_call(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a bulk edit runs the page handlers once, not once per row.

    Counted rather than timed, so there is nothing to flake: the model emits one
    spanning ``dataChanged``, and this proves the page does not fan it out per row.
    """
    calls: list[str] = []

    def _counting(name: str) -> Callable:
        """Return the page's ``name`` handler, wrapped in a call counter."""
        original = getattr(page, name)

        def _wrapper(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return _wrapper

    for name in ("_apply_filter", "_reflect_selection"):
        monkeypatch.setattr(page, name, _counting(name))
    _select(page, range(8))
    calls.clear()
    page.model.set_visible(range(8), False)
    assert calls.count("_apply_filter") == 1
    assert calls.count("_reflect_selection") == 1


# -- filtering and status --------------------------------------------------------------
def test_search_hides_rows_without_touching_the_model(page: ChannelsPage) -> None:
    """Test that the search box hides rows and changes no model state.

    A filter which drove the traces would, with 'Show: Hidden' selected, draw exactly
    the channels the user chose not to draw.
    """
    before = page.model.visible_acq_indices()
    page._search.setText("Fp")
    assert [page.view.isRowHidden(row) for row in range(8)] == [
        False,
        False,
        *[True] * 6,
    ]
    assert page.model.visible_acq_indices() == before


def test_filters_hide_exactly_the_non_matching_rows(page: ChannelsPage) -> None:
    """Test that each filter keeps exactly its matching rows, and no model state.

    The rows are asserted exactly rather than counted into a range: a count in a range
    is satisfied by no filtering at all, so all four filter branches could be deleted
    and still pass. The fixture is arranged so each case has a distinct answer:
    four eeg channels, one hidden channel and one bad channel.
    """
    page.model.set_visible([2], False)
    page.model.set_bad([5], True)
    before = page.model.visible_acq_indices()
    cases = {
        "_type_filter": ("eeg", [0, 1, 2, 3]),
        "_vis_filter": ("visible", [0, 1, 3, 4, 5, 6, 7]),
        "_bad_filter": ("good", [0, 1, 2, 3, 4, 6, 7]),
    }
    for combo, (data, expected) in cases.items():
        widget = getattr(page, combo)
        widget.setCurrentIndex(widget.findData(data))
        shown = [row for row in range(8) if not page.view.isRowHidden(row)]
        assert shown == expected, combo
        assert page.model.visible_acq_indices() == before, combo
        widget.setCurrentIndex(0)
    # the fourth branch: 'Show: Hidden' is the complement of 'Show: Visible'
    page._vis_filter.setCurrentIndex(page._vis_filter.findData("hidden"))
    assert [row for row in range(8) if not page.view.isRowHidden(row)] == [2]
    page._vis_filter.setCurrentIndex(0)
    page._bad_filter.setCurrentIndex(page._bad_filter.findData("bad"))
    assert [row for row in range(8) if not page.view.isRowHidden(row)] == [5]


def test_hidden_filter_shows_the_hidden_channels(page: ChannelsPage) -> None:
    """Test that 'Show: Hidden' lists exactly the channels which are hidden."""
    page.model.set_visible([2, 3], False)
    page._vis_filter.setCurrentIndex(page._vis_filter.findData("hidden"))
    shown = [row for row in range(8) if not page.view.isRowHidden(row)]
    assert shown == [2, 3]


def test_clear_filters_restores_every_row(page: ChannelsPage) -> None:
    """Test that clearing drops the search text and all three filters."""
    page._search.setText("Fp")
    page._type_filter.setCurrentIndex(page._type_filter.findData("ecg"))
    page._clear_filters()
    assert not any(page.view.isRowHidden(row) for row in range(8))
    assert page._search.text() == ""


def test_filter_reapplied_after_a_reorder(page: ChannelsPage) -> None:
    """Test that the same *channels* stay hidden across a reorder."""
    page._search.setText("Fp")
    page.model.order_by("alphabetical")
    shown = {
        page.model.channel(row).name
        for row in range(8)
        if not page.view.isRowHidden(row)
    }
    assert shown == {"Fp1", "Fp2"}


def test_filter_survives_a_model_reset(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that a structural change re-shows every row and drops the selection.

    A kept selection would leave the inspector editing whichever channels happen to sit
    at those rows after the rebuild.
    """
    page._search.setText("Fp")
    _select(page, [0, 1])
    mixed_stream.add_reference_channels("REF")
    page.model.refresh()
    assert page.view.selectionModel().selectedIndexes() == []
    assert page._inspector.isHidden()
    shown = [row for row in range(9) if not page.view.isRowHidden(row)]
    assert shown == [0, 1]  # the search text still applies, over the new rows


def test_status_line_counts(page: ChannelsPage) -> None:
    """Test the persistent status line's shape and counts."""
    assert page._status.text() == "0 selected · 8/8"
    _select(page, [0, 1, 2])
    assert page._status.text() == "3 selected · 8/8"
    page._search.setText("Fp")
    assert page._status.text() == "3 selected · 2/8"


def test_status_line_has_a_tooltip(page: ChannelsPage) -> None:
    """Test that the status line explains it counts list rows, not drawn traces."""
    assert page._status.toolTip()


def test_search_icon_is_a_named_decoration(page: ChannelsPage) -> None:
    """Test that the leading magnify glyph is labelled and not clickable.

    Every other compact control of the page carries an accessible name; an unnamed,
    tooltip-less action is announced as an unlabelled button by a screen reader, and
    clicking this one does nothing at all -- it is a decoration, so it says so.
    """
    assert page._search_action.text()
    assert page._search_action.toolTip()
    assert page._search_action.isEnabled() is False
    assert not page._search_action.icon().isNull()


def test_filter_only_rewrites_the_rows_whose_state_changed(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that re-filtering does not touch a row which is already in the right state.

    'setRowHidden' schedules a full relayout of the view unconditionally, so calling it
    on all 256 rows turns every eye click and every bulk edit into a whole-viewport
    repaint on top of the one Qt already scheduled.
    """
    calls: list[tuple[int, bool]] = []
    original = QListView.setRowHidden

    def _spy(self, row: int, hide: bool) -> None:
        calls.append((row, hide))
        original(self, row, hide)

    monkeypatch.setattr(QListView, "setRowHidden", _spy)
    page._apply_filter()  # nothing changed since construction
    assert calls == []
    page._search.setText("Fp")
    assert sorted(calls) == [(row, True) for row in range(2, 8)]
    calls.clear()
    page._search.setText("Fp1")
    assert calls == [(1, True)]


# -- ordering --------------------------------------------------------------------------
def test_order_control_starts_on_acquisition(page: ChannelsPage) -> None:
    """Test that the Order control and the model agree at construction."""
    assert page._order.current_index == 0
    assert page._order.current_value == "acquisition"
    assert page.model.presentation_order() == list(range(8))


def test_order_control_drives_the_model(page: ChannelsPage) -> None:
    """Test that clicking each segment applies the corresponding model ordering.

    The middle segment is asserted from a state which is neither the acquisition order
    nor the type order it produces. The fixture's type order *is* its acquisition order,
    so one channel is re-typed first: without that, and without the alphabetical click
    before it, a middle segment miswired to 'acquisition' produces the very permutation
    this asserts.
    """
    page.model.set_type([0], "misc")  # now the type order is a real permutation
    page._order._buttons[2].click()
    assert [page.model.channel(row).name for row in range(8)] == [
        "Cz",
        "ECG",
        "EOG",
        "Fp1",
        "Fp2",
        "MISC",
        "Pz",
        "STI",
    ]
    page._order._buttons[1].click()
    assert page.model.presentation_order() == [1, 2, 3, 4, 5, 6, 0, 7]
    page._order._buttons[0].click()
    assert page.model.presentation_order() == list(range(8))


def test_order_control_resyncs_on_a_structural_refresh(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that a rebuild puts the Order control back on the segment it now reflects.

    A rebuild restores the acquisition order, and re-clicking the segment the control
    already shows is a no-op, so a control left reading 'Abc' over an acquisition-order
    model offers the user no way back to alphabetical at all.
    """
    page._order._buttons[2].click()
    assert page._order.current_value == "alphabetical"
    mixed_stream.add_reference_channels("REF")
    page.model.refresh()
    assert page.model.presentation_order() == list(range(9))
    assert page._order.current_index == 0
    assert page._order.current_value == "acquisition"
    page._order._buttons[2].click()  # and the control is usable again
    assert page.model.presentation_order() != list(range(9))


# -- context menu ----------------------------------------------------------------------
def _trigger(page: ChannelsPage, rows: Sequence[int], path: Sequence[str]) -> None:
    """Trigger the context-menu action at ``path`` for ``rows``."""
    menu = page._menu_for(rows)
    for depth, text in enumerate(path):
        action = next(a for a in menu.actions() if a.text() == text)
        if depth < len(path) - 1:
            menu = action.menu()
        else:
            action.trigger()


def test_context_menu_actions_apply_to_the_selection(
    page: ChannelsPage, mixed_stream: StreamLSL
) -> None:
    """Test that every context-menu entry acts on the rows it was built for."""
    rows = [0, 1]
    _trigger(page, rows, ["Hide"])
    assert page.model.visible_acq_indices() == [2, 3, 4, 5, 6, 7]
    _trigger(page, rows, ["Show"])
    assert page.model.visible_acq_indices() == list(range(8))
    _trigger(page, rows, ["Set type", "eog"])
    assert _stream_types(mixed_stream)[:2] == ["eog", "eog"]
    _trigger(page, rows, ["Set unit", "mV"])
    assert [page.model.channel(row).unit for row in rows] == ["mV", "mV"]
    _trigger(page, rows, ["Mark bad"])
    assert set(mixed_stream.info["bads"]) == {"Fp1", "Fp2"}
    _trigger(page, rows, ["Mark good"])
    assert mixed_stream.info["bads"] == []
    _trigger(page, rows, ["Reset"])
    assert _stream_types(mixed_stream)[:2] == ["eeg", "eeg"]


def test_context_menu_on_an_unselected_row_uses_that_row(page: ChannelsPage) -> None:
    """Test that a right-click without a selection falls back to the row under it.

    Without the fallback the menu would open and do nothing at all.
    """
    page.view.resize(300, 400)
    center = page.view.visualRect(page.model.index(3, 0)).center()
    assert page._context_rows(center) == [3]
    assert page._context_rows(QPoint(10, 10_000)) == []  # past the last row
    _select(page, [0, 1])
    assert page._context_rows(center) == [0, 1]  # the selection wins


def test_context_menu_rename_enabled_only_for_one_row(page: ChannelsPage) -> None:
    """Test that the menu's Rename entry follows the single-channel rule."""
    single = next(a for a in page._menu_for([0]).actions() if a.text() == "Rename…")
    multi = next(a for a in page._menu_for([0, 1]).actions() if a.text() == "Rename…")
    assert single.isEnabled() is True
    assert multi.isEnabled() is False


def test_context_menu_unit_submenu_is_disabled_for_a_unitless_channel(
    page: ChannelsPage,
) -> None:
    """Test that the menu never offers a unit the write path would refuse."""
    misc = next(row for row in range(8) if page.model.channel(row).ch_type == "misc")
    action = next(a for a in page._menu_for([misc]).actions() if a.text() == "Set unit")
    assert action.isEnabled() is False
    assert action.menu().actions() == []


def test_context_menu_rename_dialog_reports_a_refusal(
    page: ChannelsPage, mixed_stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the dialog path reports the model's refusal exactly once.

    The dialog cannot be guarded by a button state, thus it is the one place a rename
    can still be refused -- and one explicit gesture may raise one modal.
    """
    from qtpy.QtWidgets import QInputDialog, QMessageBox

    warned: list[str] = []
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Fp2", True))
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args, **kwargs: warned.append(args[-1])
    )
    page._rename_dialog(0)
    assert len(warned) == 1
    assert "already in use" in warned[0]
    assert mixed_stream.info.ch_names[0] == "Fp1"


def test_context_menu_rename_dialog_cancelled_is_a_no_op(
    page: ChannelsPage, mixed_stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that dismissing the rename dialog writes nothing."""
    from qtpy.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Other", False))
    page._rename_dialog(0)
    assert mixed_stream.info.ch_names[0] == "Fp1"


def test_context_menu_does_not_accumulate(
    app: QApplication, page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that each right-click's menu is destroyed rather than kept by the page.

    The menu is parented to the page, and the page is a process singleton, so a menu
    left behind lives until the process ends -- with its actions and the bound callables
    of both submenus. Counted after delivering the deferred deletions, which
    'processEvents' alone does not.
    """
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: None)
    _select(page, [0])
    page.view.resize(300, 400)
    before = len(page.findChildren(QMenu))
    for _ in range(20):
        page._context_menu(page.view.visualRect(page.model.index(0, 0)).center())
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    assert len(page.findChildren(QMenu)) == before


# -- removed features ------------------------------------------------------------------
def test_removed_features_stayed_removed(page: ChannelsPage) -> None:
    """Test that the four features the accepted design dropped are still absent.

    Folded into one test: every assertion is read-only, and four separate ones cost four
    stream connections for the same page.
    """
    # the all/none/invert Select control
    for gone in ("_select_button", "_select_all", "_invert_selection"):
        assert not hasattr(page, gone), gone
    texts = {button.text() for button in page.findChildren(QAbstractButton)}
    assert "Select" not in texts

    # drag-reordering: ordering is command-driven
    assert page.view.dragDropMode() == QAbstractItemView.DragDropMode.NoDragDrop
    flags = page.model.flags(page.model.index(0, 0))
    assert not flags & Qt.ItemFlag.ItemIsDragEnabled
    assert not flags & Qt.ItemFlag.ItemIsDropEnabled

    # the move-to-top / move-to-bottom commands
    for gone in ("move_to_top", "move_to_bottom", "_move"):
        assert not hasattr(page.model, gone), gone
    menu_texts = {action.text() for action in page._menu_for([0]).actions()}
    assert not {text for text in menu_texts if "Move" in text}

    # the tiles/table duality
    for gone in ("table", "tiles", "stack"):
        assert not hasattr(page, gone), gone
    assert isinstance(page.view, QListView)
    # the other list views are the filter combo boxes' own popups, not a second
    # presentation of the channels.
    others = [
        view for view in page.findChildren(QListView) if view.model() is page.model
    ]
    assert others == [page.view]


# -- theme -----------------------------------------------------------------------------
def test_retheme_rebuilds_the_delegate_icons(
    page: ChannelsPage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that rethemeing asks the delegate to rebuild its baked glyphs.

    Spied rather than compared by identity: QtAwesome caches per colour, thus rebuilding
    without an actual colour change legitimately hands back the very same icon. The
    identity change across a real flip is pinned by the next test.
    """
    delegate = page.view.itemDelegate()
    calls: list[int] = []
    monkeypatch.setattr(delegate, "refresh_palette", lambda: calls.append(1))
    page.retheme()
    assert calls == [1]


def test_theme_flip_rethemes_without_the_document(
    app: QApplication, controller: ThemeController, page: ChannelsPage
) -> None:
    """Test that the page follows the theme on its own, with no shell involved."""
    controller.install(app, "light")
    delegate = page.view.itemDelegate()
    before = delegate._eye[(True, False)]
    controller.set_mode("dark")
    assert delegate._eye[(True, False)] is not before


def test_theme_flip_rebuilds_the_page_icons(
    app: QApplication, controller: ThemeController, page: ChannelsPage
) -> None:
    """Test that the page's own five glyphs are rebuilt, not only the delegate's.

    They bake their colour exactly as the delegate's do, and the Bad button's is keyed
    on the error token, so a retint which stopped after construction would leave the
    whole inspector in the previous mode's colours for the rest of the process.
    """
    controller.install(app, "light")
    before = {
        name: getattr(page, name).icon().cacheKey()
        for name in ("_bad_btn", "_reset_btn", "_visible_btn", "_filter_btn")
    }
    before["_search_action"] = page._search_action.icon().cacheKey()
    controller.set_mode("dark")
    after = {
        name: getattr(page, name).icon().cacheKey()
        for name in ("_bad_btn", "_reset_btn", "_visible_btn", "_filter_btn")
    }
    after["_search_action"] = page._search_action.icon().cacheKey()
    assert set(before) == set(after)
    for name in before:
        assert before[name] != after[name], name


def test_close_drops_the_theme_connection(
    app: QApplication,
    controller: ThemeController,
    model: ChannelModel,
    make_page: Callable[[ChannelModel], ChannelsPage],
) -> None:
    """Test that a closed page stops following the theme.

    The controller is a process singleton, thus a page which never disconnects is
    restyled for the rest of the process.
    """
    controller.install(app, "light")
    widget = make_page(model)
    delegate = widget.view.itemDelegate()
    widget.close()
    before = delegate._eye[(True, False)]
    controller.set_mode("dark")
    assert delegate._eye[(True, False)] is before


def test_reopen_restores_the_theme_connection(
    app: QApplication,
    controller: ThemeController,
    model: ChannelModel,
    make_page: Callable[[ChannelModel], ChannelsPage],
) -> None:
    """Test that a closed-and-reopened page follows the theme again, and catches up.

    The counterpart of the close: without it the panel keeps the previous mode's baked
    glyphs for the rest of the process. Accessibility rather than cosmetics -- the eye
    glyph is the only visibility cue which is not a colour, and it ends up drawn in the
    other mode's text colour.
    """
    controller.install(app, "light")
    widget = make_page(model)
    delegate = widget.view.itemDelegate()
    widget.close()
    controller.set_mode("dark")  # flipped while closed, so the glyphs are now stale
    stale = delegate._eye[(True, False)]
    widget.show()
    app.processEvents()
    assert delegate._eye[(True, False)] is not stale  # the show caught the missed flip
    following = delegate._eye[(True, False)]
    controller.set_mode("light")
    assert delegate._eye[(True, False)] is not following  # and it follows again
    widget.hide()


def test_reopen_does_not_duplicate_the_theme_connection(
    app: QApplication,
    controller: ThemeController,
    model: ChannelModel,
    make_page: Callable[[ChannelModel], ChannelsPage],
) -> None:
    """Test that showing an already-following page connects nothing a second time.

    A duplicate connection makes every later flip rebuild every glyph twice, and there
    is no portable 'UniqueConnection' for a Python slot.
    """
    controller.install(app, "light")
    widget = make_page(model)
    widget.show()
    app.processEvents()
    widget.show()  # a second show must not add a second connection
    removed = 0
    while True:
        try:
            controller.theme_changed.disconnect(widget.retheme)
        except TypeError:
            break
        removed += 1
    assert removed == 1
    controller.theme_changed.connect(widget.retheme)  # so the teardown close matches
    widget.hide()


def test_page_themes_through_palette_roles_only(page: ChannelsPage) -> None:
    """Test that no style sheet of the page hardcodes a colour.

    A literal colour is correct in one mode and unreadable in the other.
    """
    sheets = [page.styleSheet()]
    sheets += [child.styleSheet() for child in page.findChildren(QWidget)]
    assert not [sheet for sheet in sheets if _HEX.search(sheet)]
