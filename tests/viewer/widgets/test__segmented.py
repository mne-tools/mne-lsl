from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QButtonGroup

from mne_lsl.viewer.widgets import AnimatedSegmentedControl

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qtpy.QtWidgets import QApplication

_ITEMS = [
    ("Acq", "Acquisition order", "acquisition"),
    ("Type", "Channel type", "type"),
    ("Abc", "Alphabetical", "alphabetical"),
]


@pytest.fixture
def segmented(
    flush_deletes: Callable[..., None],
) -> Generator[tuple[AnimatedSegmentedControl, list[str]]]:
    """Yield an Order control and the list of values its 'changed' signal carried.

    The teardown really destroys the C++ object -- see the ``flush_deletes`` fixture --
    so that the use-after-delete a stale connection causes is exercised.
    """
    widget = AnimatedSegmentedControl(_ITEMS)
    widget.resize(240, 26)
    emitted: list[str] = []
    widget.changed.connect(emitted.append)
    yield widget, emitted
    widget.close()
    flush_deletes(widget)


def test_starts_on_the_first_segment(
    segmented: tuple[AnimatedSegmentedControl, list[str]],
) -> None:
    """Test that the control starts active on its first segment.

    A control starting with no selection makes 'current_value' raise, and the panel and
    the display would disagree about the order at t=0.
    """
    widget, emitted = segmented
    assert widget.current_index == 0
    assert widget.current_value == "acquisition"
    assert emitted == []


def test_click_selects_and_emits_the_value(
    segmented: tuple[AnimatedSegmentedControl, list[str]],
) -> None:
    """Test that a click emits the segment's *value*, not its label.

    The labels are abbreviations while the values are the model's ordering commands, so
    emitting the label would silently push 'Abc' into 'order_by'.
    """
    widget, emitted = segmented
    widget._buttons[2].click()
    assert widget.current_index == 2
    assert widget.current_value == "alphabetical"
    assert emitted == ["alphabetical"]


def test_reclick_is_a_no_op(
    segmented: tuple[AnimatedSegmentedControl, list[str]],
) -> None:
    """Test that clicking the active segment emits nothing."""
    widget, emitted = segmented
    widget._buttons[0].click()
    assert emitted == []
    assert widget.current_index == 0


def test_set_index_emit_false_is_silent(
    segmented: tuple[AnimatedSegmentedControl, list[str]],
) -> None:
    """Test that a silent sync updates the state without emitting.

    Restoring a persisted order must not write it straight back into the model.
    """
    widget, emitted = segmented
    widget.set_index(1, emit=False)
    assert widget.current_index == 1
    assert widget.current_value == "type"
    assert emitted == []


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_set_index_rejects_out_of_range(
    segmented: tuple[AnimatedSegmentedControl, list[str]], index: int
) -> None:
    """Test that an out-of-range index raises instead of an 'IndexError'.

    Phase F restores a persisted value through here, i.e. from user-editable JSON.
    """
    widget, emitted = segmented
    with pytest.raises(ValueError, match="segment index must be in"):
        widget.set_index(index)
    assert widget.current_index == 0
    assert emitted == []


def test_set_index_rejects_a_non_integer_and_stays_usable(
    app: QApplication, segmented: tuple[AnimatedSegmentedControl, list[str]]
) -> None:
    """Test that a non-integer index is refused, with the widget left intact.

    A persisted value comes back from JSON, which has no integer/float distinction and
    no boolean of its own, so '1.0', '"1"', 'None' and 'True' all reach this method.
    '1.0' passes a bare range comparison and then fails on the list lookup -- and if the
    index was already stored the widget is poisoned for good: 'current_value' raises,
    and so does every later resize, out of a virtual Qt itself invokes. A bare 'True' is
    worse still, being silently accepted as segment 1.
    """
    widget, emitted = segmented
    widget.show()
    app.processEvents()
    for index in (1.0, "1", None, True, False, 2.5):
        with pytest.raises(ValueError, match="segment index must be an integer"):
            widget.set_index(index)
    assert widget.current_index == 0
    assert widget.current_value == "acquisition"
    assert emitted == []
    # the widget is still alive: the two paths a poisoned index took down
    widget.resize(480, 26)
    app.processEvents()
    assert widget._highlight.geometry() == widget._buttons[0].geometry()
    widget.set_index(2)
    assert widget.current_value == "alphabetical"
    widget.hide()


def test_set_index_animate_false_snaps(
    app: QApplication, segmented: tuple[AnimatedSegmentedControl, list[str]]
) -> None:
    """Test that a non-animated select puts the highlight on the button immediately.

    Shown first, because the slide is only ever started for a visible highlight: on a
    hidden control the snap and the slide are indistinguishable, and Phase F restores a
    persisted order on a live panel, where a slide would read as a user action.
    """
    widget, _ = segmented
    widget.show()
    app.processEvents()
    widget.set_index(2, animate=False)
    assert widget._highlight.geometry() == widget._buttons[2].geometry()
    widget.hide()


def test_set_index_animate_true_slides(
    app: QApplication, segmented: tuple[AnimatedSegmentedControl, list[str]]
) -> None:
    """Test that an animated select starts from the previous geometry."""
    widget, _ = segmented
    widget.show()
    app.processEvents()
    before = widget._highlight.geometry()
    widget.set_index(2, animate=True)
    assert widget._anim.startValue() == before
    assert widget._anim.endValue() == widget._buttons[2].geometry()
    widget._anim.stop()
    widget.hide()


def test_selected_button_is_the_heaviest(
    app: QApplication, segmented: tuple[AnimatedSegmentedControl, list[str]]
) -> None:
    """Test that the selected segment's text really is drawn heavier than its siblings.

    Asserted on the resolved font and not on the dynamic property alone: the property
    changes nothing on screen by itself, since Qt does not re-run the style-sheet
    selectors when one changes. This kills all three ways the cue disappears at once --
    dropping the 'unpolish'/'polish' pair, clearing the property, and clearing the
    object name the selector is keyed on, which takes the border and radius with it.
    """
    widget, _ = segmented
    widget.show()
    app.processEvents()
    widget.set_index(1)
    app.processEvents()
    flags = [bool(button.property("seg_selected")) for button in widget._buttons]
    assert flags == [False, True, False]
    weights = [button.fontInfo().weight() for button in widget._buttons]
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
    widget.hide()


def test_resize_repositions_the_highlight(
    app: QApplication, segmented: tuple[AnimatedSegmentedControl, list[str]]
) -> None:
    """Test that the highlight tracks its button across a resize.

    The segments are equal-flex, thus they move under the pill whenever the panel
    holding the control changes width. The control is shown first because a hidden
    widget defers its layout, which would leave every button at its default geometry
    and make the assertion pass for the wrong reason.
    """
    widget, _ = segmented
    widget.show()
    app.processEvents()
    widget.set_index(2)
    before = widget._buttons[2].geometry()
    widget.resize(480, 26)
    assert widget._buttons[2].geometry() != before
    assert widget._highlight.geometry() == widget._buttons[2].geometry()
    widget.hide()


def test_show_repositions_the_highlight(
    app: QApplication, segmented: tuple[AnimatedSegmentedControl, list[str]]
) -> None:
    """Test that showing the control snaps the highlight onto the active button."""
    widget, _ = segmented
    widget.set_index(1)
    widget.show()
    app.processEvents()
    assert widget._highlight.isVisible()
    assert widget._highlight.geometry() == widget._buttons[1].geometry()
    widget.hide()


def test_a_real_mouse_click_reaches_the_segment(
    app: QApplication, segmented: tuple[AnimatedSegmentedControl, list[str]]
) -> None:
    """Test that a segment is clickable through hit-testing, not only programmatically.

    Every other test calls 'QToolButton.click()', which synthesizes no mouse event at
    all; this one goes through a real press and release at a real position. Which child
    a position resolves to is asserted too, and is the half that matters: the sliding
    highlight is lowered and transparent for mouse events, and a highlight which lost
    either property would eat the clicks on its own segment -- making the control inert
    with every programmatic test still green.
    """
    widget, emitted = segmented
    widget.show()
    app.processEvents()
    for index, value in ((2, "alphabetical"), (0, "acquisition")):
        widget._anim.stop()
        centre = widget._buttons[index].geometry().center()
        assert widget.childAt(centre) is widget._buttons[index], index
        QTest.mouseClick(
            widget._buttons[index],
            Qt.MouseButton.LeftButton,
            pos=widget._buttons[index].rect().center(),
        )
        app.processEvents()
        assert widget.current_index == index
        assert emitted[-1] == value
    assert emitted == ["alphabetical", "acquisition"]
    # the highlight now sits over segment 0 and must still not be what a click hits
    widget._anim.stop()
    widget._highlight.setGeometry(widget._buttons[0].geometry())
    app.processEvents()
    assert widget.childAt(widget._buttons[0].geometry().center()) is widget._buttons[0]
    widget.hide()


def test_empty_items_rejected(app: QApplication) -> None:
    """Test that a control with no segment is refused at construction."""
    with pytest.raises(ValueError, match="at least one segment"):
        AnimatedSegmentedControl([])


def test_accessible_names_are_set(
    segmented: tuple[AnimatedSegmentedControl, list[str]],
) -> None:
    """Test that every segment carries an accessible name and a tooltip."""
    widget, _ = segmented
    for button, (_label, tooltip, _value) in zip(widget._buttons, _ITEMS, strict=True):
        assert button.toolTip() == tooltip
        assert button.accessibleName() == tooltip


def test_no_button_group(
    segmented: tuple[AnimatedSegmentedControl, list[str]],
) -> None:
    """Test that exclusivity is the current index alone.

    A 'QButtonGroup' would be a second exclusivity mechanism fighting '_index'.
    """
    widget, _ = segmented
    assert widget.findChildren(QButtonGroup) == []
    assert all(not button.isCheckable() for button in widget._buttons)
