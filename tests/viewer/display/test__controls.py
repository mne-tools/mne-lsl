from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from qtpy.QtWidgets import QAbstractSpinBox, QLabel, QToolBar, QToolButton

from mne_lsl.viewer.display import DisplayControls
from mne_lsl.viewer.widgets import EditableReadout

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Any

    from qtpy.QtGui import QImage
    from qtpy.QtWidgets import QApplication, QWidget

    from mne_lsl.viewer.theme import ThemeController

_KEYS = ("rows", "window", "scale", "color_mode", "labels", "events")


@pytest.fixture
def controls(app: QApplication) -> Generator[DisplayControls]:
    """Yield a control bar, closed afterwards."""
    bar = DisplayControls()
    yield bar
    bar.close()
    bar.deleteLater()
    app.processEvents()


@pytest.fixture
def emitted(controls: DisplayControls) -> dict[str, list]:
    """Return the values emitted per signal, recorded from the bar."""
    received: dict[str, list] = {key: [] for key in _KEYS}
    controls.rows_changed.connect(received["rows"].append)
    controls.window_changed.connect(received["window"].append)
    controls.scale_changed.connect(received["scale"].append)
    controls.color_mode_changed.connect(received["color_mode"].append)
    controls.labels_toggled.connect(received["labels"].append)
    controls.events_toggled.connect(received["events"].append)
    return received


def _icon_images(controls: DisplayControls) -> list[QImage]:
    """Return the rendered image of every icon on the bar, stepper prefixes included.

    The prefix icons live on a 'QLabel' pixmap and the stepper icons on a 'QToolButton',
    thus both have to be collected or the three prefixes -- a third of the registry --
    stay outside every assertion. 'qt_toolbar_ext_button' is the overflow button
    'QToolBar' builds itself, and its arrow comes from the style, not from the bar.
    """
    images = [
        button.icon().pixmap(16, 16).toImage()
        for button in controls.findChildren(QToolButton)
        if button.objectName() != "qt_toolbar_ext_button"
    ]
    images += [
        label.pixmap().toImage()
        for label in controls.findChildren(QLabel)
        if not label.pixmap().isNull()
    ]
    return images


def test_is_a_toolbar(controls: DisplayControls) -> None:
    """Test that the bar is the toolbar the theme style sheet skins."""
    assert isinstance(controls, QToolBar)
    assert not controls.isMovable()


def test_default_state(controls: DisplayControls) -> None:
    """Test the documented defaults of every control."""
    assert controls.state == {
        "rows": 20,
        "window": 5.0,
        "scale": 1.0,
        "color_mode": "channel",
        "labels": True,
        "events": True,
    }


def test_readout_formats(controls: DisplayControls) -> None:
    """Test the read-out formatting, unit suffixes included."""
    assert controls._rows_readout.text == "20"
    assert controls._window_readout.text == "5s"
    assert controls._scale_readout.text == "1×"


@pytest.mark.parametrize(
    ("tip", "key", "expected"),
    [
        ("Visible rows: increase", "rows", 21),
        ("Visible rows: decrease", "rows", 19),
        ("Time window (seconds): increase", "window", 5.5),
        ("Time window (seconds): decrease", "window", 4.5),
    ],
)
def test_stepper_buttons_emit_once(
    controls: DisplayControls,
    emitted: dict[str, list],
    tool_button: Callable[[QWidget, str], QToolButton],
    tip: str,
    key: str,
    expected: float,
) -> None:
    """Test that a stepper click emits its signal once, with the new value."""
    tool_button(controls, tip).click()
    assert emitted[key] == [expected]
    assert controls.state[key] == expected


def test_scale_stepper_is_multiplicative(
    controls: DisplayControls,
    emitted: dict[str, list],
    tool_button: Callable[[QWidget, str], QToolButton],
) -> None:
    """Test that the amplitude stepper multiplies rather than adds."""
    tool_button(controls, "Amplitude scale: increase").click()
    assert emitted["scale"] == [pytest.approx(1.15)]
    tool_button(controls, "Amplitude scale: decrease").click()
    assert controls.state["scale"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("setter", "value", "key", "clamped"),
    [
        ("set_rows", 1, "rows", 2),
        ("set_rows", 1000, "rows", 60),
        ("set_window", 0.1, "window", 0.5),
        ("set_window", 100.0, "window", 20.0),
        ("set_scale", 0.001, "scale", 0.05),
        ("set_scale", 1000.0, "scale", 50.0),
    ],
)
def test_clamping(
    controls: DisplayControls,
    emitted: dict[str, list],
    setter: str,
    value: float,
    key: str,
    clamped: float,
) -> None:
    """Test that an out-of-range value is clamped, emitted once and read back."""
    getattr(controls, setter)(value)
    assert controls.state[key] == clamped
    assert emitted[key] == [clamped]


def test_clamping_at_bounds_does_not_emit(
    controls: DisplayControls,
    emitted: dict[str, list],
    tool_button: Callable[[QWidget, str], QToolButton],
) -> None:
    """Test that a value already at a bound emits nothing at all."""
    controls.set_rows(60)
    emitted["rows"].clear()
    tool_button(controls, "Visible rows: increase").click()
    assert emitted["rows"] == []
    assert controls.state["rows"] == 60


@pytest.mark.parametrize(
    ("setter", "attribute", "key", "bound", "snapped"),
    [
        ("set_rows", "_rows_readout", "rows", 60, "60"),
        ("set_window", "_window_readout", "window", 20.0, "20s"),
        ("set_scale", "_scale_readout", "scale", 50.0, "50×"),
    ],
)
def test_out_of_range_edit_snaps_the_readout_back(
    controls: DisplayControls,
    emitted: dict[str, list],
    finish_edit: Callable[[EditableReadout, str], None],
    setter: str,
    attribute: str,
    key: str,
    bound: float,
    snapped: str,
) -> None:
    """Test that a clamped value refreshes its read-out even though nothing changed.

    Every setter refreshes unconditionally, which is what makes a read-out and the value
    it displays converge again. A refresh gated on the emission cannot: a clamped edit
    emits nothing, so the stale text would stay on screen. The read-out is put out of
    sync first, because that is the only state in which the mechanism is observable at
    all -- the editor seeds itself *from* the label and never writes it, thus the label
    of a synchronized read-out already reads correctly, refreshed or not.
    """
    getattr(controls, setter)(bound)
    readout = getattr(controls, attribute)
    readout.set_text("999")
    assert readout.text == "999"
    emitted[key].clear()
    finish_edit(readout, "999")
    assert emitted[key] == []  # clamped onto the bound it already sat on
    assert controls.state[key] == bound
    assert readout.text == snapped


def test_click_to_edit_applies(
    controls: DisplayControls,
    emitted: dict[str, list],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that editing a read-out applies the value and reverts to the label."""
    finish_edit(controls._rows_readout, "24")
    assert emitted["rows"] == [24]
    assert controls.state["rows"] == 24
    assert controls._rows_readout.text == "24"
    assert controls._rows_readout._stack.currentIndex() == 0


def test_edit_rejects_garbage(
    controls: DisplayControls,
    emitted: dict[str, list],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that an unparsable edit leaves the value untouched."""
    finish_edit(controls._window_readout, "later")
    assert emitted["window"] == []
    assert controls.state["window"] == 5.0
    assert controls._window_readout.text == "5s"


def test_set_scale_is_idempotent_with_its_read_out(
    controls: DisplayControls,
    emitted: dict[str, list],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that re-committing the displayed scale emits nothing.

    The multiplicative step produced values like 2.0113571874999994, shown as
    '2.01136×'; committing that very text back counted as a change and re-transformed
    every curve of the display for a value the user never altered. The stored value is
    therefore quantized to the precision the read-out shows.
    """
    for _ in range(5):
        controls.step_scale(up=True)
    text = controls._scale_readout.text
    assert text not in ("1×", "50×")  # a value the read-out really had to round
    emitted["scale"].clear()
    finish_edit(controls._scale_readout, text)
    assert emitted["scale"] == []
    assert controls._scale_readout.text == text


def test_color_mode_signal(controls: DisplayControls, emitted: dict[str, list]) -> None:
    """Test that the color combo publishes the mode string, not its index."""
    controls._color_combo.setCurrentIndex(1)
    assert emitted["color_mode"] == ["type"]
    assert controls.state["color_mode"] == "type"
    controls._color_combo.setCurrentIndex(0)
    assert emitted["color_mode"] == ["type", "channel"]


def test_labels_events_toggles(
    controls: DisplayControls, emitted: dict[str, list]
) -> None:
    """Test that both toggle switches publish their new state."""
    controls._labels_switch.setChecked(False)
    controls._events_switch.setChecked(False)
    assert emitted["labels"] == [False]
    assert emitted["events"] == [False]
    assert controls.state["labels"] is False
    assert controls.state["events"] is False


def test_state_roundtrip(controls: DisplayControls, emitted: dict[str, list]) -> None:
    """Test that a saved state restores every control and emits once per changed key."""
    target = {
        "rows": 8,
        "window": 12.5,
        "scale": 2.5,
        "color_mode": "type",
        "labels": False,
        "events": False,
    }
    controls.set_state(target)
    assert controls.state == target
    for key, value in target.items():
        assert emitted[key] == [value], key
    # re-applying the same state is a no-op: no curve pool is rebuilt for nothing.
    for key in _KEYS:
        emitted[key].clear()
    controls.set_state(target)
    assert all(len(values) == 0 for values in emitted.values()), emitted


def test_set_state_ignores_unknown_keys(
    controls: DisplayControls, emitted: dict[str, list]
) -> None:
    """Test that an unknown key is ignored rather than raising."""
    controls.set_state({"rows": 30, "bogus": 1})
    assert controls.state["rows"] == 30
    assert emitted["rows"] == [30]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rows", None),
        ("rows", "24"),
        ("rows", True),  # an 'int' subclass, thus it would otherwise clamp to a count
        ("rows", float("inf")),
        ("window", float("nan")),
        ("window", []),
        ("scale", True),
        ("scale", float("-inf")),
        ("color_mode", "rainbow"),
        ("color_mode", None),
        ("labels", 1),
        ("events", "yes"),
    ],
)
def test_set_state_skips_an_unusable_value(
    controls: DisplayControls,
    emitted: dict[str, list],
    caplog: pytest.LogCaptureFixture,
    key: str,
    value: Any,
) -> None:
    """Test that a hand-edited state value is logged and skipped, never raising.

    This is a trust boundary: the state comes from a configuration file the user can
    edit by hand. Raising would abandon the restore part-way and leave the bar holding a
    mix of the saved and the previous state, thus every *other* key of the same mapping
    has to be applied all the same. A non-finite number is rejected here rather than
    downstream, where 'round' raises from inside a Qt slot.
    """
    caplog.set_level(logging.WARNING, logger="mne_lsl")
    before = controls.state
    valid = ("window", 7.5) if key == "rows" else ("rows", 8)
    controls.set_state({key: value, valid[0]: valid[1]})  # must not raise
    assert controls.state[key] == before[key]
    assert emitted[key] == []
    assert controls.state[valid[0]] == valid[1]  # the valid key still applied
    assert emitted[valid[0]] == [valid[1]]
    assert "Ignoring the display state" in caplog.text
    assert key in caplog.text


def test_no_spin_boxes(controls: DisplayControls) -> None:
    """Test that the accepted look has no spin box: the read-outs are the fields."""
    assert controls.findChildren(QAbstractSpinBox) == []
    assert len(controls.findChildren(EditableReadout)) == 3


def test_retint_icons_changes_every_pixmap(
    controls: DisplayControls, app: QApplication, controller: ThemeController
) -> None:
    """Test that a retint really rebuilds *every* registered icon, colors included.

    A ``QIcon`` bakes its color at creation and 'qtawesome' memoizes on the icon name
    only, thus asserting that the method was called would prove nothing: the rendered
    pixmap itself has to change. Every icon of the bar is checked, not one of the nine:
    dropping the three stepper prefixes from the registry leaves them in the colors of
    the previous mode, and a single-button assertion cannot see that.

    The theme is flipped through the 'controller' fixture rather than 'apply_theme'
    directly, which is what restores the controller state afterwards -- this is the one
    theme-mutating test which used to bypass the suite's only guard.
    """
    assert len(controls._icon_setters) == 9  # 3 prefixes + 3 x (minus, plus)
    controller.install(app, "light")
    controls.retint_icons()
    light = _icon_images(controls)
    assert len(light) == 9
    assert all(not image.isNull() for image in light)
    controller.set_mode("dark")
    controls.retint_icons()
    dark = _icon_images(controls)
    for index, (before, after) in enumerate(zip(light, dark, strict=True)):
        assert before != after, index


def test_color_label_present(controls: DisplayControls) -> None:
    """Test that the color combo carries its descriptor label on the bar."""
    assert "Color" in {label.text() for label in controls.findChildren(QLabel)}
