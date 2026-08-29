from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QMouseEvent

from mne_lsl.viewer.widgets import EditableReadout
from mne_lsl.viewer.widgets._readout import _parse_number

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pytestqt.qtbot import QtBot
    from qtpy.QtWidgets import QApplication


@pytest.fixture
def readout(app: QApplication) -> Generator[tuple[EditableReadout, list[float]]]:
    """Yield a read-out and the list of values its commit callback received."""
    committed: list[float] = []
    widget = EditableReadout(committed.append, "Visible rows")
    widget.set_text("20")
    yield widget, committed
    widget.close()
    widget.deleteLater()
    app.processEvents()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12", 12.0),
        ("12.5s", 12.5),
        ("1.5×", 1.5),
        ("-3", -3.0),
        ("1e3", 1000.0),  # parsed before the strip, which would eat the exponent
        ("", None),
        ("abc", None),
        ("s", None),
        # 'float' accepts all four; a consumer rounding one raises 'OverflowError' or
        # 'ValueError' from inside a Qt slot, where it is at best logged.
        ("inf", None),
        ("-inf", None),
        ("nan", None),
        ("1e400", None),
    ],
)
def test_parse_number(text: str, expected: float | None) -> None:
    """Test that a read-out value parses, unit suffix included, and rejects the rest."""
    assert _parse_number(text) == expected


def test_set_text_and_text_property(
    readout: tuple[EditableReadout, list[float]],
) -> None:
    """Test that the read-out text is what was set."""
    widget, _ = readout
    assert widget.text == "20"
    widget.set_text("5s")
    assert widget.text == "5s"


def test_click_begins_edit(readout: tuple[EditableReadout, list[float]]) -> None:
    """Test that a click over the label swaps it for the editor, seeded."""
    widget, _ = readout
    assert widget._stack.currentIndex() == 0
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(5.0, 5.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(event)
    assert widget._stack.currentIndex() == 1
    assert widget._edit.text() == "20"


def test_commit_parses_a_suffixed_value(
    readout: tuple[EditableReadout, list[float]],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that a committed value reaches the callback, unit suffix included.

    One commit test rather than one per entry of the parse table: the table itself is
    asserted by 'test_parse_number', and what this adds is the widget path from the
    editor to the callback -- the same path whatever the text.
    """
    widget, committed = readout
    finish_edit(widget, "12.5s")
    assert committed == [12.5]


def test_commit_rejects_garbage(
    readout: tuple[EditableReadout, list[float]],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that an unparsable value is dropped and the read-out is untouched."""
    widget, committed = readout
    finish_edit(widget, "not a number")
    assert committed == []
    assert widget.text == "20"


def test_reverts_to_label(
    readout: tuple[EditableReadout, list[float]],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that the editor is transient: no permanent edit box remains."""
    widget, _ = readout
    finish_edit(widget, "24")
    assert widget._stack.currentIndex() == 0


def test_focus_out_after_enter_commits_once(
    readout: tuple[EditableReadout, list[float]],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that the focus-out echoing an Enter commit does not commit twice."""
    widget, committed = readout
    finish_edit(widget, "24")
    widget._edit.editingFinished.emit()  # the focus-out which follows the Enter key
    assert committed == [24.0]


def test_commit_callback_may_refresh_the_text(
    readout: tuple[EditableReadout, list[float]],
    finish_edit: Callable[[EditableReadout, str], None],
) -> None:
    """Test that a callback refreshing the read-out is applied once, not re-entered."""
    widget, _ = readout
    calls: list[float] = []

    def _commit(value: float) -> None:
        calls.append(value)
        widget.set_text(f"{value:g}")  # what a stepper does on every commit

    widget._commit = _commit
    finish_edit(widget, "24")
    assert calls == [24.0]
    assert widget.text == "24"


def test_set_text_abandons_an_open_edit(
    readout: tuple[EditableReadout, list[float]],
) -> None:
    """Test that a text change while editing drops the editor and its stale value.

    A stepper button takes no focus, thus a click during an open edit would otherwise
    leave the editor on top showing a value the owner no longer holds -- and the next
    focus change would commit it, silently undoing the stepping.
    """
    widget, committed = readout
    widget.begin_edit()
    widget._edit.setText("999")
    widget.set_text("21")  # what a stepper click does
    assert widget._stack.currentIndex() == 0
    assert widget.text == "21"
    widget._edit.editingFinished.emit()  # the focus change which eventually follows
    assert committed == []


def test_set_text_keeps_an_unchanged_text_editing(
    readout: tuple[EditableReadout, list[float]],
) -> None:
    """Test that refreshing a read-out with the same text leaves the edit alone.

    A bar refreshes all of its read-outs on every commit, thus an unconditional abandon
    would kill an edit open on one of the *other* read-outs.
    """
    widget, _ = readout
    widget.begin_edit()
    widget._edit.setText("999")
    widget.set_text("20")  # the text it already shows
    assert widget._stack.currentIndex() == 1
    assert widget._edit.text() == "999"


def test_escape_abandons_the_edit(
    qtbot: QtBot, readout: tuple[EditableReadout, list[float]]
) -> None:
    """Test that Escape discards the typed value instead of arming its commit.

    'QLineEdit' consumes the key events while focused, thus Escape is only visible from
    the event filter; without it, Escape left the editor on top and the *next* focus
    change committed the very value the user tried to abandon.
    """
    widget, committed = readout
    widget.begin_edit()
    widget._edit.setText("999")
    qtbot.keyClick(widget._edit, Qt.Key.Key_Escape)
    assert widget._stack.currentIndex() == 0
    assert widget.text == "20"
    widget._edit.editingFinished.emit()  # the focus change which follows
    assert committed == []
    assert widget.text == "20"
