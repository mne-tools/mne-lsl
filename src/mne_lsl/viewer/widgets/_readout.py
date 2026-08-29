"""Compact read-out label which becomes an inline editor on click."""

from __future__ import annotations

import re
from math import isfinite
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QLabel, QLineEdit, QStackedLayout, QWidget

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtGui import QMouseEvent

# Fixed footprint: a read-out keeps a stepper compact instead of stretching with its
# bar, and a trailing stretch in the bar absorbs the slack. The width fits the longest
# text its consumers format; it is deliberately not derived from any of their value
# ranges, which this widget knows nothing about.
_WIDTH = 62


def _parse_number(text: str) -> float | None:
    """Extract a float from ``text``, e.g. from a unit-suffixed read-out.

    Parameters
    ----------
    text : str
        Text typed in the editor, e.g. ``'12.5s'`` or ``'1.5×'``.

    Returns
    -------
    value : float | None
        The parsed number, or ``None`` if ``text`` carries none or is not finite.

    Notes
    -----
    ``float(text)`` is attempted before the characters are stripped, as the strip eats
    the exponent of a value in scientific notation: ``'1e3'`` would otherwise parse as
    ``13.0`` rather than ``1000.0``.

    A non-finite result is rejected rather than passed on. ``float`` accepts ``'inf'``,
    ``'nan'`` and an overflowing literal such as ``'1e400'``, and a consumer rounding
    such a value to an integer raises ``OverflowError`` or ``ValueError`` from inside a
    Qt slot, where the exception is at best logged and at worst fatal.
    """
    for candidate in (text, re.sub(r"[^0-9.\-]", "", text)):
        try:
            value = float(candidate)
        except ValueError:
            continue
        return value if isfinite(value) else None
    return None


class EditableReadout(QWidget):
    """Read-out label which becomes an inline editor on click.

    Clicking the label swaps it in place for a line edit seeded with the current value;
    committing with Enter or on focus-out parses the text, calls ``commit`` with the
    parsed number and reverts to the label. There is no permanent edit or spin box: the
    label is the field only while editing, keeping the control strip compact.

    Parameters
    ----------
    commit : callable
        Called with the parsed value when an edit is committed.
    tip : str
        Tooltip of the read-out, suffixed with a hint that it is editable.
    parent : QWidget | None
        Parent widget.
    """

    def __init__(
        self, commit: Callable[[float], None], tip: str, parent: QWidget | None = None
    ) -> None:
        """Initialize the read-out."""
        super().__init__(parent)
        self._commit = commit
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)

        # the cursor sits on this widget rather than on the label: the label is
        # transparent to the mouse, and Qt resolves the cursor by hit-testing, thus a
        # cursor set on the label would never be shown.
        self.setCursor(Qt.CursorShape.IBeamCursor)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setToolTip(f"{tip} (click to edit)")
        # a 'QLabel' ignores a mouse press, so a click reaches this widget's
        # 'mousePressEvent' either way; the attribute makes that explicit rather than
        # incidental, and keeps the label out of the hit test entirely.
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._edit = QLineEdit()
        self._edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._edit.editingFinished.connect(self._commit_edit)
        # 'QLineEdit' consumes the key events while it has the focus, thus Escape can
        # only be seen from a filter installed on it.
        self._edit.installEventFilter(self)

        self._stack.addWidget(self._label)  # index 0
        self._stack.addWidget(self._edit)  # index 1
        self.setFixedWidth(_WIDTH)

    @property
    def text(self) -> str:
        """Displayed read-out text."""
        return self._label.text()

    def set_text(self, text: str) -> None:
        """Set the displayed read-out text, abandoning an edit in progress.

        Parameters
        ----------
        text : str
            Formatted value, e.g. ``'12.5s'``.

        Notes
        -----
        The edit is abandoned when the text actually moves, because the owner of the
        value has just changed it and the text being typed is stale. Without this, a
        stepper button clicked during an open edit would leave the editor on top -- a
        ``QToolButton`` takes no focus, so nothing would end the edit -- showing a value
        the owner no longer holds, and the next focus change would commit it, silently
        undoing the stepping.

        An unchanged text leaves the editor alone, so that refreshing a whole bar of
        read-outs does not abandon an edit on one of the others.
        """
        if text == self._label.text():
            return
        self._label.setText(text)
        self.cancel_edit()

    def begin_edit(self) -> None:
        """Swap the label for the editor, seeded with the current read-out text."""
        self._edit.setText(self._label.text())
        self._stack.setCurrentIndex(1)
        self._edit.selectAll()
        self._edit.setFocus()

    def cancel_edit(self) -> None:
        """Revert to the label without committing; idempotent."""
        self._stack.setCurrentIndex(0)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """Begin editing on a click over the read-out label."""
        if self._stack.currentIndex() == 0:
            self.begin_edit()
        super().mousePressEvent(ev)

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        """Abandon the edit on Escape, so a typed value can be discarded.

        Parameters
        ----------
        obj : QObject
            The watched object, i.e. the editor.
        ev : QEvent
            The intercepted event.

        Returns
        -------
        handled : bool
            ``True`` when the event was consumed.

        Notes
        -----
        Without this, Escape left the editor on top and the *next* focus change
        committed the very value the user tried to abandon.
        """
        if (
            obj is self._edit
            and ev.type() == QEvent.Type.KeyPress
            and ev.key() == Qt.Key.Key_Escape
        ):
            self.cancel_edit()
            return True
        return super().eventFilter(obj, ev)

    def _commit_edit(self) -> None:
        """Parse and apply the edited value, and revert to the label.

        Notes
        -----
        Returns early when the editor is not the current widget: Qt emits
        ``editingFinished`` once for the Enter key and once more for the focus-out which
        follows it, and this guard is what makes the second one a no-op. The revert
        happens before the text is parsed for the same reason: ``commit`` refreshes the
        read-out, which would otherwise re-enter this method.
        """
        if self._stack.currentIndex() != 1:
            return  # already reverted, ignore the focus-out echo
        self._stack.setCurrentIndex(0)
        value = _parse_number(self._edit.text())
        if value is not None:
            self._commit(value)
