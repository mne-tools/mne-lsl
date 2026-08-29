"""Segmented control with a highlight which slides on selection."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from qtpy.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QToolButton, QWidget

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qtpy.QtGui import QResizeEvent, QShowEvent

# Slide duration of the highlight, in milliseconds, and its easing.
_SLIDE_MS = 160
# Minimum height of a segment, which also sets the pill radius.
_SEGMENT_H = 22

# Every color is a palette role, so the control follows a theme flip on its own: it
# holds no baked color and therefore needs no 'theme_changed' handler.
_QSS = (
    "#segControl { background: palette(base); border: 1px solid palette(mid); "
    "border-radius: 6px; }"
    "#segControl QToolButton { background: transparent; border: none; "
    "padding: 3px 6px; color: palette(text); }"
    "#segControl QToolButton:hover, #segControl QToolButton:pressed "
    "{ background: transparent; }"
    '#segControl QToolButton[seg_selected="true"] '
    "{ color: palette(highlighted-text); font-weight: 600; }"
)


class AnimatedSegmentedControl(QWidget):
    """Compact segmented control with a highlight which slides on selection.

    Exactly one segment is active at a time. Flat transparent equal-flex buttons sit
    over a single backing highlight widget, animated on its ``geometry``; exclusivity is
    the current index, not a ``QButtonGroup``.

    Parameters
    ----------
    items : sequence of tuple of str
        ``(label, tooltip, value)`` triples, left to right. The first is active.
    parent : QWidget | None
        Parent widget.

    Attributes
    ----------
    changed : Signal
        Emitted with the ``value`` of the newly selected segment.

    Raises
    ------
    ValueError
        If ``items`` is empty. An empty control has no active segment, thus
        :attr:`current_value` could only raise -- from a property a caller has every
        reason to read.
    """

    changed = Signal(str)

    def __init__(
        self, items: Sequence[tuple[str, str, str]], parent: QWidget | None = None
    ) -> None:
        """Initialize the segmented control."""
        super().__init__(parent)
        items = list(items)
        if not items:
            raise ValueError("A segmented control needs at least one segment.")
        self._values = [value for _, _, value in items]
        self._index = 0
        self._buttons: list[QToolButton] = []

        # The moving highlight sits behind the buttons, which are transparent so that it
        # shows through, and never eats their clicks.
        self._highlight = QWidget(self)
        self._highlight.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._highlight.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._highlight.lower()
        self._anim = QPropertyAnimation(self._highlight, b"geometry", self)
        self._anim.setDuration(_SLIDE_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.setObjectName("segControl")
        self.setStyleSheet(_QSS)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        for index, (label, tooltip, _value) in enumerate(items):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tooltip)
            # the labels are abbreviations, thus the tooltip is the accessible name too:
            # a screen reader would otherwise announce 'Abc'.
            button.setAccessibleName(tooltip)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setMinimumHeight(_SEGMENT_H)
            button.clicked.connect(
                lambda _checked=False, idx=index: self._on_clicked(idx)
            )
            layout.addWidget(button, 1)
            self._buttons.append(button)
        self._restyle_buttons()

    @property
    def current_index(self) -> int:
        """Index of the active segment."""
        return self._index

    @property
    def current_value(self) -> str:
        """Value of the active segment."""
        return self._values[self._index]

    def set_index(
        self, index: int, *, animate: bool = False, emit: bool = True
    ) -> None:
        """Activate the segment at ``index``.

        Parameters
        ----------
        index : int
            Segment to activate.
        animate : bool
            If ``True``, slide the highlight; else snap it, e.g. on resize or restore.
        emit : bool
            If ``True``, emit :attr:`changed`. Set to ``False`` to sync silently.

        Raises
        ------
        ValueError
            If ``index`` is not an integer or is out of range. This is a trust boundary:
            a persisted control value is restored through here from user-editable JSON,
            which has no integer/float distinction and no boolean of its own, so
            ``1.0``, ``'1'``, ``None`` and ``True`` all arrive here. An unchecked index
            would raise ``IndexError`` -- or, for a float, ``TypeError`` from the list
            lookup past the range check -- from inside a Qt slot instead.

        Notes
        -----
        Both checks run before ``_index`` is written, so the widget can never be left
        holding an index its own segments do not have. That state poisons it permanently
        rather than failing once: :attr:`current_value` then raises, and so does every
        later ``resizeEvent`` -- out of a virtual Qt itself invokes, which is a process
        abort for anyone who kept the default exception hook.
        """
        # 'bool' is an 'int' subclass, thus 'operator.index' happily turns 'True' into
        # segment 1; a restored 'true' is a corrupt value, not a selection.
        if isinstance(index, bool):
            raise ValueError(f"The segment index must be an integer, got {index!r}.")
        try:
            index = operator.index(index)
        except TypeError:
            raise ValueError(
                f"The segment index must be an integer, got {index!r}."
            ) from None
        if not 0 <= index < len(self._values):
            raise ValueError(
                f"The segment index must be in [0, {len(self._values)}), got {index}."
            )
        self._index = index
        self._restyle_buttons()
        self._position_highlight(animate=animate)
        if emit:
            self.changed.emit(self._values[index])

    def _on_clicked(self, index: int) -> None:
        """Activate the clicked segment, sliding the highlight; re-click is a no-op."""
        if index != self._index:
            self.set_index(index, animate=True)

    def _restyle_buttons(self) -> None:
        """Restyle the button texts through a dynamic-property swap.

        Notes
        -----
        The property swap needs an explicit ``unpolish``/``polish``: Qt does not re-run
        the style-sheet selectors when a dynamic property changes, so the selected text
        weight -- the only selection cue which is not the animation -- would never
        appear.
        """
        for index, button in enumerate(self._buttons):
            button.setProperty("seg_selected", index == self._index)
            button.style().unpolish(button)
            button.style().polish(button)

    def _position_highlight(self, *, animate: bool) -> None:
        """Move the highlight over the active button, sliding it when ``animate``."""
        target = self._buttons[self._index].geometry()
        # The pill radius tracks the height, which is what keeps it round at any compact
        # size rather than at one hardcoded height.
        self._highlight.setStyleSheet(
            f"background: palette(highlight); border-radius: {target.height() // 2}px;"
        )
        if animate and self._highlight.isVisible():
            self._anim.stop()
            self._anim.setStartValue(self._highlight.geometry())
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._highlight.setGeometry(target)
        self._highlight.show()

    def resizeEvent(self, ev: QResizeEvent) -> None:
        """Snap the highlight back onto the active button after a resize.

        Notes
        -----
        Without this the pill detaches from its segment as soon as the panel holding the
        control is resized, since the buttons are equal-flex and move with it.
        """
        super().resizeEvent(ev)
        self._position_highlight(animate=False)

    def showEvent(self, ev: QShowEvent) -> None:
        """Snap the highlight onto the active button once the layout has run.

        Notes
        -----
        Measured redundant with :meth:`resizeEvent`, which the first show also delivers;
        kept because that ordering is not a documented guarantee.
        """
        super().showEvent(ev)
        self._position_highlight(animate=False)
