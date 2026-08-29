"""Control bar of the trace display, the single source of truth for display state.

The controller has no Display page: this bar owns the display state, i.e. the visible
row count, the time window, the amplitude scale, the trace color mode and the channel /
event label toggles. Rows, window and scale are ``− VALUE +`` steppers whose read-out is
click-to-edit; there is no permanent spin box.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QWidget,
)
from superqt import QToggleSwitch

from ...utils.logs import logger
from ..theme import icon
from ..widgets import EditableReadout

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from qtpy.QtGui import QIcon

# Ranges and steps of the three steppers. The scale step is multiplicative, so that a
# notch feels the same at 0.1× as at 10×.
_ROWS = (2, 60)
_ROWS_STEP = 1
# Public, and the only declaration of the selectable time window: the stream buffer is
# derived from the widest window this bar can reach, and a second literal elsewhere is
# how a buffer narrower than the window ships.
WINDOW_RANGE = (0.5, 20.0)
_WINDOW_STEP = 0.5
_SCALE = (0.05, 50.0)
_SCALE_STEP = 1.15
# Combo entries, as (label, published mode). One table rather than an index compared
# against a literal in both directions, which a third mode would silently break.
_COLOR_MODES = (("By channel", "channel"), ("By type", "type"))


class DisplayControls(QToolBar):
    """Button bar owning the display state of one trace display.

    Attributes
    ----------
    rows_changed : Signal
        Emitted with the new visible row count.
    window_changed : Signal
        Emitted with the new time window, in seconds.
    scale_changed : Signal
        Emitted with the new amplitude multiplier.
    color_mode_changed : Signal
        Emitted with the new trace color mode, ``'channel'`` or ``'type'``.
    labels_toggled : Signal
        Emitted when the channel labels are shown or hidden.
    events_toggled : Signal
        Emitted when the event overlays are shown or hidden.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget.

    Notes
    -----
    A :class:`~qtpy.QtWidgets.QToolBar` rather than a plain widget, as the accepted look
    is the toolbar the theme style sheet already skins, and ``addWidget`` /
    ``addSeparator`` then come for free. A toolbar in an ordinary layout, under a parent
    which is not a :class:`~qtpy.QtWidgets.QMainWindow`, is supported.

    Every setter clamps to its range, always refreshes its read-out -- so a typed
    out-of-range value visibly snaps back -- and emits only when the clamped value
    actually changed, as a no-op emission would re-transform every curve of the display.
    """

    rows_changed = Signal(int)
    window_changed = Signal(float)
    scale_changed = Signal(float)
    color_mode_changed = Signal(str)
    labels_toggled = Signal(bool)
    events_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the control bar."""
        super().__init__(parent)
        self._rows = 20
        self._window = 5.0
        self._scale = 1.0
        self._color_mode = "channel"
        self._labels = True
        self._events = True
        # (setter, icon name) pairs replayed by 'retint_icons': a QIcon bakes its color
        # at creation, thus a theme flip needs every icon rebuilt.
        self._icon_setters: list[tuple[Callable[[QIcon], None], str]] = []

        self.setMovable(False)
        self.setIconSize(QSize(16, 16))

        self._rows_readout = self._add_stepper(
            "mdi6.format-line-spacing",
            "Visible rows",
            lambda: self.set_rows(self._rows - _ROWS_STEP),
            lambda: self.set_rows(self._rows + _ROWS_STEP),
            self.set_rows,
        )
        self.addSeparator()
        self._window_readout = self._add_stepper(
            "mdi6.timer-outline",
            "Time window (seconds)",
            lambda: self.set_window(self._window - _WINDOW_STEP),
            lambda: self.set_window(self._window + _WINDOW_STEP),
            self.set_window,
        )
        self.addSeparator()
        self._scale_readout = self._add_stepper(
            "mdi6.arrow-expand-vertical",
            "Amplitude scale",
            lambda: self.step_scale(up=False),
            lambda: self.step_scale(up=True),
            self.set_scale,
        )
        self.addSeparator()

        color_label = QLabel("Color")
        color_label.setContentsMargins(4, 0, 4, 0)
        self.addWidget(color_label)
        self._color_combo = QComboBox()
        self._color_combo.addItems([label for label, _ in _COLOR_MODES])
        self._color_combo.setToolTip("Trace color mode")
        self._color_combo.currentIndexChanged.connect(self._on_color_index)
        self.addWidget(self._color_combo)

        self._labels_switch = QToggleSwitch("Labels")
        self._labels_switch.setChecked(True)
        self._labels_switch.setToolTip("Show channel labels on the left axis")
        self._labels_switch.toggled.connect(self._on_labels_toggled)
        self.addWidget(self._labels_switch)

        self._events_switch = QToggleSwitch("Events")
        self._events_switch.setChecked(True)
        self._events_switch.setToolTip("Show stim-event overlays")
        self._events_switch.toggled.connect(self._on_events_toggled)
        self.addWidget(self._events_switch)

        # trailing stretch soaking up the extra width, so the cluster stays compact and
        # left-packed instead of the steppers spreading out.
        trailing = QWidget()
        trailing.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.addWidget(trailing)

        self.retint_icons()  # bake the initial icon colors from the active theme
        self._refresh_readouts()

    # -- construction ------------------------------------------------------------------
    def _add_stepper(
        self,
        prefix_icon: str,
        tip: str,
        on_minus: Callable[[], None],
        on_plus: Callable[[], None],
        commit: Callable[[float], None],
    ) -> EditableReadout:
        """Add one ``[icon] − <read-out> +`` stepper and return its read-out.

        Parameters
        ----------
        prefix_icon : str
            QtAwesome name of the icon prefixing the stepper.
        tip : str
            Tooltip base, suffixed per element.
        on_minus : callable
            Called with no argument when the ``−`` button is clicked.
        on_plus : callable
            Called with no argument when the ``+`` button is clicked.
        commit : callable
            Called with the parsed value when the read-out is edited.

        Returns
        -------
        readout : EditableReadout
            The click-to-edit read-out between the two buttons.
        """
        prefix = QLabel()
        prefix.setToolTip(tip)
        prefix.setContentsMargins(4, 0, 2, 0)
        self._icon_setters.append(
            (lambda ic, p=prefix: p.setPixmap(ic.pixmap(16, 16)), prefix_icon)
        )
        self.addWidget(prefix)

        minus = QToolButton()
        minus.setAutoRaise(True)
        minus.setToolTip(f"{tip}: decrease")
        minus.clicked.connect(lambda: on_minus())
        self._icon_setters.append((lambda ic, b=minus: b.setIcon(ic), "mdi6.minus"))
        self.addWidget(minus)

        readout = EditableReadout(commit, tip)
        self.addWidget(readout)

        plus = QToolButton()
        plus.setAutoRaise(True)
        plus.setToolTip(f"{tip}: increase")
        plus.clicked.connect(lambda: on_plus())
        self._icon_setters.append((lambda ic, b=plus: b.setIcon(ic), "mdi6.plus"))
        self.addWidget(plus)
        return readout

    def _refresh_readouts(self) -> None:
        """Sync the three read-outs to the current state."""
        self._rows_readout.set_text(f"{self._rows:g}")
        self._window_readout.set_text(f"{self._window:g}s")
        self._scale_readout.set_text(f"{self._scale:g}×")

    # -- state -------------------------------------------------------------------------
    @property
    def state(self) -> dict[str, Any]:
        """Current display state, one entry per control."""
        return {
            "rows": self._rows,
            "window": self._window,
            "scale": self._scale,
            "color_mode": self._color_mode,
            "labels": self._labels,
            "events": self._events,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore the display state, e.g. from a configuration.

        Parameters
        ----------
        state : dict
            A mapping shaped like :attr:`state`. Unknown keys are ignored, as is any
            value of the wrong type.

        Notes
        -----
        Every value is applied through the same path a user click takes, thus the
        matching signal is emitted for each key which actually changed. That is what
        makes a restore reach the display without any additional plumbing.

        Every value is validated, because this is a trust boundary: the state comes from
        a configuration file the user can edit by hand. An unusable value is logged and
        skipped rather than raising, which would abandon the restore part-way and leave
        the bar holding a mix of the saved and the previous state. Note ``bool`` is
        rejected for a number -- it is an ``int`` subclass, so ``True`` would otherwise
        clamp to a row count.
        """
        for key, setter in (
            ("rows", self.set_rows),
            ("window", self.set_window),
            ("scale", self.set_scale),
        ):
            if key not in state:
                continue
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                logger.warning(
                    "Ignoring the display state %r: expected a number, got %r.",
                    key,
                    value,
                )
            elif not isfinite(value):
                logger.warning(
                    "Ignoring the display state %r: %r is not finite.", key, value
                )
            else:
                setter(value)
        # the combo and the two switches publish through their own slots, and Qt only
        # emits when the value actually changes, thus a restore cannot emit twice.
        if "color_mode" in state:
            modes = [mode for _, mode in _COLOR_MODES]
            if state["color_mode"] in modes:
                self._color_combo.setCurrentIndex(modes.index(state["color_mode"]))
            else:
                logger.warning(
                    "Ignoring the display state 'color_mode': %r is not one of %s.",
                    state["color_mode"],
                    modes,
                )
        for key, switch in (
            ("labels", self._labels_switch),
            ("events", self._events_switch),
        ):
            if key not in state:
                continue
            if isinstance(state[key], bool):
                switch.setChecked(state[key])
            else:
                logger.warning(
                    "Ignoring the display state %r: expected a bool, got %r.",
                    key,
                    state[key],
                )

    def set_rows(self, value: int) -> None:
        """Set the visible row count and emit :attr:`rows_changed`.

        Parameters
        ----------
        value : int
            Number of rows, clamped to ``[2, 60]``.
        """
        rows = int(min(max(_ROWS[0], round(value)), _ROWS[1]))
        changed = rows != self._rows
        self._rows = rows
        self._refresh_readouts()
        if changed:
            self.rows_changed.emit(rows)

    def set_window(self, value: float) -> None:
        """Set the time window in seconds and emit :attr:`window_changed`.

        Parameters
        ----------
        value : float
            Window duration in seconds, clamped to :data:`WINDOW_RANGE` -- named and not
            spelled out, as the buffer size of every connection is derived from the same
            bound and a second spelling of it is what the shared name exists to prevent.
        """
        window = float(min(max(WINDOW_RANGE[0], value), WINDOW_RANGE[1]))
        changed = window != self._window
        self._window = window
        self._refresh_readouts()
        if changed:
            self.window_changed.emit(window)

    def set_scale(self, value: float) -> None:
        """Set the amplitude multiplier and emit :attr:`scale_changed`.

        Parameters
        ----------
        value : float
            Amplitude multiplier, clamped to ``[0.05, 50]``.
        """
        # Quantized to the precision the read-out shows, so that the stored value and
        # the displayed one never disagree. Without this, the multiplicative step
        # produced values like 2.0113571874999994 shown as '2.01136×', and committing
        # that very text back counted as a change -- re-transforming every curve for a
        # value the user did not alter.
        scale = float(f"{min(max(_SCALE[0], value), _SCALE[1]):g}")
        changed = scale != self._scale
        self._scale = scale
        self._refresh_readouts()
        if changed:
            self.scale_changed.emit(scale)

    def step_scale(self, up: bool) -> None:
        """Step the amplitude multiplier by one multiplicative notch.

        Parameters
        ----------
        up : bool
            Whether to step up; a step down divides by the same factor.

        Notes
        -----
        The step lives here rather than in the display, so that the bar remains the one
        place which knows the step size and the one place which holds the value the next
        step is derived from.
        """
        self.set_scale(self._scale * (_SCALE_STEP if up else 1.0 / _SCALE_STEP))

    def retint_icons(self) -> None:
        """Rebuild the bar icons after a theme flip, as a ``QIcon`` bakes its color."""
        for setter, name in self._icon_setters:
            setter(icon(name))

    # -- slots -------------------------------------------------------------------------
    def _on_color_index(self, index: int) -> None:
        """Publish the color mode of the combo index."""
        self._color_mode = _COLOR_MODES[index][1]
        self.color_mode_changed.emit(self._color_mode)

    def _on_labels_toggled(self, checked: bool) -> None:
        """Publish the channel-label toggle."""
        self._labels = bool(checked)
        self.labels_toggled.emit(self._labels)

    def _on_events_toggled(self, checked: bool) -> None:
        """Publish the event-overlay toggle."""
        self._events = bool(checked)
        self.events_toggled.emit(self._events)
