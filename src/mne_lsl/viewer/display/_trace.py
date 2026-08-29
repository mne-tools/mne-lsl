"""Multichannel trace display.

Accepted rendering design: a fixed pool of persistent :class:`pyqtgraph.PlotDataItem`
curves stacked with ``setPos`` and amplitude-scaled with a ``QTransform``, so changing
the scale never rewrites the sample arrays. A curve is only ever reassigned to a channel
while it sits in the off-screen overscan band, which gives smooth fractional row
scrolling without pop-in. The window is drawn against relative time on a fixed ``0 → W``
axis, and the acquisition and render cadences stay independent.

Three index spaces meet in this module and never mix:

- **acquisition index**, the channel identity, which ``info.ch_names``, the buffer
  and ``picks`` speak, and which seeds the trace color;
- **presentation index**, the row order of the Channels panel, hidden channels included,
  which lives entirely in the controller and never reaches this module;
- **visible-row index**, what this display stacks, scrolls and labels.

The whole indirection is one list, ``self._rows``, mapping a visible row to its
acquisition index -- and it doubles as the ``get_data`` picks argument, so that numpy
performs the translation and the render loop needs no map at all.
"""

from __future__ import annotations

from math import ceil, floor
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor, QTransform
from qtpy.QtWidgets import (
    QGraphicsRectItem,
    QHBoxLayout,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from ..theme import (
    follow_theme,
    plot_colors,
    theme_controller,
    tokens,
    trace_color,
    type_color,
)
from ._axis import ChannelAxis, TraceViewBox
from ._controls import DisplayControls

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qtpy.QtGui import QCloseEvent, QKeyEvent, QPen, QShowEvent

    from ...stream import BaseStream

# Rows kept off-screen above and below the visible band, where a curve may be
# (re)assigned to another channel.
_OVERSCAN = 4
# Render clock period, in milliseconds; independent of the acquisition cadence.
_RENDER_MS = 33
# Rows scrolled per wheel notch.
_SCROLL_ROWS_PER_NOTCH = 3.0
# Scrollbar integer units per row, i.e. the sub-row resolution of the scrollbar.
_SB_RES = 100
# Resident event overlays, shown and hidden rather than added and removed.
_EVENT_POOL = 32
# Rows of headroom reserved above the top row for the event labels.
_EVENT_LABEL_MARGIN = 0.9

# Fraction of a row the expected peak-to-peak amplitude of a channel fills.
_ROW_FILL = 0.6
# Expected peak-to-peak amplitude per channel type, in SI base units. Combined with the
# 'unit_mul' the stream declares, this is what makes a channel legible whatever the
# sender pushes: MNE stores EEG in volts, so a replayed recording delivers ~5e-5 and a
# fixed gain would draw a flat line.
# ponytail: first-guess ranges, they want the same eyeball pass as the 'bad' token. The
# deferred per-type user control multiplies one more factor into the same expression.
# A type absent from this table falls back to a range of 1.0, i.e. becomes unit-blind:
# a channel declared in µV would then draw dead flat. The intracranial and physiological
# types are therefore listed even though the Channels page is what can produce them.
_RANGE_SI = {
    "eeg": 50e-6,
    "seeg": 50e-6,
    "dbs": 50e-6,
    "ecog": 50e-6,
    "eog": 200e-6,
    "ecg": 2e-3,
    "emg": 1e-3,
    "bio": 1e-3,
    "resp": 1.0,
    "misc": 1.0,
}
# Types which are not a physical quantity: their range is in native units as pushed. A
# small trigger code therefore draws a nearly flat trace, deliberately -- the event
# overlay is the read-out for a stim channel.
_RANGE_NATIVE = {"stim": 255.0}


class TraceDisplay(QWidget):
    """Trace display of one stream document: a control bar over a scrolling plot.

    Parameters
    ----------
    stream : BaseStream
        The connected stream which is polled by the render clock.
    parent : QWidget | None
        Parent widget.

    Notes
    -----
    The display starts in acquisition order with every channel visible; the presentation
    order and the visibility arrive from the outside through
    :meth:`TraceDisplay.set_channel_layout`, and the channel metadata is read live from
    ``stream.info``. The display therefore holds no channel state of its own, and does
    not know the channel model exists.

    The stream is borrowed, never owned: it is polled and never disconnected here.

    Attributes
    ----------
    polled : Signal
        Emitted after every render tick, whether or not the tick drew anything. Bare, on
        purpose: what the tick observed about the *connection* is read by the consumer
        off the stream and off :attr:`TraceDisplay.last_timestamp`, so that no
        connection semantics enter this widget's signal signature.
    """

    polled = Signal()

    def __init__(self, stream: BaseStream, parent: QWidget | None = None) -> None:
        """Initialize the display."""
        super().__init__(parent)
        self._stream = stream
        self._names: list[str] = []
        self._types: list[str] = []
        self._gain: list[float] = []
        self._bads: set[str] = set()
        self._event_acq: list[int] = []
        self._read_metadata()

        self._top = 0.0
        # A concrete 'light' / 'dark', refreshed only on a theme change: resolving
        # 'auto' may shell out to the OS, and the render path reads this per curve.
        self._mode = theme_controller.mode

        self._rows = list(range(self.n_channels))
        self._picks: list[int] = []
        self._event_pos: list[int] = []
        self._rebuild_picks()

        self._pool: list[pg.PlotDataItem] = []
        self._free: list[pg.PlotDataItem] = []
        self._assigned: dict[int, pg.PlotDataItem] = {}
        # The window the last poll returned, as '(picks, data, relative times)', or
        # 'None' before the first one. What a stopped clock repaints, see '_redraw'.
        # ponytail: it doubles the peak footprint of one window, which is ~10 MB for 256
        # channels over a 5 s window at 1 kHz. The upgrade is to retain only the banded
        # rows, which is the same change as narrowing the fetch itself.
        self._frame: tuple[list[int], np.ndarray, np.ndarray] | None = None
        # Newest timestamp of the last window fetched, or 'None' before the first fetch.
        self._last_ts: float | None = None
        # Whether the stream may be touched at all, see 'suspend'.
        self._suspended = False

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_ui()
        # Render-loop caches of the control-bar state, seeded from the bar itself rather
        # than from a second set of literals. The bar is the single source of truth: a
        # setter emits only when its value changed, so a default declared twice would
        # silently leave the display running on the stale one -- invisibly so for the
        # event toggle, which would read off while the overlays kept drawing.
        # These are written here and by the signal handlers below, nowhere else.
        state = self._controls.state
        self._n_visible = state["rows"]
        self._winsize = state["window"]
        self._amp_mult = state["scale"]
        self._color_mode = state["color_mode"]
        self._events_on = state["events"]
        self._axis.setStyle(showValues=state["labels"])
        self._build_event_lines()
        self._build_pool()
        self._apply_scroll()

        self._timer = QTimer(self)
        self._timer.setInterval(_RENDER_MS)
        self._timer.timeout.connect(self._render)  # started by 'start()', not here

        # One connection per display: the pyqtgraph items and the bar icons do not read
        # the QPalette, thus they need an explicit update on every theme flip.
        self._following_theme = False
        self._was_running = False  # whether 'closeEvent' stopped a running clock
        follow_theme(self, self._on_theme_changed, True)

    # -- construction ------------------------------------------------------------------
    def _read_metadata(self) -> None:
        """Read the channel names, types, units and bads from the stream.

        Notes
        -----
        The picks are an explicit integer range rather than the ``None`` default because
        resolving ``None`` costs ~140x more: measured on the resolution itself, 341 µs
        against 2.5 µs at 256 channels, and this runs again on every metadata change.
        ``strict=True`` on the ``zip`` is the check that the two lists still agree in
        length, so a type and a unit can never be paired across channels.
        """
        if not self._stream.connected:
            return
        info = self._stream.info
        self._names = list(info.ch_names)
        self._bads = set(info["bads"])
        picks = list(range(len(self._names)))
        self._types = list(self._stream.get_channel_types(picks=picks))
        units = self._stream.get_channel_units(picks=picks)
        self._gain = [
            _ROW_FILL / _range_native(ch_type, unit_mul)
            for ch_type, (_, unit_mul) in zip(self._types, units, strict=True)
        ]
        # recomputed here and not cached from the construction: a type change can create
        # or remove a stim channel.
        self._event_acq = [
            idx for idx, ch_type in enumerate(self._types) if ch_type == "stim"
        ]

    def _build_ui(self) -> None:
        """Build the control bar, the plot and the vertical scrollbar."""
        self._axis = ChannelAxis(self)
        self._vb = TraceViewBox(self)
        self._plot = pg.PlotWidget(viewBox=self._vb, axisItems={"left": self._axis})
        self._plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # this widget owns the keys
        plot_item = self._plot.getPlotItem()
        plot_item.hideButtons()
        plot_item.setMenuEnabled(False)
        plot_item.setLabel("bottom", "time", units="s")

        self._scroll = QScrollBar(Qt.Orientation.Vertical)
        self._scroll.valueChanged.connect(self._on_scrollbar)

        self._controls = DisplayControls()
        self._controls.rows_changed.connect(self._on_rows)
        self._controls.window_changed.connect(self._on_window)
        self._controls.scale_changed.connect(self._on_scale)
        self._controls.color_mode_changed.connect(self._on_color_mode)
        self._controls.labels_toggled.connect(self._on_labels)
        self._controls.events_toggled.connect(self._on_events)

        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.addWidget(self._plot, 1)
        plot_row.addWidget(self._scroll)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._controls)
        layout.addLayout(plot_row, 1)

    def _build_event_lines(self) -> None:
        """Create the resident event overlays: the mask band, lines and labels.

        Notes
        -----
        An event is a vertical line spanning the plot height plus a value label in
        the headroom reserved above the top row. An opaque band masks the overscan
        which would otherwise peek into that headroom. Every overlay is resident and
        toggles its visibility, rather than being added and removed.

        The items are created bare and colored by :meth:`_style_overlays`, which is the
        one place their colors are set -- here and again on a theme flip, so that adding
        an overlay cannot leave it readable in one mode only.
        """
        # drawn above the curves and below the lines and labels; its y is set by
        # '_apply_scroll'.
        self._headband = QGraphicsRectItem()
        self._headband.setPen(pg.mkPen(None))
        self._headband.setZValue(5)
        self._vb.addItem(self._headband, ignoreBounds=True)

        self._event_lines: list[pg.InfiniteLine] = []
        self._event_labels: list[pg.TextItem] = []
        for _ in range(_EVENT_POOL):
            line = pg.InfiniteLine(angle=90, movable=False)
            line.setZValue(10)
            line.setVisible(False)
            self._vb.addItem(line, ignoreBounds=True)
            self._event_lines.append(line)

            label = pg.TextItem(anchor=(0.5, 0.5))
            label.setZValue(20)
            label.setVisible(False)
            self._vb.addItem(label, ignoreBounds=True)
            self._event_labels.append(label)

        # A resident text item rather than a plot title or a legend: neither of those is
        # re-styled on a theme flip, thus both are left unreadable after one.
        self._empty_label = pg.TextItem("No channels displayed", anchor=(0.5, 0.5))
        self._empty_label.setZValue(30)
        self._empty_label.setVisible(False)
        self._vb.addItem(self._empty_label, ignoreBounds=True)
        self._style_overlays()

    def _style_overlays(self) -> None:
        """Color every event overlay and the placeholder for the current mode.

        Notes
        -----
        The colors come from the theme rather than from
        :func:`pyqtgraph.getConfigOption`, so that a display built before the theme was
        applied is still correct.
        """
        background = pg.mkBrush(plot_colors(self._mode)["background"])
        color = QColor(tokens(self._mode).success)
        self._headband.setBrush(background)
        for line in self._event_lines:
            line.setPen(pg.mkPen(color, width=2))
        for label in self._event_labels:
            label.fill = background
            label.setColor(color)
        self._empty_label.setColor(QColor(tokens(self._mode).text_secondary))

    def _build_pool(self) -> None:
        """Allocate the persistent curve pool, sized from the visible row count.

        Notes
        -----
        The size is deliberately ``_n_visible + 2 * _OVERSCAN + 2``: the largest band at
        a fractional offset spans exactly ``_n_visible + 1 + 2 * _OVERSCAN`` rows, thus
        the pool holds precisely one spare curve. Two fewer would make ``_free.pop()``
        raise mid-scroll -- one fewer would not, since :meth:`_render` releases the rows
        which left the band before it assigns the rows which entered it.

        The pool depends on the visible row count alone, never on the channel count nor
        on the layout, which is what makes a layout change free of any allocation.
        """
        for curve in self._pool:
            self._vb.removeItem(curve)
        self._pool = []
        self._free = []
        self._assigned = {}
        for _ in range(self._n_visible + 2 * _OVERSCAN + 2):
            curve = pg.PlotDataItem()
            # Per-curve peak downsampling and view clipping. This repeats the work for
            # every visible curve; mne-qt-browser instead downsamples the whole
            # (channels, samples) array once, which could be adopted later.
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            curve.setVisible(False)
            self._vb.addItem(curve, ignoreBounds=True)
            self._pool.append(curve)
            self._free.append(curve)

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        """Start the render clock."""
        self._timer.start()

    def stop(self) -> None:
        """Stop the render clock; idempotent."""
        self._timer.stop()

    def suspend(self, value: bool) -> None:
        """Stop polling the stream, without stopping the render clock.

        Parameters
        ----------
        value : bool
            Whether the stream is off limits.

        Notes
        -----
        Stopping the clock is not enough to keep this widget off a stream: a scroll, a
        row-count step and a layout change all repaint on their own, and a repaint which
        finds nothing it can reuse re-reads the window. That is exactly what must not
        happen while a reconnection is in flight -- the stream is briefly connected with
        a channel set the layout has not been rebuilt against yet, and the fetch then
        raises from inside whichever interaction triggered it.

        Separate from the clock rather than folded into it, because the two say
        different things: a stopped clock is a viewport which does not advance, and a
        suspended display is a stream which may not be read.
        """
        self._suspended = bool(value)

    @property
    def running(self) -> bool:
        """Whether the render clock is running."""
        return self._timer.isActive()

    @property
    def last_timestamp(self) -> float | None:
        """Newest timestamp of the last window this display fetched.

        ``None`` before the first fetch, and ``0.0`` for a stream which has never
        delivered a sample: the library allocates the timestamps with ``np.zeros`` and a
        real timestamp is never ``0.0``.

        :type: :class:`float` | None

        Notes
        -----
        A rendering fact and not a connection fact: it is what the last fetch returned,
        it does not move when a fetch returns the same window twice, and it keeps its
        value when a tick fetched nothing at all. Whether the stream is connected, and
        what to make of a timestamp which stopped moving, is the consumer's business.
        """
        return self._last_ts

    # -- channel layout ----------------------------------------------------------------
    def set_channel_layout(self, rows: Sequence[int]) -> None:
        """Set the visible rows of the display, in display order.

        Parameters
        ----------
        rows : sequence of int
            Acquisition indices of the channels to draw, in display order. Hidden
            channels are simply absent. An empty sequence hides every trace.

        Raises
        ------
        ValueError
            If a row is not a valid acquisition index. Checked before anything is
            mutated: an out-of-range index used to raise ``IndexError`` half-way and
            leave the display unusable, and a negative one silently drew a different
            channel -- the one failure mode the index model exists to rule out.

        Notes
        -----
        The curve pool is bound to y-slots, not to channels, thus a layout change
        creates, destroys, releases and repositions nothing: it re-pens and re-scales
        the assigned curves, because the identity beneath each row changed, and repaints
        once. That trailing repaint closes the window in which a curve would draw the
        previous channel's samples in the new channel's color.
        """
        rows = [int(row) for row in rows]
        n_channels = self.n_channels
        invalid = sorted({row for row in rows if not 0 <= row < n_channels})
        if invalid:
            raise ValueError(
                f"The channel layout must hold acquisition indices in "
                f"[0, {n_channels}), got {invalid}."
            )
        self._rows = rows
        self._rebuild_picks()
        self._restyle_assigned()
        self._empty_label.setVisible(self.n_rows == 0)
        if self.n_rows == 0:
            for curve in self._assigned.values():
                curve.setVisible(False)
            self._free.extend(self._assigned.values())
            self._assigned.clear()
            self._hide_events()
        self.scroll_to(self._top)  # re-clamp the extent, the range and the scrollbar
        self._repaint()

    def refresh_metadata(self) -> None:
        """Re-read the channel names, types, units and bads from the stream.

        Notes
        -----
        The transforms are re-applied as well as the pens: a channel's type, and
        therefore its gain, can change under a row.

        A structural change which *shrank* the channel set -- a reconnect to a narrower
        stream, or a ``pick`` on it -- leaves the layout holding indices which no longer
        exist. Those rows are dropped, because otherwise this method raised half-way and
        :meth:`_render` then raised on every tick, logging the "open an issue on GitHub"
        error 30 times a second. Rebuilding the layout in acquisition order is the job
        of whoever owns the channel model.
        """
        self._read_metadata()
        rows = [row for row in self._rows if row < self.n_channels]
        if len(rows) != len(self._rows):
            self.set_channel_layout(rows)
            return
        self._rebuild_picks()  # the set of stim channels may have changed
        self._restyle_assigned()

    def _restyle_assigned(self) -> None:
        """Re-pen and re-transform every assigned row, and invalidate the axis.

        Notes
        -----
        The extent guard is not optional: ``_assigned`` can still hold rows beyond the
        current extent, and both the pen and the transform of such a row index past the
        layout. A bulk hide reaches this method before the offset is re-clamped, and a
        hide while the stream is down leaves the stale rows in place for longer still,
        because :meth:`_render` returns before its release loop. Every caller therefore
        goes through here rather than looping over ``_assigned``, or a theme flip, an
        amplitude step or a color-mode change would raise mid-loop and leave half the
        curves restyled.
        """
        for row, curve in self._assigned.items():
            if row < self.n_rows:
                curve.setPen(self._pen_for(row))
                curve.setTransform(self._amp_transform(row))
        self._axis.invalidate()  # force the labels to be repainted

    def _rebuild_picks(self) -> None:
        """Rebuild the ``get_data`` picks and the position of the event channels.

        Notes
        -----
        ``self._picks`` is ``self._rows`` followed by the event channels the layout
        does not already carry, so that hiding a stim channel drops its trace and keeps
        its event overlays -- the overlays belong to the Events page, not to channel
        visibility.

        ``self._picks[:len(self._rows)] == self._rows`` is an invariant the render loop
        depends on: ``data[row]`` is the row's samples only because the prefix matches.
        Sorting or deduplicating the picks would silently draw the wrong channels.
        """
        position = {acq: index for index, acq in enumerate(self._rows)}
        self._picks = list(self._rows)
        self._event_pos = []
        for acq in self._event_acq:
            pos = position.get(acq)
            if pos is None:
                self._picks.append(acq)
                pos = len(self._picks) - 1
            self._event_pos.append(pos)

    @property
    def n_rows(self) -> int:
        """Number of visible rows, i.e. the scroll extent."""
        return len(self._rows)

    @property
    def n_channels(self) -> int:
        """Total number of acquisition channels of the stream."""
        return len(self._names)

    @property
    def n_visible(self) -> int:
        """Number of rows the viewport shows at once."""
        return self._n_visible

    @property
    def controls(self) -> DisplayControls:
        """Control bar owning the display state."""
        return self._controls

    # -- vertical navigation -----------------------------------------------------------
    @property
    def top_offset(self) -> float:
        """Fractional row index at the top of the visible band."""
        return self._top

    def scroll_to(self, row: float) -> None:
        """Move the top of the viewport to the fractional row ``row``.

        Parameters
        ----------
        row : float
            Target offset, in channel rows. Fractional values are allowed and clamped to
            the last page.
        """
        self._top = min(max(0.0, row), self._max_offset())
        self._apply_scroll()
        # The band moved, thus the rows which entered it carry no samples yet. Free of a
        # poll: the retained window is at most one tick old while the clock runs, and it
        # is the only thing a stopped clock -- a frozen viewport -- can follow at all.
        self._redraw()

    def scroll_by(self, rows: float) -> None:
        """Scroll the viewport by ``rows`` channel rows, fractional allowed."""
        self.scroll_to(self._top + rows)

    def _max_offset(self) -> float:
        """Return the largest legal top-row offset."""
        return max(0.0, float(self.n_rows - self._n_visible))

    def _apply_scroll(self) -> None:
        """Push the current offset to the y range, the scrollbar and the overlays."""
        band_top = self._top - 0.5 - _EVENT_LABEL_MARGIN
        self._vb.setYRange(band_top, self._top + self._n_visible - 0.5, padding=0)
        # wide rect, clipped to the view, masking the reserved label headroom.
        self._headband.setRect(-1e6, band_top, 2e6, _EVENT_LABEL_MARGIN)
        # The x range is owned by '_render', which returns early while everything is
        # hidden, thus it cannot be relied upon here: the placeholder is centred on the
        # range the view actually has, or it lands off-screen exactly when it is the
        # only thing left to show.
        self._vb.setXRange(0.0, self._winsize, padding=0)
        self._empty_label.setPos(self._winsize / 2, (self._n_visible - 1) / 2)
        blocked = self._scroll.blockSignals(True)
        self._scroll.setRange(0, round(self._max_offset() * _SB_RES))
        self._scroll.setPageStep(self._n_visible * _SB_RES)
        self._scroll.setSingleStep(_SB_RES)
        self._scroll.setValue(round(self._top * _SB_RES))
        self._scroll.blockSignals(blocked)

    # -- metadata accessors, called by the axis; every row is a visible row ------------
    def channel_name(self, row: int) -> str:
        """Return the channel name of a visible row.

        Parameters
        ----------
        row : int
            Visible-row index.

        Returns
        -------
        name : str
            Name of the channel drawn on that row.
        """
        return self._names[self._rows[row]]

    def is_bad(self, row: int) -> bool:
        """Return whether the channel of a visible row is marked bad.

        Parameters
        ----------
        row : int
            Visible-row index.

        Returns
        -------
        bad : bool
            Whether the channel is in ``info['bads']``.
        """
        return self._names[self._rows[row]] in self._bads

    def color_for(self, row: int) -> QColor:
        """Return the trace and label color of a visible row.

        Parameters
        ----------
        row : int
            Visible-row index.

        Returns
        -------
        color : QColor
            The theme-aware color of the channel drawn on that row.

        Notes
        -----
        The color is seeded by the **acquisition** index, never by the row, so that
        reordering and hiding channels recolors nothing.
        """
        acq = self._rows[row]
        if self.is_bad(row):
            return QColor(tokens(self._mode).bad)
        if self._color_mode == "type":
            return type_color(self._types[acq])
        return trace_color(acq, self._mode)

    def _pen_for(self, row: int) -> QPen:
        """Return the pen of a visible row, one pixel wide and solid."""
        return pg.mkPen(self.color_for(row), width=1)

    def _amp_transform(self, row: int) -> QTransform:
        """Return the vertical-scale transform of a visible row.

        Parameters
        ----------
        row : int
            Visible-row index.

        Returns
        -------
        transform : QTransform
            Scales y only, so that x stays untouched and the clipping and the
            downsampling keep working. The sign flips the samples into the inverted y
            axis, and the magnitude is the channel's gain times the bar's multiplier.
        """
        return QTransform().scale(1.0, -self._gain[self._rows[row]] * self._amp_mult)

    # -- input -------------------------------------------------------------------------
    def on_wheel(self, delta: int, modifiers) -> None:
        """Route a wheel notch to the amplitude scale or to the channel scroll.

        Parameters
        ----------
        delta : int
            Wheel delta in eighths of a degree, i.e. 120 per notch.
        modifiers : Qt.KeyboardModifiers
            Modifiers of the event; the control modifier selects the amplitude.

        Notes
        -----
        The scale goes through the control bar rather than being applied here, as it
        is the single source of truth: the bar owns the step size and the value the step
        is derived from, and the result comes back through ``scale_changed``.
        """
        if bool(modifiers & Qt.KeyboardModifier.ControlModifier):
            self._controls.step_scale(up=delta > 0)
        else:
            self.scroll_by(-delta / 120.0 * _SCROLL_ROWS_PER_NOTCH)

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        """Scroll the channels with the arrow, page, home and end keys."""
        key = ev.key()
        if key == Qt.Key.Key_Up:
            self.scroll_by(-1)
        elif key == Qt.Key.Key_Down:
            self.scroll_by(1)
        elif key == Qt.Key.Key_PageUp:
            self.scroll_by(-self._n_visible)
        elif key == Qt.Key.Key_PageDown:
            self.scroll_by(self._n_visible)
        elif key == Qt.Key.Key_Home:
            self.scroll_to(0.0)
        elif key == Qt.Key.Key_End:
            self.scroll_to(self._max_offset())
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    def showEvent(self, ev: QShowEvent) -> None:
        """Grab the keyboard focus and restore whatever a previous close dropped.

        Notes
        -----
        The counterpart of :meth:`TraceDisplay.closeEvent`. Without it a display closed
        and reopened keeps the previous mode's baked pens and bar icons for the rest of
        the process, and -- worse -- stays frozen on the last frame it drew, since the
        render clock is never restarted. The clock comes back only if the close stopped
        a running one, so showing a display which was never started still draws nothing.
        """
        super().showEvent(ev)
        if not self._following_theme:
            follow_theme(self, self._on_theme_changed, True)
            self._on_theme_changed(theme_controller.mode)  # catch up a missed flip
        if self._was_running:
            self._was_running = False
            self.start()
        self.setFocus()

    def closeEvent(self, ev: QCloseEvent) -> None:
        """Stop the render clock and drop the theme connection.

        Notes
        -----
        The stream is deliberately **not** disconnected: it is borrowed, and its
        ownership belongs to whoever built it.
        """
        self._was_running = self.running
        self.stop()
        follow_theme(self, self._on_theme_changed, False)
        super().closeEvent(ev)

    # -- control-bar handlers, the only writers of the render-loop caches --------------
    def _on_scrollbar(self, value: int) -> None:
        """Handle the vertical scrollbar moving."""
        self.scroll_to(value / _SB_RES)

    def _on_rows(self, value: int) -> None:
        """Resize the curve pool for a new visible row count."""
        self._n_visible = int(value)
        self._build_pool()
        self.scroll_to(self._top)  # re-clamp the offset and refresh the scrollbar

    def _on_window(self, value: float) -> None:
        """Set the time window, in seconds."""
        self._winsize = float(value)
        self._apply_scroll()  # the placeholder is centred on the window
        self._redraw()  # the samples map onto [0, W], thus x moved with the width

    def _on_scale(self, value: float) -> None:
        """Set the amplitude multiplier and re-apply every transform."""
        self._amp_mult = float(value)
        self._restyle_assigned()

    def _on_color_mode(self, mode: str) -> None:
        """Set the trace color mode and re-pen every assigned row."""
        self._color_mode = mode
        self._restyle_assigned()

    def _on_labels(self, on: bool) -> None:
        """Show or hide the channel labels on the left axis."""
        # 'setStyle' already drops the cached picture, re-fits the width and repaints.
        self._axis.setStyle(showValues=bool(on))

    def _on_events(self, on: bool) -> None:
        """Show or hide the stim-event overlays."""
        self._events_on = bool(on)
        if not self._events_on:
            self._hide_events()

    # -- re-theme ----------------------------------------------------------------------
    def _on_theme_changed(self, mode: str) -> None:
        """Recolor the pyqtgraph items and the bar icons for ``mode``.

        Parameters
        ----------
        mode : str
            The resolved mode which was applied, ``'light'`` or ``'dark'``.

        Notes
        -----
        :func:`~mne_lsl.viewer.theme.apply_theme` has already restyled the canvas and
        the axes and dropped the icon cache by the time this runs; what is left is all
        which baked a color: the trace pens, the axis labels, the event overlays and the
        toolbar icons.
        """
        self._mode = mode
        self._restyle_assigned()
        self._style_overlays()
        self._controls.retint_icons()

    # -- rendering ---------------------------------------------------------------------
    def _repaint(self) -> None:
        """Repaint the curves, polling the stream unless the clock stopped on a frame.

        Notes
        -----
        A stopped clock is what Freeze is, and a frozen viewport has to stay on the
        window it was frozen on: polling here would advance it to the newest samples,
        which is what hiding a channel while frozen used to do. Before the first frame
        exists there is nothing to redraw, so the poll is also what draws a display
        which has never ticked.

        :meth:`_poll` and never :meth:`_render`: this runs from an interaction and not
        from the clock, thus emitting ``polled`` here would report a render tick which
        never happened -- and a consumer driving a state machine off that signal would
        then advance it on a mouse click.
        """
        if self.running or self._frame is None:
            self._poll()
        else:
            self._redraw()

    def _redraw(self) -> None:
        """Repaint the retained window, without polling the stream.

        Notes
        -----
        What lets a stopped clock follow a scroll, a row-count change, a window change
        and a layout change: none of those repaint by themselves, so before this existed
        they left the visible rows of a frozen viewport blank.

        A no-op until the first frame exists. The frame is reindexed onto the current
        picks, as they are what its rows are ordered by; a channel which was hidden when
        the frame was taken has no samples in it at all, and the window is then re-read
        rather than leaving a row blank, which nothing distinguishes from a defect. That
        re-read goes through :meth:`_poll` and not :meth:`_render`, for the reason
        :meth:`_repaint` records.
        """
        if self._frame is None or not self._rows:
            return
        picks, data, relative = self._frame
        if picks == self._picks:
            self._draw(data, relative)
            return
        position = {acq: index for index, acq in enumerate(picks)}
        if any(acq not in position for acq in self._picks):
            self._poll()
            return
        self._draw(data[[position[acq] for acq in self._picks]], relative)

    def _render(self) -> None:
        """Poll the stream, paint the window it returned, and report the tick.

        Notes
        -----
        ``polled`` is emitted from here rather than from :meth:`_poll`, so that it fires
        on **every** path, including the early returns of a suspended display, of a
        disconnected stream and of an all-hidden layout. It is the only clock a consumer
        gets out of this widget, and a branch which did not emit would be a consumer
        which stops being told anything precisely when something went wrong.

        This is the one caller which emits, and it is the render clock's slot: the
        interaction paths go through :meth:`_repaint` and :meth:`_redraw`, so ``polled``
        means a tick of the clock and nothing else.
        """
        self._poll()
        self.polled.emit()

    def _poll(self) -> None:
        """Poll the stream, retain the window it returned and paint it.

        Notes
        -----
        The stream is polled on every tick, without consulting ``n_new_samples``: the
        curve-to-row reassignment at the band edge and the event placement depend on
        the scroll offset, thus skipping the fetch would leave a newly banded row blank
        on a stalled stream and turn the repaint at the end of
        :meth:`TraceDisplay.set_channel_layout` into a no-op.

        The two guards are the whole "do not touch the stream" rule of this widget, and
        every path which reads the buffer goes through here: a suspended display may not
        read it at all, see :meth:`suspend`, and a disconnected one has nothing to read.

        The connection guard is not enough on its own, which is what the ``try`` covers:
        the acquisition thread resets the stream on its own account, so a source lost
        between the guard and the read below raises ``RuntimeError`` from inside a
        ``QTimer`` slot. It is re-raised for a stream which is still connected, so a
        genuine failure -- the one ``get_data`` logs a bug report for -- is never
        swallowed; the tick which follows takes the guard instead.

        The un-filled part of a buffer is drawn as ``NaN`` rather than as samples, see
        the comment in the body: that is what puts a visible gap where no data exists.
        """
        if self._suspended or not self._stream.connected:
            return
        if not self._rows:
            # 'get_data(picks=[])' raises *and* logs an error asking for a bug report,
            # which at 30 Hz would fill the terminal.
            return
        try:
            # An irregularly-sampled stream declares 'sfreq == 0', and 'get_data' then
            # reads 'winsize' as a *sample count* rather than as seconds, so a float
            # raises. The whole buffer is fetched instead and the fixed [0, W] mapping
            # below clips it.
            winsize = self._winsize if self._stream.info["sfreq"] else None
            data, ts = self._stream.get_data(winsize, picks=self._picks, exclude=())
        except RuntimeError:
            if self._stream.connected:
                raise  # see the note above
            return
        # 'exclude' documents the intent only: integer picks bypass it entirely. 'picks'
        # is never 'None', which costs ~140x more to resolve.
        self._last_ts = float(ts[-1])
        if ts[0] == 0.0:
            # 'connect()' allocates the buffer and its timestamps with 'np.zeros', so an
            # un-filled region reads exactly 0.0 while a real timestamp never does.
            # Drawn as samples it joins across the outage: 'relative' below sends the
            # un-filled x far to the left of the view, and 'setClipToView' keeps one
            # point of it at the row baseline, so the curve enters the plot from the
            # left edge and rises into the first real sample. The guard is 'False' in
            # steady state, thus the common path costs one float compare.
            #
            # After a reconnection there is no gap *between* two runs of data: the
            # library allocates a fresh buffer, so the pre-outage samples are gone from
            # the stream entirely and the plot empties and refills from the right.
            # Stitching the two sides of an outage is not attempted.
            if data.dtype.kind != "f":
                # 'NaN' cannot be stored in an integer array and an integer stream is
                # drawable, so the assignment below would raise 'ValueError' on every
                # tick of one. 'data' is a copy -- 'get_data' indexes the buffer -- so
                # neither this nor the assignment can reach the acquisition buffer,
                # while 'ts' is a *view* into it and is never written. 'float32' and not
                # 'float64': the promoted window only ever reaches 'curve.setData',
                # single-precision on the wire anyway, and the copy is half the size.
                data = data.astype(np.float32)
            # A slice and not a boolean mask: the un-filled region is a contiguous
            # *prefix* by construction -- the buffer is allocated zeroed and every push
            # rolls it in from the right -- so the count of zeros is its length.
            # Measured 353 µs against 61 µs at 256 channels, on a 33 ms frame budget.
            data[:, : np.count_nonzero(ts == 0.0)] = np.nan
        # ponytail: the fetch is the whole layout, not just the ~29 banded rows, thus it
        # is O(n_rows) rather than O(rows drawn) -- measured 836 µs against 72 µs at 256
        # channels, i.e. 2.3% of the 33 ms budget spent copying rows nobody draws. The
        # upgrade is 'picks = rows[lo:hi] + events' with 'data[row - lo]', at the cost
        # of rebuilding the picks every tick; it pays off well past 256 channels.
        #
        # The sample times relative to the newest sample, i.e. ending at 0. Retained
        # rather than the absolute ones, which 'get_data' returns as a *view* into the
        # buffer the acquisition thread keeps rolling, and rather than the mapped x,
        # which a change of the window width makes stale. The newest sample is the one
        # already read out above, rather than converted a second time.
        relative = ts - self._last_ts
        self._frame = (list(self._picks), data, relative)
        self._draw(data, relative)

    def _draw(self, data: np.ndarray, relative: np.ndarray) -> None:
        """Paint one window onto the banded curves and the event overlays.

        Parameters
        ----------
        data : array of shape (n_picks, n_samples)
            One window, in the order of ``self._picks``.
        relative : array of shape (n_samples,)
            Sample times relative to the newest sample of the window, ending at 0.

        Notes
        -----
        Requires a non-empty layout: both callers return before this while every trace
        is hidden, so that the event overlays stay hidden with the placeholder.
        """
        window = self._winsize
        # The window's relative timestamps map onto a fixed [0, W]: the newest sample
        # sits at the right edge and the oldest at the left one, so new data enters at
        # the right and the traces sweep left under a static axis. x stays monotonic,
        # thus the clipping and the peak downsampling still apply.
        x = relative + window
        self._vb.setXRange(0.0, window, padding=0)

        lo = max(0, int(floor(self._top)) - _OVERSCAN)
        hi = min(self.n_rows, int(ceil(self._top + self._n_visible)) + _OVERSCAN)
        band = set(range(lo, hi))

        # release the curves which scrolled out of the overscan band, i.e. off-screen.
        for row in [row for row in self._assigned if row not in band]:
            curve = self._assigned.pop(row)
            curve.setVisible(False)
            self._free.append(curve)
        # Assign a free curve to every newly banded row. This only ever happens in the
        # overscan zone, so a row already carries data by the time it becomes visible.
        for row in band:
            if row not in self._assigned:
                curve = self._free.pop()
                self._assigned[row] = curve
                curve.setPos(0, row)  # vertical stacking, one y-slot per row
                curve.setTransform(self._amp_transform(row))
                curve.setPen(self._pen_for(row))
                curve.setVisible(True)
        # 'data[row]' is the row's own samples only because the picks prefix is the
        # layout, see '_rebuild_picks'.
        for row, curve in self._assigned.items():
            curve.setData(x, data[row])

        self._update_events(data, x)

    def _update_events(self, data: np.ndarray, x: np.ndarray) -> None:
        """Place the event overlays on the rising stim edges of the window.

        Parameters
        ----------
        data : array of shape (n_picks, n_samples)
            The fetched window, in picks order.
        x : array of shape (n_samples,)
            Sample times mapped onto ``[0, W]``.

        Notes
        -----
        The lines share the relative coordinates of the traces, thus they sweep with
        the window, and each label is parked in the reserved headroom above the top row.

        ponytail: an edge is an exact ``== 0`` to non-zero transition, which matches the
        legacy semantics and loses the edges of a noisy stim channel. Making the
        semantics previewable belongs to the Events page.

        Only finite samples count as an edge. ``inf > 0`` is true, thus an infinite
        sample used to be detected as one and then raised ``OverflowError`` on the label
        conversion -- from inside the render tick, which left the previous frame's
        overlays frozen on screen and repeated the failure at every tick.
        """
        if not self._events_on or not self._event_pos or x.size < 2:
            # hidden rather than simply skipped, so that a stim channel which lost its
            # type does not leave its last overlays on screen forever.
            self._hide_events()
            return
        label_y = self._top - 0.5 - _EVENT_LABEL_MARGIN / 2  # middle of the headroom
        placed = 0
        for pos in self._event_pos:
            samples = data[pos]
            rising = (
                np.flatnonzero(
                    (samples[1:] > 0) & (samples[:-1] == 0) & np.isfinite(samples[1:])
                )
                + 1
            )
            for index in rising:
                if placed >= _EVENT_POOL:
                    break
                xi = float(x[index])
                self._event_lines[placed].setPos(xi)
                self._event_lines[placed].setVisible(True)
                self._event_labels[placed].setText(str(int(samples[index])))
                self._event_labels[placed].setPos(xi, label_y)
                self._event_labels[placed].setVisible(True)
                placed += 1
        for line in self._event_lines[placed:]:
            line.setVisible(False)
        for label in self._event_labels[placed:]:
            label.setVisible(False)

    def _hide_events(self) -> None:
        """Hide every event line and label."""
        for line in self._event_lines:
            line.setVisible(False)
        for label in self._event_labels:
            label.setVisible(False)


def _range_native(ch_type: str, unit_mul: int) -> float:
    """Return the expected peak-to-peak amplitude of a channel, in native units.

    Parameters
    ----------
    ch_type : str
        Channel type, e.g. ``'eeg'``.
    unit_mul : int
        FIFF unit multiplier the stream declares, e.g. ``-6`` for micro.

    Returns
    -------
    range : float
        The range in the units the sender actually pushes.

    Notes
    -----
    The gain of a channel is :data:`_ROW_FILL` over this range, which is what makes the
    display unit-aware. For EEG declared in µV the result is ``0.012``, i.e. exactly the
    fixed gain of the accepted prototype, so the accepted look is preserved while a
    volt-declared stream -- what a replayed MNE recording is -- renders identically
    instead of drawing a flat line.
    """
    if ch_type in _RANGE_NATIVE:
        return _RANGE_NATIVE[ch_type]
    return _RANGE_SI.get(ch_type, 1.0) * 10.0 ** (-int(unit_mul))
