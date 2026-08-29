"""Custom pyqtgraph axis and view box of the trace display."""

from __future__ import annotations

from math import ceil, floor
from typing import TYPE_CHECKING

import pyqtgraph as pg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qtpy.QtGui import QColor, QPainter

    from ._trace import TraceDisplay

# Longest channel name painted in full on the axis; a longer one is elided. 'AxisItem'
# grows to fit its widest tick label, and the 30-60 character names ordinary in clinical
# LSL and EDF streams then take most of the plot: measured on a 600 px plot, a
# 40-character name leaves the traces 261 px and a 200-character name leaves them 8 px.
# Eliding bounds the axis while leaving it auto-sized for the short names of a montage.
_MAX_LABEL = 18


class ChannelAxis(pg.AxisItem):
    """Left axis labelling the visible rows with their channel name.

    One integer tick per visible channel row, each label painted in the color of its
    trace. A bad channel is prefixed with ``'X '``, a non-color cue on top of the
    distinct label color, so the bad state never depends on color alone.

    Parameters
    ----------
    display : TraceDisplay
        The trace display the axis reads its channel names, colors and row offset from.
    **kwargs
        Additional keyword arguments are provided to :class:`pyqtgraph.AxisItem`.

    Notes
    -----
    Every index this axis handles is a **visible-row** index, i.e. a position in the
    layout the display was given, not an acquisition index: the display owns that
    translation and this axis never sees it.
    """

    def __init__(self, display: TraceDisplay, **kwargs) -> None:
        """Initialize the axis."""
        # assigned before 'super().__init__', which reaches paths calling back into this
        # subclass: an axis without its display raises 'AttributeError' at construction.
        self._display = display
        self._color_by_text: dict[str, QColor] = {}
        super().__init__(orientation="left", **kwargs)

    def invalidate(self) -> None:
        """Drop the cached picture and repaint, re-fitting the axis width.

        Notes
        -----
        ``picture = None`` alone is not enough: the axis width is cached too, thus
        a label which became longer -- a channel renamed, or a channel which became bad
        and gained its ``'X '`` prefix -- would be painted clipped until something else
        happened to resize the axis.
        """
        self.picture = None
        self._adjustSize()
        self.update()

    def tickValues(self, minVal: float, maxVal: float, size: float):
        """Return one integer tick per visible channel row.

        Parameters
        ----------
        minVal, maxVal : float
            Bounds of the visible range, in rows.
        size : float
            Length of the axis in pixels; unused, as the tick spacing is one row.

        Returns
        -------
        ticks : list
            One ``(spacing, values)`` pair, or an empty list when the range holds no
            content row.

        Notes
        -----
        The lower bound is clamped to the top content row, so that the event-label
        headroom reserved above it never grows a stray label, and the upper bound
        to the last row of the layout.
        """
        content_top = self._display.top_offset - 0.5
        lo = max(0, int(ceil(min(minVal, maxVal))), int(ceil(content_top)))
        hi = min(self._display.n_rows - 1, int(floor(max(minVal, maxVal))))
        if hi < lo:
            return []  # includes the all-hidden case, where 'n_rows' is 0
        return [(1.0, [float(value) for value in range(lo, hi + 1)])]

    def tickStrings(
        self, values: Sequence[float], scale: float, spacing: float
    ) -> list[str]:
        """Return the channel names of ``values`` and cache their label colors.

        Parameters
        ----------
        values : sequence of float
            Tick values, i.e. visible-row indices.
        scale, spacing : float
            Unused; the labels are names, not scaled numbers.

        Returns
        -------
        strings : list of str
            One label per value, empty for a value outside the layout. A name longer
            than ``_MAX_LABEL`` is elided, so that one long name cannot squeeze the
            traces out of the plot.
        """
        self._color_by_text = {}
        strings = []
        for value in values:
            row = int(round(value))
            if not 0 <= row < self._display.n_rows:
                strings.append("")
                continue
            name = self._display.channel_name(row)
            if len(name) > _MAX_LABEL:
                name = f"{name[: _MAX_LABEL - 1]}…"
            text = f"X {name}" if self._display.is_bad(row) else name
            # keyed on the rendered string, which is what 'drawPicture' receives.
            self._color_by_text[text] = self._display.color_for(row)
            strings.append(text)
        return strings

    def drawPicture(self, p: QPainter, axisSpec, tickSpecs, textSpecs) -> None:
        """Paint the axis, coloring every channel label individually.

        Parameters
        ----------
        p : QPainter
            Painter of the axis picture.
        axisSpec : tuple
            The ``(pen, start, stop)`` of the axis line.
        tickSpecs : list of tuple
            One ``(pen, start, stop)`` per tick.
        textSpecs : list of tuple
            One ``(rect, flags, text)`` per label.

        Notes
        -----
        Hand-rolled rather than delegated to :class:`pyqtgraph.AxisItem`, which paints
        every label with one pen. The cache filled by :meth:`tickStrings` is what
        recovers the color of each label, and a label with no cached color falls back to
        the axis text pen.
        """
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)
        pen, start, stop = axisSpec
        p.setPen(pen)
        p.drawLine(start, stop)
        for pen, start, stop in tickSpecs:
            p.setPen(pen)
            p.drawLine(start, stop)
        if self.style["tickFont"] is not None:
            p.setFont(self.style["tickFont"])
        p.setClipRect(self.boundingRect().toAlignedRect())
        default = self.textPen()
        for rect, flags, text in textSpecs:
            color = self._color_by_text.get(text)
            p.setPen(pg.mkPen(color) if color is not None else default)
            p.drawText(rect, int(flags), text)

    def wheelEvent(self, ev) -> None:
        """Scroll the channels; the axis never zooms."""
        self._display.on_wheel(ev.delta(), ev.modifiers())
        ev.accept()


class TraceViewBox(pg.ViewBox):
    """View box with the default pan, zoom and menu disabled.

    The display drives the range explicitly: a wheel event scrolls the channels, or
    scales the amplitude with the control modifier, and a drag is swallowed. This is
    deliberate, the legacy viewer silently changed the amplitude on a wheel event.

    Parameters
    ----------
    display : TraceDisplay
        The trace display the view box routes its input events to.
    **kwargs
        Additional keyword arguments are provided to :class:`pyqtgraph.ViewBox`.
    """

    def __init__(self, display: TraceDisplay, **kwargs) -> None:
        """Initialize the view box."""
        # assigned before 'super().__init__', as in 'ChannelAxis'.
        self._display = display
        super().__init__(**kwargs)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.disableAutoRange()
        self.invertY(True)  # row 0 at the top

    def wheelEvent(self, ev, axis=None) -> None:
        """Scroll the channels, or scale the amplitude with the control modifier."""
        self._display.on_wheel(ev.delta(), ev.modifiers())
        ev.accept()

    def mouseDragEvent(self, ev, axis=None) -> None:
        """Swallow the drag, so the plot never pans nor rubber-band zooms."""
        ev.ignore()
