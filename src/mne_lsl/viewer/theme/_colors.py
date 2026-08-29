"""Deterministic plot colors, channel-type colors and icons.

pyqtgraph items and ``QIcon`` objects do not read the ``QPalette``: they bake their
colors, thus they must be rebuilt from these helpers on every theme change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import qtawesome
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from ._tokens import resolve_mode, tokens

if TYPE_CHECKING:
    from qtpy.QtGui import QIcon

# Golden-ratio conjugate: successive multiples spread the hues maximally.
_GOLDEN_RATIO = 0.618033988749895


def _hue(index: int) -> float:
    """Return the golden-ratio spaced hue of a channel index."""
    # shared by 'channel_color' and 'trace_color': the two differ only in saturation and
    # value, and a channel must keep its hue across both modes.
    return (index * _GOLDEN_RATIO) % 1.0


# Fixed per-type colors: categorical hues muted to S=0.50, V=0.75 so a wall of traces
# reads calmly; misc stays neutral gray.
_TYPE_COLORS: dict[str, str] = {
    "eeg": "#6098bf",
    "eog": "#bf8c60",
    "ecg": "#bf6060",
    "emg": "#60bf60",
    "stim": "#9260bf",
    "misc": "#999999",
}


def plot_colors(mode: str = "auto") -> dict[str, str]:
    """Return the pyqtgraph canvas colors for ``mode``.

    Parameters
    ----------
    mode : str
        ``'auto'``, ``'light'`` or ``'dark'``.

    Returns
    -------
    colors : dict
        ``'background'``, ``'foreground'`` and ``'grid'`` as ``#rrggbb`` strings.
    """
    t = tokens(mode)
    return {"background": t.plot_bg, "foreground": t.plot_fg, "grid": t.grid}


def trace_color(index: int, mode: str = "auto") -> QColor:
    """Return the deterministic trace color of a channel, tuned for ``mode``.

    :func:`channel_color` is tuned for a dark plot (S=0.50, V=0.75) and is reused as-is
    for the dark mode. The light mode needs deeper, more saturated hues to read on a
    white canvas, thus the same golden-ratio hue is rebuilt at S=0.62, V=0.60.

    Parameters
    ----------
    index : int
        Channel index in **acquisition** order, i.e. the channel's identity. Seeding on
        a presentation or display position instead would recolor every channel whenever
        the user reorders or hides one.
    mode : str
        ``'auto'``, ``'light'`` or ``'dark'``.

    Returns
    -------
    color : QColor
        A muted, distinct color legible on the mode's plot background.

    Notes
    -----
    A render path calls this once per visible channel per frame and must therefore pass
    an already-resolved ``'light'`` / ``'dark'``, never the ``'auto'`` default: see
    :func:`mne_lsl.viewer.theme.resolve_mode`.
    """
    if resolve_mode(mode) == "dark":
        return channel_color(index)
    return QColor.fromHsvF(_hue(index), 0.62, 0.60)


def channel_color(index: int) -> QColor:
    """Return the deterministic color of a channel index, golden-ratio spaced.

    Parameters
    ----------
    index : int
        Channel index in **acquisition** order, i.e. the channel's identity, so that a
        channel keeps its color across a reorder or a visibility change.

    Returns
    -------
    color : QColor
        A muted color tuned for a dark plot background.
    """
    return QColor.fromHsvF(_hue(index), 0.50, 0.75)


def type_color(ch_type: str) -> QColor:
    """Return the fixed color of a channel type.

    Parameters
    ----------
    ch_type : str
        Channel type, e.g. ``'eeg'``. Unknown types fall back to the misc color.

    Returns
    -------
    color : QColor
        The channel-type color.
    """
    return QColor(_TYPE_COLORS.get(ch_type, _TYPE_COLORS["misc"]))


def _relative_luminance(color: QColor) -> float:
    """Return the WCAG relative luminance of an opaque color."""

    def _lin(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = _lin(color.redF()), _lin(color.greenF()), _lin(color.blueF())
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: QColor, bg: QColor) -> float:
    """Return the WCAG contrast ratio between two opaque colors.

    Parameters
    ----------
    fg : QColor
        Foreground color.
    bg : QColor
        Background color.

    Returns
    -------
    ratio : float
        The contrast ratio, between 1 and 21.
    """
    lf, lb = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(lf, lb), min(lf, lb)
    return (hi + 0.05) / (lo + 0.05)


# Size, in pixels, of every toolbar icon of the viewer. Declared next to 'icon' because
# a bound must not be written twice: the application toolbar and the document toolbar
# are the same chrome, and a second literal is how the two drift apart.
_ICON_PX = 16


def icon(name: str, **kwargs) -> QIcon:
    """Return a QtAwesome icon, so every component draws icons the same way.

    Parameters
    ----------
    name : str
        QtAwesome icon name, e.g. ``'mdi6.refresh'``.
    **kwargs
        Additional keyword arguments are provided to :func:`qtawesome.icon`, e.g.
        ``color`` or ``scale_factor``.

    Returns
    -------
    icon : QIcon
        The rendered icon, colored with the active theme's icon token by default.

    Raises
    ------
    RuntimeError
        If no ``QApplication`` is running.
    """
    if QApplication.instance() is None:
        # QtAwesome caches its 'IconicFont' on first use. Built without an application,
        # that cache holds no font and every later call raises 'Invalid font prefix',
        # for the whole process, even once an application exists. Fail loudly instead.
        raise RuntimeError(
            "'mne_lsl.viewer.theme.icon' requires a running QApplication: QtAwesome "
            "caches a font-less icon set when it is first used without one, which "
            "breaks every later icon in the process. Call "
            "'mne_lsl.viewer._bootstrap.ensure_application()' first."
        )
    return qtawesome.icon(name, **kwargs)
