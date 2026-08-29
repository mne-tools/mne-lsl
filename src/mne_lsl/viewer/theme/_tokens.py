"""Semantic color tokens for the light and dark theme.

The single source of truth of the theming foundation: the ``QPalette``, the thin QSS,
the pyqtgraph colors and the QtAwesome icon color all derive from these two tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import darkdetect
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from ...utils._checks import check_value


@dataclass(frozen=True, slots=True)
class Tokens:
    """Semantic color tokens for one theme mode (hex strings, ``#rrggbb``).

    Background levels go from the outer chrome (``window``) inward to insets (``base``)
    and up to raised surfaces (``surface`` / ``raised``).
    """

    window: str  # main window / dialog chrome background
    base: str  # inset background: text fields, list/table, plot canvas
    surface: str  # raised panels, cards, alternate rows, tooltips
    raised: str  # button faces and other elevated controls
    text: str  # primary text
    text_secondary: str  # secondary / hint text, also dividers (QPalette.Mid)
    text_disabled: str  # disabled text and placeholders
    accent: str  # accent / highlight color (links, focus)
    accent_text: str  # text drawn on top of the accent
    selection: str  # selection background (QPalette.Highlight)
    selection_text: str  # text on a selection
    link: str  # hyperlink text
    link_visited: str  # visited hyperlink text
    icon: str  # QtAwesome default icon color
    plot_bg: str  # pyqtgraph plot background
    plot_fg: str  # pyqtgraph axes text / lines
    grid: str  # pyqtgraph grid lines
    bad: str  # bad-channel trace and axis label; the 'X ' prefix is the non-color cue
    error: str  # status: error
    warning: str  # status: warning
    success: str  # status: success


# The palette tables. Contrast note (WCAG): primary text sits at >=4.5:1 on window/base/
# surface, and accent_text on the selection background at >=4.5:1.
# 'bad' is the one value outside that review: the '#888888' of the trace-display
# prototype reads at only 3.54:1 on the light plot background, below the 4.5:1 the axis
# label drawn in the same color needs, thus a near-neutral grey clearing 4.5:1 on each
# mode's 'plot_bg' is used instead. 'tests/viewer/theme/test__tokens.py' pins both.
_LIGHT = Tokens(
    window="#f2f3f5",
    base="#ffffff",
    surface="#e8eaed",
    raised="#fbfcfd",
    text="#1b2028",
    text_secondary="#5b636e",
    text_disabled="#a6acb5",
    accent="#1f6feb",
    accent_text="#ffffff",
    selection="#1f6feb",
    selection_text="#ffffff",
    link="#0a5cc7",
    link_visited="#7048c0",
    icon="#3b424c",
    plot_bg="#ffffff",
    plot_fg="#454b54",
    grid="#dee1e6",
    bad="#6e7681",
    error="#c02b25",
    warning="#8a5a00",
    success="#1a7f37",
)

_DARK = Tokens(
    window="#21262d",
    base="#171b21",
    surface="#2a313a",
    raised="#333c46",
    text="#e6edf3",
    text_secondary="#9aa4b0",
    text_disabled="#5b6572",
    accent="#4c93f0",
    accent_text="#ffffff",
    selection="#2f6fd0",
    selection_text="#ffffff",
    link="#6cb0ff",
    link_visited="#b699f5",
    icon="#c3ccd6",
    plot_bg="#171b21",
    plot_fg="#b3bdc8",
    grid="#2c343d",
    bad="#8b949e",
    error="#f0837e",
    warning="#e0b054",
    success="#5fcf80",
)

_TABLE: dict[str, Tokens] = {"light": _LIGHT, "dark": _DARK}

# The theme-mode vocabulary, in the order the user interface offers it. Declared once
# and consumed by everything which validates or lists a mode -- 'resolve_mode', the
# public 'theme' parameter of the viewer and the window's Auto/Light/Dark toggle: a
# second copy is how the toggle ends up sitting on 'Auto' while the theme is dark.
_MODES = ("auto", "light", "dark")


def resolve_mode(mode: str = "auto") -> str:
    """Resolve ``mode`` to a concrete ``'light'`` or ``'dark'``.

    Parameters
    ----------
    mode : str
        ``'light'`` / ``'dark'`` force the mode; ``'auto'`` follows the OS through
        ``QApplication.styleHints().colorScheme()`` (Qt 6.5+), falling back to
        :mod:`darkdetect` and finally to ``'light'``.

    Returns
    -------
    mode : str
        Either ``'light'`` or ``'dark'``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'auto'``, ``'light'`` or ``'dark'``.

    Notes
    -----
    Resolving ``'auto'`` may query the OS, and :mod:`darkdetect` shells out to
    ``gsettings`` on Linux. Callers on a render path must pass an already-resolved
    ``'light'`` / ``'dark'`` -- which short-circuits on the first line -- and never the
    ``'auto'`` default.
    """
    check_value(mode, _MODES, "mode")
    if mode != "auto":
        return mode
    # 'instance()' returns whatever QCoreApplication exists, and that class has no
    # 'styleHints' -- an embedding host may well run a non-GUI application.
    hints = getattr(QApplication.instance(), "styleHints", None)
    if hints is not None:
        scheme = hints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return "dark"
        if scheme == Qt.ColorScheme.Light:
            return "light"
    try:
        detected = darkdetect.theme()
    except Exception:  # optional pre-Qt-6.5 fallback, never fatal
        detected = None
    return "dark" if detected == "Dark" else "light"


def tokens(mode: str = "auto") -> Tokens:
    """Return the :class:`Tokens` table for ``mode`` (``'auto'`` is resolved).

    Parameters
    ----------
    mode : str
        ``'auto'``, ``'light'`` or ``'dark'``.

    Returns
    -------
    tokens : Tokens
        The token table of the resolved mode.
    """
    return _TABLE[resolve_mode(mode)]
