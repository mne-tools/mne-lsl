"""``QPalette`` construction, theme application and the OS-following controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
import qtawesome
from qtpy.QtCore import QObject, QTimer, Signal
from qtpy.QtGui import QColor, QPalette

from ._qss import _QSS
from ._tokens import resolve_mode, tokens

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtWidgets import QApplication, QWidget


def build_qpalette(mode: str = "auto") -> QPalette:
    """Build a complete Fusion ``QPalette`` for ``mode``.

    Maps the tokens onto every relevant role for the Active/Inactive groups and a
    dedicated Disabled group so disabled text stays legible. The bevel ramp
    (``Light``/``Midlight``/``Dark``/``Shadow``) is derived from ``window`` so the token
    table stays purely semantic; ``Mid`` is the secondary-text color.

    Parameters
    ----------
    mode : str
        ``'auto'``, ``'light'`` or ``'dark'``.

    Returns
    -------
    palette : QPalette
        The palette to hand to :meth:`QApplication.setPalette`.
    """
    t = tokens(mode)
    role = QPalette.ColorRole
    pal = QPalette()

    pal.setColor(role.Window, QColor(t.window))
    pal.setColor(role.WindowText, QColor(t.text))
    pal.setColor(role.Base, QColor(t.base))
    pal.setColor(role.AlternateBase, QColor(t.surface))
    pal.setColor(role.ToolTipBase, QColor(t.surface))
    pal.setColor(role.ToolTipText, QColor(t.text))
    pal.setColor(role.PlaceholderText, QColor(t.text_disabled))
    pal.setColor(role.Text, QColor(t.text))
    pal.setColor(role.Button, QColor(t.raised))
    pal.setColor(role.ButtonText, QColor(t.text))
    pal.setColor(role.BrightText, QColor(t.error))
    pal.setColor(role.Highlight, QColor(t.selection))
    pal.setColor(role.HighlightedText, QColor(t.selection_text))
    pal.setColor(role.Link, QColor(t.link))
    pal.setColor(role.LinkVisited, QColor(t.link_visited))

    # bevel ramp derived from the window color, so 3D frames read in both modes.
    win = QColor(t.window)
    pal.setColor(role.Light, win.lighter(150))
    pal.setColor(role.Midlight, win.lighter(120))
    pal.setColor(role.Mid, QColor(t.text_secondary))
    pal.setColor(role.Dark, win.darker(140))
    pal.setColor(role.Shadow, win.darker(220))

    # disabled group: legible-but-muted text, flat inset/button faces.
    dis = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText, role.PlaceholderText):
        pal.setColor(dis, r, QColor(t.text_disabled))
    pal.setColor(dis, role.Base, QColor(t.window))
    pal.setColor(dis, role.Button, QColor(t.window))
    pal.setColor(dis, role.Highlight, QColor(t.surface))
    pal.setColor(dis, role.HighlightedText, QColor(t.text_disabled))
    return pal


def _restyle_existing_plots(app: QApplication, mode: str) -> None:
    """Re-color the already-created pyqtgraph plots for ``mode``.

    ``pg.setConfigOption`` only affects newly created items, thus the existing plots
    must be walked and restyled: the canvas background and every axis pen / text pen.
    Curves and other data items are the consumer's responsibility, as they recompute on
    ``theme_changed``.

    Parameters
    ----------
    app : QApplication
        The running application, walked with :meth:`QApplication.allWidgets`.
    mode : str
        ``'auto'``, ``'light'`` or ``'dark'``.
    """
    t = tokens(mode)
    bg, fg = QColor(t.plot_bg), QColor(t.plot_fg)
    for widget in app.allWidgets():
        if not isinstance(widget, pg.GraphicsView):
            continue
        widget.setBackground(bg)
        scene = widget.scene()
        if scene is None:
            continue
        for item in scene.items():
            if not isinstance(item, pg.PlotItem):
                continue
            for entry in item.axes.values():
                axis = entry["item"]
                axis.setPen(fg)
                axis.setTextPen(fg)


def apply_theme(app: QApplication, mode: str = "auto") -> str:
    """Apply the full theme to ``app`` and return the resolved mode.

    Sets the Fusion style, installs the mode's palette and the thin QSS, pushes the
    pyqtgraph background/foreground config, restyles the existing plots and refreshes
    the QtAwesome default icon color. Does not emit ``theme_changed``, which is
    :class:`ThemeController`'s job.

    Parameters
    ----------
    app : QApplication
        The running application.
    mode : str
        ``'auto'`` follows the OS scheme; ``'light'`` / ``'dark'`` force it.

    Returns
    -------
    mode : str
        The concrete mode which was applied, ``'light'`` or ``'dark'``.

    Notes
    -----
    The order of the calls below is load-bearing. The style must precede the palette, as
    a native style replaces the palette wholesale and only Fusion makes it
    authoritative. The style sheet is re-pushed after the palette even though every one
    of its colors is a ``palette(<role>)`` reference, as Qt caches the resolved values
    per style. The QtAwesome default color is refreshed before any consumer rebuilds an
    icon, as a ``QIcon`` bakes its color at creation.

    :func:`qtawesome.reset_cache` is not optional. :func:`qtawesome.icon` memoizes on
    the icon name and the explicit keyword arguments only, thus the default color is
    *not* part of the cache key: without dropping the cache, a consumer rebuilding its
    icons on ``theme_changed`` gets the colors of the previous mode back.
    """
    resolved = resolve_mode(mode)
    t = tokens(resolved)
    app.setStyle("Fusion")
    app.setPalette(build_qpalette(resolved))
    app.setStyleSheet(_QSS)
    pg.setConfigOption("background", t.plot_bg)
    pg.setConfigOption("foreground", t.plot_fg)
    _restyle_existing_plots(app, resolved)
    qtawesome.set_defaults(color=t.icon)
    qtawesome.reset_cache()
    return resolved


class ThemeController(QObject):
    """Re-apply the theme on OS/user changes and notify the consumers.

    Wraps :func:`apply_theme` with the runtime plumbing: it follows
    ``styleHints().colorSchemeChanged`` while the user setting is ``'auto'``, exposes
    the user setting and the resolved mode, and emits :attr:`theme_changed` after every
    (re)theme so consumers can recompute pyqtgraph colors, rebuild cached icons and
    refresh trace pens.

    Attributes
    ----------
    theme_changed : Signal
        Emitted with the resolved mode (``'light'`` / ``'dark'``) after every apply.
    """

    theme_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app: QApplication | None = None
        self._setting = "auto"
        self._mode: str | None = None
        self._following = False
        self._reapply_pending = False

    @property
    def mode(self) -> str:
        """Resolved concrete mode, ``'light'`` or ``'dark'``."""
        if self._mode is None:
            return resolve_mode("auto")
        return self._mode

    @property
    def setting(self) -> str:
        """User setting, ``'auto'``, ``'light'`` or ``'dark'``."""
        return self._setting

    def install(self, app: QApplication, setting: str = "auto") -> str:
        """Bind to ``app``, connect the OS follower and apply ``setting`` once.

        Parameters
        ----------
        app : QApplication
            The running application.
        setting : str
            Initial user setting, ``'auto'``, ``'light'`` or ``'dark'``.

        Returns
        -------
        mode : str
            The concrete mode which was applied.

        Notes
        -----
        ``colorSchemeChanged`` is connected at most once, and only ever by this
        instance: the flag is per-instance, not per-signal, so building further
        :class:`ThemeController` objects would connect once each. Use the module
        singleton, :data:`theme_controller`. The flag is also a one-way latch -- a
        connection dropped from the outside is not restored by calling this again.
        """
        # validated first: an invalid setting must not leave the controller bound to the
        # application and following the OS after the call raised.
        resolve_mode(setting)
        self._app = app
        # Connect a *bound method* of this instance to
        # 'app.styleHints().colorSchemeChanged', and do not pass
        # 'Qt.ConnectionType.UniqueConnection': connecting a free function with
        # 'UniqueConnection' silently fails to register on PySide6 6.11.1, which would
        # leave the OS theme-following dead under PySide6 while working under PyQt6. The
        # '_following' flag replaces it: a viewer can be built twice in one process, and
        # a duplicated connection would re-theme once per connection on every OS flip.
        if not self._following:
            app.styleHints().colorSchemeChanged.connect(self._on_os_scheme_changed)
            self._following = True
        return self.set_mode(setting)

    def set_mode(self, setting: str) -> str:
        """Set the user ``setting``, re-apply the theme and emit the change.

        Parameters
        ----------
        setting : str
            ``'auto'`` follows the OS; ``'light'`` / ``'dark'`` force the mode.

        Returns
        -------
        mode : str
            The concrete mode which was applied.

        Raises
        ------
        RuntimeError
            If :meth:`ThemeController.install` was not called first.
        ValueError
            If ``setting`` is not ``'auto'``, ``'light'`` or ``'dark'``.
        """
        resolved = resolve_mode(setting)  # validate before touching any state
        if self._app is None:
            raise RuntimeError("'ThemeController.install()' must be called first.")
        # applied before either attribute is written, so that a failure inside
        # 'apply_theme' leaves the controller reporting the mode still on screen rather
        # than a setting which was never applied.
        mode = apply_theme(self._app, resolved)
        # the *setting* is stored, not the resolved mode, so that 'Auto' stays selected
        # in the user interface when the OS later flips the resolved mode.
        self._setting = setting
        self._mode = mode
        self.theme_changed.emit(mode)
        return mode

    def _on_os_scheme_changed(self, _scheme: object) -> None:
        """Follow an OS color-scheme flip, while the user setting is ``'auto'``.

        Left undecorated on purpose: the signal carries a ``Qt.ColorScheme`` enum and a
        mis-specified ``@Slot`` fails silently at connect time, while a plain Python
        callable is correct on both bindings.
        """
        if self._setting != "auto" or self._reapply_pending:
            # coalesce: macOS and Windows emit this more than once per appearance
            # switch, and each re-apply rebuilds the style, walks every widget and drops
            # the icon cache, which every consumer then rebuilds.
            return
        # Defer: when the signal fires, the old palette is still in effect and Qt is
        # midway through reloading the system palette into the default one, thus
        # re-applying synchronously would race that reload and read stale colors.
        self._reapply_pending = True
        QTimer.singleShot(0, self._reapply_auto)

    def _reapply_auto(self) -> None:
        """Re-apply ``'auto'``, as the deferred target of the OS-scheme handler.

        A named bound method rather than a lambda so that it appears by name if it ever
        raises: a deferred slot exception is reported through the hook installed by
        :func:`mne_lsl.viewer._bootstrap.install_exception_policy`, and ``<lambda>`` in
        that traceback would be untraceable back to here. The mode is re-resolved
        instead of trusting the scheme carried by the signal.
        """
        self._reapply_pending = False
        if self._setting != "auto":
            # the user forced a mode while this re-apply was queued; their choice wins.
            return
        self.set_mode("auto")


# Module singleton shared by every consumer, e.g. to connect 'theme_changed'. A QObject
# is safe to build before the QApplication exists, unlike a QWidget.
theme_controller = ThemeController()


def follow_theme(consumer: QWidget, slot: Callable[[str], None], follow: bool) -> None:
    """Connect or drop a consumer's ``theme_changed`` connection; idempotent.

    Parameters
    ----------
    consumer : QWidget
        The consumer. Its ``_following_theme`` attribute holds the connection state and
        must be initialized to ``False`` before the first call.
    slot : callable
        The consumer's own handler, taking the resolved mode.
    follow : bool
        Whether the consumer follows the theme from now on.

    Notes
    -----
    :data:`theme_controller` is a process singleton and a widget which is closed but
    still referenced stays connected to it, so it keeps re-theming itself -- rebuilding
    the icons of a dead toolbar on every operating-system flip. Every consumer therefore
    follows from ``showEvent`` and unfollows from ``closeEvent``, and both of those fire
    more than once: a second ``connect`` re-themes twice per flip while a second
    ``disconnect`` raises, which is what the flag rules out.
    """
    if follow == consumer._following_theme:
        return
    if follow:
        theme_controller.theme_changed.connect(slot)
    else:
        theme_controller.theme_changed.disconnect(slot)
    consumer._following_theme = follow
