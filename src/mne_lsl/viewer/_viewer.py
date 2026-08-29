"""Public facade of the viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils._checks import check_value
from ._bootstrap import assert_binding_coherence, ensure_application
from ._window import ViewerWindow
from .theme import _MODES, theme_controller

if TYPE_CHECKING:
    from ..stream import BaseStream

# Initial window size, in pixels: wide enough for the controller panel next to a legible
# trace display.
_INITIAL_SIZE = (1400, 850)


class Viewer:
    """Qt 6 viewer to discover, inspect and monitor LSL streams.

    Every launch starts disconnected: no stream is reconnected automatically, and the
    discovery runs in the background while the window is already responsive.

    Parameters
    ----------
    stream : BaseStream | None
        A connected stream to open a document for. It is **borrowed**: the viewer never
        disconnects it and closing its document leaves it connected, which is the
        :meth:`~mne_lsl.stream.BaseStream.plot` path. A stream which the viewer creates
        itself from the landing page is owned and is disconnected when its document
        closes.
    theme : str
        ``'auto'`` follows the OS color scheme, ``'light'`` and ``'dark'`` force it.

    Notes
    -----
    :meth:`Viewer.show` sets the **process-wide** configuration flags of the Qt Advanced
    Docking System, which a ``CDockManager`` constructor consumes and which cannot be
    changed once one exists. An application which embeds the viewer must therefore build
    its own dock manager, if any, *after* the viewer: a manager built first is left
    inconsistent with the flags and the process aborts with a segmentation fault on its
    next ``addDockWidget``. Qt-ADS exposes no way to detect an existing manager, so this
    ordering cannot be checked at runtime.
    """

    def __init__(
        self, stream: BaseStream | None = None, *, theme: str = "auto"
    ) -> None:
        """Initialize the viewer without creating any Qt object."""
        # Validated here and resolved nowhere: the theme string is only *checked* at
        # construction, so that an invalid one fails before an application exists and
        # before the OS is queried for its color scheme.
        check_value(theme, _MODES, "theme")
        self._stream = stream
        self._theme = theme
        self._window: ViewerWindow | None = None

    def show(self) -> ViewerWindow:
        """Create the application and the window, then show it.

        Does not enter the event loop, thus it returns immediately. This is the entry
        point used by the tests and by a caller which already runs an event loop.

        Returns
        -------
        window : ViewerWindow
            The window which was shown.

        Notes
        -----
        Idempotent: a second call returns the window it already built, rather than a
        second window with a second dock manager and a second pair of worker threads --
        which :attr:`Viewer.window` could then only lie about.

        A window which was **closed** is treated as absent instead, and a fresh one is
        built. Its close was the whole teardown -- both workers stopped, every document
        torn down -- so handing it back returns an invisible window which cannot be
        shown again, and :meth:`start` would then enter a real event loop with no
        visible window at all, which ``quitOnLastWindowClosed`` can never end.
        """
        if self._window is not None:
            if not self._window.closed:
                return self._window
            self._window = None
        app = ensure_application()
        assert_binding_coherence()
        # applied before any widget exists: the palette, the style sheet, the pyqtgraph
        # colors and the default icon color are all read at widget construction.
        theme_controller.install(app, self._theme)
        window = ViewerWindow()
        window.resize(*_INITIAL_SIZE)
        try:
            if self._stream is not None:
                window.adopt_stream(self._stream)
            window.show()
            window.refresh()
        except BaseException:
            # 'close()' is the whole teardown: it closes every document -- stopping the
            # render clocks and disconnecting only the streams the viewer owns, thus
            # leaving a borrowed stream connected -- and stops both workers.
            window.close()
            raise
        self._window = window
        return window

    def start(self) -> int:
        """Show the window and run the event loop until the window closes.

        The blocking entry point of the ``mne-lsl viewer`` command. A failure during the
        startup tears down whatever was already built, including a borrowed stream's
        document, before propagating.

        Returns
        -------
        code : int
            The exit code of the event loop.
        """
        self.show()
        # The loop returns when the last window closes, as 'quitOnLastWindowClosed' is
        # true, and the window's own close event has stopped the workers by then.
        return ensure_application().exec()

    @property
    def window(self) -> ViewerWindow | None:
        """The window, or ``None`` before :meth:`show` was called."""
        return self._window
