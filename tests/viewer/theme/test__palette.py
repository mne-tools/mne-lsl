from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette

from mne_lsl.viewer.theme import apply_theme, build_qpalette, tokens

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qtpy.QtWidgets import QApplication

    from mne_lsl.viewer.theme import ThemeController

_MODES = ("light", "dark")


@pytest.fixture
def plot(flush_deletes: Callable[..., None]) -> Generator[pg.PlotWidget, None, None]:
    """Yield a plot widget holding one curve, closed and destroyed afterwards."""
    widget = pg.PlotWidget()
    widget.plot([0, 1, 2], [0, 1, 0])
    yield widget
    widget.close()
    flush_deletes(widget)


def test_build_qpalette_modes_differ() -> None:
    """Test that the light and dark palettes differ on the key roles."""
    light, dark = build_qpalette("light"), build_qpalette("dark")
    role = QPalette.ColorRole
    for r in (role.Window, role.Base, role.Text, role.Highlight):
        assert light.color(r) != dark.color(r), r


@pytest.mark.parametrize("mode", _MODES)
def test_build_qpalette_roles(mode: str) -> None:
    """Test that every documented role carries its token color."""
    t = tokens(mode)
    role = QPalette.ColorRole
    pal = build_qpalette(mode)
    expected = {
        role.Window: t.window,
        role.WindowText: t.text,
        role.Base: t.base,
        role.AlternateBase: t.surface,
        role.ToolTipBase: t.surface,
        role.ToolTipText: t.text,
        role.PlaceholderText: t.text_disabled,
        role.Text: t.text,
        role.Button: t.raised,
        role.ButtonText: t.text,
        role.BrightText: t.error,
        role.Highlight: t.selection,
        role.HighlightedText: t.selection_text,
        role.Link: t.link,
        role.LinkVisited: t.link_visited,
        role.Mid: t.text_secondary,
    }
    for r, value in expected.items():
        assert pal.color(r) == QColor(value), (r, value)
    # the bevel ramp is derived from 'window' instead of being a token of its own.
    window = QColor(t.window)
    assert pal.color(role.Light) == window.lighter(150)
    assert pal.color(role.Midlight) == window.lighter(120)
    assert pal.color(role.Dark) == window.darker(140)
    assert pal.color(role.Shadow) == window.darker(220)


@pytest.mark.parametrize("mode", _MODES)
def test_build_qpalette_disabled_group(mode: str) -> None:
    """Test that the disabled group keeps text legible and the faces flat."""
    t = tokens(mode)
    role = QPalette.ColorRole
    disabled = QPalette.ColorGroup.Disabled
    pal = build_qpalette(mode)
    for r in (role.WindowText, role.Text, role.ButtonText, role.PlaceholderText):
        assert pal.color(disabled, r) == QColor(t.text_disabled), r
    for r in (role.Base, role.Button):
        assert pal.color(disabled, r) == QColor(t.window), r
    assert pal.color(disabled, role.Highlight) == QColor(t.surface)
    assert pal.color(disabled, role.HighlightedText) == QColor(t.text_disabled)


def test_build_qpalette_invalid_mode() -> None:
    """Test that an unknown mode raises."""
    with pytest.raises(ValueError, match="Invalid value for the 'mode' parameter"):
        build_qpalette("bogus")


def test_apply_theme(app: QApplication) -> None:
    """Test that the style, the palette, the QSS and the pyqtgraph config are pushed."""
    assert apply_theme(app, "dark") == "dark"
    sheet = app.styleSheet()
    assert sheet
    # the thin QSS wraps Fusion in a QStyleSheetStyle proxy, thus clearing the sheet is
    # what unwraps it and reveals the base style name.
    app.setStyleSheet("")
    try:
        assert app.style().objectName().lower() == "fusion"
    finally:
        app.setStyleSheet(sheet)
    t = tokens("dark")
    assert app.palette().color(QPalette.ColorRole.Window) == QColor(t.window)
    assert pg.getConfigOption("background") == t.plot_bg
    assert pg.getConfigOption("foreground") == t.plot_fg


def test_apply_theme_restyles_existing_plot(
    app: QApplication, plot: pg.PlotWidget
) -> None:
    """Test that a plot created before the flip follows the new mode."""
    apply_theme(app, "light")
    assert plot.backgroundBrush().color() == QColor(tokens("light").plot_bg)
    apply_theme(app, "dark")
    t = tokens("dark")
    assert plot.backgroundBrush().color() == QColor(t.plot_bg)
    axis = plot.getPlotItem().axes["left"]["item"]
    assert axis.pen().color() == QColor(t.plot_fg)
    assert axis.textPen().color() == QColor(t.plot_fg)


def test_theme_controller_requires_install(controller: ThemeController) -> None:
    """Test that setting a mode before 'install' raises."""
    controller._app = None
    with pytest.raises(RuntimeError, match="install"):
        controller.set_mode("dark")


def test_theme_controller_install(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that 'install' applies the setting once and exposes it."""
    assert controller.install(app, "light") == "light"
    assert controller.setting == "light"
    assert controller.mode == "light"
    assert recorder == ["light"]
    assert controller.set_mode("dark") == "dark"
    assert recorder == ["light", "dark"]
    assert controller.setting == "dark"
    assert controller.mode == "dark"


def test_theme_controller_invalid_setting(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that an invalid setting raises and leaves the state untouched."""
    controller.install(app, "light")
    with pytest.raises(ValueError, match="Invalid value for the 'mode' parameter"):
        controller.set_mode("bogus")
    assert controller.setting == "light"
    assert controller.mode == "light"
    assert recorder == ["light"]


def test_theme_controller_mode_before_install(controller: ThemeController) -> None:
    """Test that the resolved mode is valid before anything was applied."""
    controller._mode = None
    assert controller.mode in _MODES


def test_theme_controller_forced_mode_ignores_os(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that a forced light/dark setting does not follow the OS."""
    controller.install(app, "dark")
    recorder.clear()
    controller._on_os_scheme_changed(Qt.ColorScheme.Light)
    app.processEvents()
    assert recorder == []
    assert controller.mode == "dark"
    assert controller.setting == "dark"


def test_theme_controller_auto_follows_os(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that 'auto' re-applies the theme, deferred to the event loop."""
    controller.install(app, "auto")
    recorder.clear()
    controller._on_os_scheme_changed(Qt.ColorScheme.Dark)
    # the re-apply is posted with 'QTimer.singleShot(0, ...)', as the old palette is
    # still in effect while the signal is being delivered.
    assert recorder == []
    app.processEvents()
    assert len(recorder) == 1, recorder
    assert recorder[0] in _MODES


def test_theme_controller_user_choice_beats_queued_os_flip(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that forcing a mode while an OS re-apply is queued keeps the user's choice.

    The OS handler defers with 'QTimer.singleShot(0, ...)', thus the user can pick a
    mode before it fires. The queued re-apply must then do nothing, rather than
    reverting the setting to 'auto' and flipping the theme back.
    """
    controller.install(app, "auto")
    controller._on_os_scheme_changed(Qt.ColorScheme.Dark)  # queues the re-apply
    controller.set_mode("light")  # the user picks a mode before it fires
    recorder.clear()
    app.processEvents()  # the queued re-apply runs
    assert controller.setting == "light"
    assert controller.mode == "light"
    assert recorder == [], recorder


def test_theme_controller_coalesces_os_flips(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that several OS flips in one event-loop turn re-apply the theme once.

    macOS and Windows emit 'colorSchemeChanged' more than once per appearance switch,
    and each re-apply rebuilds the style, walks every widget and drops the icon cache.
    """
    controller.install(app, "auto")
    recorder.clear()
    for _ in range(3):
        controller._on_os_scheme_changed(Qt.ColorScheme.Dark)
    app.processEvents()
    assert len(recorder) == 1, recorder


def _emit_os_scheme_change(app: QApplication) -> None:
    """Emit 'colorSchemeChanged' from Python, skipping if the binding refuses to."""
    try:
        app.styleHints().colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    except (AttributeError, RuntimeError, TypeError) as error:  # pragma: no cover
        pytest.skip(f"the binding refuses to emit a foreign C++ signal: {error}")
    app.processEvents()


def test_theme_controller_connection_registered(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that the 'colorSchemeChanged' connection actually registered.

    The guard against the PySide6 6.11.1 bug where a 'UniqueConnection' to this signal
    is silently dropped: emitting it must reach the controller.
    """
    controller.install(app, "auto")
    recorder.clear()
    _emit_os_scheme_change(app)
    assert len(recorder) == 1, recorder


def test_theme_controller_install_twice(
    app: QApplication, controller: ThemeController, recorder: list[str]
) -> None:
    """Test that installing twice does not duplicate the OS connection."""
    controller.install(app, "auto")
    controller.install(app, "auto")
    recorder.clear()
    _emit_os_scheme_change(app)
    assert len(recorder) == 1, recorder
