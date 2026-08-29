from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

import mne_lsl._commands.viewer as viewer_command
from mne_lsl.stream import BaseStream
from mne_lsl.viewer import Viewer
from mne_lsl.viewer._window import ViewerWindow

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qtpy.QtWidgets import QApplication

    from mne_lsl.stream import StreamLSL
    from mne_lsl.viewer.theme import ThemeController


@pytest.fixture
def viewers(flush_deletes: Callable[..., None]) -> Generator[list[Viewer]]:
    """Yield a list of viewers whose windows are closed at teardown.

    Closing the window is what stops the two worker threads and tears every document
    down, thus a test which shows a viewer registers it here rather than closing it
    itself: a failing assertion would otherwise leave both threads and an inlet behind.
    """
    created: list[Viewer] = []
    yield created
    windows = [
        viewer.window for viewer in reversed(created) if viewer.window is not None
    ]
    for window in windows:
        window.close()
    flush_deletes(*windows)
    created.clear()


def _no_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep 'show()' from starting a real discovery pass.

    A pass costs about a second and the 'stop()' of the close waits for the one in
    flight. Only 'test_show_starts_discovery' asserts on that call, thus every other
    test of this module opts out of the cost.
    """
    monkeypatch.setattr(ViewerWindow, "refresh", lambda self: None)


def test_init_creates_nothing(app: QApplication) -> None:
    """Test that the constructor builds no Qt object at all.

    'Viewer' has to be a plain importable object: the stub generator imports every
    module of the package, and a constructor building a widget would need an app.
    """
    before = set(app.topLevelWidgets())
    viewer = Viewer()
    assert viewer.window is None
    assert set(app.topLevelWidgets()) == before


@pytest.mark.parametrize("theme", ["Dark", "", "system", None])
def test_theme_validation(theme: object) -> None:
    """Test that an invalid theme is refused at construction.

    Validated here rather than inside 'show()', which would fail after the application
    was created and the OS queried.
    """
    with pytest.raises(ValueError, match="Invalid value for the 'theme' parameter"):
        Viewer(theme=theme)


def test_show_returns_a_visible_window(
    controller: ThemeController,
    viewers: list[Viewer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that 'show()' returns the window it put on screen."""
    _no_discovery(monkeypatch)
    viewer = Viewer()
    viewers.append(viewer)
    window = viewer.show()
    assert isinstance(window, ViewerWindow)
    assert window.isVisible()
    assert viewer.window is window


def test_show_is_idempotent(
    controller: ThemeController,
    viewers: list[Viewer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a second 'show()' returns the same window.

    A second window would come with a second dock manager and a second pair of worker
    threads, and 'Viewer.window' would then report the wrong one.
    """
    _no_discovery(monkeypatch)
    viewer = Viewer()
    viewers.append(viewer)
    assert viewer.show() is viewer.show()


def test_show_after_a_close_builds_a_live_window(
    controller: ThemeController,
    viewers: list[Viewer],
    flush_deletes: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a closed window is treated as absent rather than handed back.

    A closed window is spent: its close stopped both workers and tore every document
    down. Returning it hands back an invisible window, and 'start()' then enters a real
    event loop with zero visible windows -- which 'quitOnLastWindowClosed' can never
    end, so the process hangs.
    """
    _no_discovery(monkeypatch)
    viewer = Viewer()
    viewers.append(viewer)
    first = viewer.show()
    first.close()
    assert first.closed
    second = viewer.show()
    assert second is not first
    assert second.isVisible()
    assert not second.closed
    assert viewer.window is second
    flush_deletes(first)


def test_show_applies_the_theme(
    app: QApplication,
    controller: ThemeController,
    viewers: list[Viewer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the requested theme is installed before the window is built.

    Without the install the viewer runs on whatever theme the process was left in, which
    for the very first viewer of a process is no theme at all.
    """
    _no_discovery(monkeypatch)
    viewer = Viewer(theme="dark")
    viewers.append(viewer)
    viewer.show()
    assert controller.setting == "dark"
    assert controller.mode == "dark"
    assert app.styleSheet()


def test_show_starts_discovery(
    controller: ThemeController,
    viewers: list[Viewer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that showing the viewer starts exactly one discovery pass.

    The window itself deliberately starts none, thus without this call the launcher sits
    on 'Checking for streams…' forever.
    """
    calls: list[object] = []
    monkeypatch.setattr(ViewerWindow, "refresh", lambda self: calls.append(self))
    viewer = Viewer()
    viewers.append(viewer)
    window = viewer.show()
    assert calls == [window]


def test_show_adopts_a_borrowed_stream(
    controller: ThemeController,
    viewers: list[Viewer],
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the borrowed-stream contract end to end from the public API.

    This is what 'BaseStream.plot()' does: the caller keeps the stream, and closing the
    viewer must leave it connected.
    """
    _no_discovery(monkeypatch)
    stream, _ = lsl_stream()
    viewer = Viewer(stream=stream)
    viewers.append(viewer)
    window = viewer.show()
    assert len(window.documents) == 1
    doc = window.documents[0]
    assert not doc.owns_stream
    assert doc.stream is stream
    assert doc.trace.running
    window.close()
    assert window.documents == ()
    assert stream.connected


def test_show_tears_down_on_failure(
    controller: ThemeController,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    flush_deletes: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a failed startup leaves nothing behind and keeps the borrowed stream.

    Without the teardown, a startup failure strands a window, two worker threads and a
    live document with no handle left on any of them.
    """
    closed: list[ViewerWindow] = []
    original = ViewerWindow.close

    def _close(self: ViewerWindow) -> bool:
        closed.append(self)
        return original(self)

    def _raise(self: ViewerWindow) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(ViewerWindow, "close", _close)
    monkeypatch.setattr(ViewerWindow, "refresh", _raise)
    stream, _ = lsl_stream()
    viewer = Viewer(stream=stream)
    with pytest.raises(RuntimeError, match="boom"):
        viewer.show()
    assert viewer.window is None
    assert len(closed) == 1
    window = closed[0]
    assert window.documents == ()
    assert stream.connected  # borrowed, thus untouched by the teardown
    # No thread assertion here: 'refresh' is what raises, thus no worker was ever
    # started and both reads pass with the whole teardown removed.
    # 'test_close_event_stops_the_workers' is where the stop is pinned.
    # the failed viewer holds no reference to it, thus this test is the only place which
    # can free it.
    flush_deletes(window)


def test_start_returns_the_exit_code(
    app: QApplication,
    controller: ThemeController,
    viewers: list[Viewer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that 'start()' shows the window and returns the event loop's own code.

    The loop is spied rather than entered. Measured: a real 'app.exec()' inside a test
    session leaves the process unable to run a nested 'QEventLoop' afterwards -- every
    later 'qtbot.wait*' then blocks far past its own timeout -- which would poison
    whichever test happens to follow under a random order. The spy is also the stricter
    assertion: a sentinel code proves the value comes from the loop rather than being a
    hardcoded 0.
    """
    _no_discovery(monkeypatch)
    entered: list[object] = []
    monkeypatch.setattr(type(app), "exec", lambda self: entered.append(self) or 42)
    viewer = Viewer()
    viewers.append(viewer)
    assert viewer.start() == 42
    assert entered == [app]
    assert viewer.window is not None
    assert viewer.window.isVisible()


def test_command_returns_the_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that 'mne-lsl viewer' exits with the code the event loop returned.

    click reports the return value of a command nowhere, thus returning it leaves the
    shell seeing 0 however the viewer ended. Lives with the viewer tests rather than
    with the other command tests because it is the only one needing a Qt binding.
    """
    monkeypatch.setattr(viewer_command, "set_log_level", lambda _level: None)
    monkeypatch.setattr(Viewer, "start", lambda self: 7)
    result = CliRunner().invoke(viewer_command.run, [])
    assert result.exit_code == 7


def test_plot_borrows_the_stream_and_returns_the_code(
    app: QApplication, stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that 'plot()' hands its own stream to a viewer and returns the loop's code.

    'start()' is spied rather than entered -- it blocks on a real event loop -- which
    leaves the whole body of the method pinned: the stream reaches the viewer as the
    borrowed one, this call builds no window of its own, and the exit code is returned
    instead of dropped, as it is all a script or a shell ever sees of the viewer.
    """
    started: list[Viewer] = []
    monkeypatch.setattr(Viewer, "start", lambda self: started.append(self) or 7)
    before = set(app.topLevelWidgets())
    assert stream.plot() == 7
    assert len(started) == 1
    assert started[0]._stream is stream
    assert set(app.topLevelWidgets()) == before


def test_viewer_documents_the_process_wide_docking_flags() -> None:
    """Test that the public docstring warns an embedder about the docking flags.

    Showing the viewer sets the process-wide Qt-ADS configuration flags, which a
    'CDockManager' constructor consumes: an application which built its own manager
    first is taken down by a segmentation fault on its next 'addDockWidget'. Qt-ADS
    exposes no way to detect an existing manager, thus the documentation is the only
    available fix and it has to stay there.
    """
    notes = Viewer.__doc__.split("Notes")[1]
    assert "CDockManager" in notes
    assert "segmentation fault" in notes


def test_plot_documents_why_it_blocks() -> None:
    """Test that 'BaseStream.plot()' gives the real reason it blocks, and the way out.

    It used to justify blocking with "every ``plot`` of the scientific Python stack"
    being blocking, which is false for the closest neighbour an mne-lsl user transfers
    from: 'mne.io.Raw.plot' takes 'block' and defaults it to 'False'.
    """
    notes = BaseStream.plot.__doc__.split("Notes")[1]
    assert "scientific Python stack" not in notes
    assert "block=False" in notes
    assert "Viewer(stream=...).show()" in notes
