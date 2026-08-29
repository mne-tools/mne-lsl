from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import QItemSelectionModel, QRect
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QInputDialog, QMessageBox

from mne_lsl.stream import StreamLSL
from mne_lsl.viewer import _window
from mne_lsl.viewer._bootstrap import import_ads
from mne_lsl.viewer._document import LIVE, StreamDocument
from mne_lsl.viewer._launcher import PROGRESS_TEXT
from mne_lsl.viewer.backend import (
    STATE_AVAILABLE,
    STATE_CHECKING,
    STATE_INVALID,
    STATE_LOADING,
    STATE_UNAVAILABLE_NO_MATCH,
    Connector,
    Discovery,
    Prober,
    StreamDescriptor,
    StreamIdentity,
    ViewerConfig,
    channel_key,
    connect_stream,
    list_configurations,
    save_configuration,
)
from mne_lsl.viewer.display import WINDOW_RANGE
from mne_lsl.viewer.theme import _ADS_ICONS, _MODES

ads = import_ads()

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pytestqt.qtbot import QtBot
    from qtpy.QtWidgets import QApplication

    from mne_lsl.viewer._window import ViewerWindow
    from mne_lsl.viewer.theme import ThemeController


def _descriptor_for(
    stream: StreamLSL, *, source_id: str | None = None, uid: str | None = None
) -> StreamDescriptor:
    """Return a descriptor for a stream the fixture created.

    Deliberately not built with 'backend.stream_identity': that helper is code of this
    phase and must not be what validates the rest of it.
    """
    return StreamDescriptor(
        identity=StreamIdentity(
            name=stream.name,
            stype=stream.stype,
            source_id=stream.source_id if source_id is None else source_id,
        ),
        n_channels=len(stream.ch_names),
        sfreq=stream.info["sfreq"],
        hostname="localhost",
        dtype="float32",
        uid=str(uuid.uuid4()) if uid is None else uid,
    )


def _open(
    window: ViewerWindow, stream: StreamLSL, *, source_id: str | None = None
) -> StreamDocument:
    """Open one document for ``stream``, through the connector's own callback.

    Every test but the end-to-end one drives this path rather than the network: the
    connector is what hands a connected stream over, and this is the slot it hands to.
    """
    descriptor = _descriptor_for(stream, source_id=source_id)
    window._on_connected(descriptor, stream)
    return window.documents[-1]


def _borrow(
    window: ViewerWindow, stream: StreamLSL, *, source_id: str | None = None
) -> StreamDocument:
    """Open one document over a *borrowed* stream, so closing it leaves it connected.

    The 'BaseStream.plot()' ownership, reached with an explicit source ID so that one
    connection can carry several identities. Used where a test has to close a workspace
    and then hand the very same stream to a configuration load.
    """
    identity = _descriptor_for(stream, source_id=source_id).identity
    doc = StreamDocument(window._manager, stream, identity, owns_stream=False)
    window._register(doc)
    return doc


def _spy_open(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[StreamDescriptor, ...], float]]:
    """Record every 'Connector.open' call instead of connecting anything."""
    calls: list[tuple[tuple[StreamDescriptor, ...], float]] = []
    monkeypatch.setattr(
        Connector,
        "open",
        lambda self, descriptors, bufsize: calls.append((tuple(descriptors), bufsize)),
    )
    return calls


# -- the landing page and the workspace ------------------------------------------------
def test_landing_first(window: ViewerWindow) -> None:
    """Test that a fresh window shows the launcher and not an empty workspace."""
    assert window._stack.currentWidget() is window._landing
    assert window.documents == ()
    assert window.active_document is None
    assert window._sb_state.text() == "Disconnected"
    assert window._sb_identity.text() == ""
    assert window._sb_meta.text() == ""


def test_window_configures_the_docking_flags(
    flush_deletes: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the window sets the docking flags itself, before its dock manager.

    Spied, and not read back off 'CDockManager': the flags are process-global static
    state, thus any earlier test which set them leaves a window which never configures
    them indistinguishable -- and the reading form then passes in every collection order
    but the one where this module runs first.
    """
    calls: list[int] = []
    original = _window.configure_docking

    def _configure() -> None:
        calls.append(1)
        original()

    monkeypatch.setattr(_window, "configure_docking", _configure)
    built = _window.ViewerWindow()
    try:
        assert calls == [1]
    finally:
        built.close()
        flush_deletes(built)


def test_focus_highlighting_is_live(
    app: QApplication,
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the focus signal really fires, i.e. the docking flags were set in time.

    The behavioural proxy for 'the flags were set before the manager was constructed':
    with 'FocusHighlighting' off at construction the signal never fires, and switching
    it on afterwards crashes the process on the next 'addDockWidget'.

    Two documents over **one** stream: what a second document needs here is a second
    identity, not a second connection, and a connection is 1.68 s of liblsl handshake.
    """
    seen: list[object] = []
    window._manager.focusedDockWidgetChanged.connect(lambda _old, now: seen.append(now))
    stream, _ = lsl_stream()
    first = _open(window, stream, source_id="unit-1")
    second = _open(window, stream, source_id="unit-2")
    window.show()
    app.processEvents()
    seen.clear()
    window._raise(first)
    app.processEvents()
    assert seen
    assert window._manager.focusedDockWidget() in (first, second)


def test_open_streams_connects_and_opens(
    window: ViewerWindow,
    qtbot: QtBot,
    outlets: Callable[..., StreamDescriptor],
) -> None:
    """Test the whole connect path, once, against a real outlet on the network.

    The one test which goes through discovery and the connector thread: a mis-called
    buffer derivation, a missing connector connection or a document which is never
    registered all surface here and nowhere else. Kept to exactly one test because the
    liblsl resolution is intermittently flaky under parallel load.

    The 'failed' signal is watched as well, so that a connection which liblsl lost the
    race for reports its own reason instead of an opaque wait timeout:
    'StreamLSL.connect' applies a 2 s timeout to the resolution, which a loaded machine
    can exceed.
    """
    descriptor = outlets(n_channels=3, ch_names=["Fp1", "Fp2", "Cz"])
    failures: list[str] = []
    window._connector.failed.connect(lambda _d, message: failures.append(message))
    window.open_streams([descriptor])
    qtbot.waitUntil(lambda: len(window.documents) == 1 or bool(failures), timeout=20000)
    assert not failures, failures
    doc = window.documents[0]
    assert doc.identity == descriptor.identity
    assert doc.owns_stream
    assert doc.stream.connected
    assert doc.trace.running
    assert window._stack.currentWidget() is window._dock_host
    assert window.active_document is doc


def test_open_streams_derives_bufsize(
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the batch carries a buffer covering the widest selectable window.

    A hardcoded size, or the *current* window instead of the maximum, leaves the display
    drawing over part of its time axis as soon as the window is widened.

    The derivation is spied rather than recomputed: 'derive_bufsize(WINDOW_RANGE[1])' is
    30.0 today, thus asserting the value alone passes with the literal '30.0' written at
    the call site -- the second-literal failure the shared bound exists to prevent.
    """
    calls = _spy_open(monkeypatch)
    derived: list[float] = []
    original = _window.derive_bufsize

    def _derive(seconds: float) -> float:
        derived.append(seconds)
        return original(seconds)

    monkeypatch.setattr(_window, "derive_bufsize", _derive)
    entry = descriptor()
    window.open_streams([entry])
    assert len(calls) == 1
    wanted, bufsize = calls[0]
    assert wanted == (entry,)
    assert derived == [WINDOW_RANGE[1]]
    assert bufsize == original(WINDOW_RANGE[1])
    assert bufsize >= WINDOW_RANGE[1]
    assert bufsize == int(bufsize)


def test_open_streams_skips_event_sources(
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an event source is never connected as a document.

    'StreamLSL' refuses a buffer in seconds for an irregularly sampled stream, thus the
    user would see an unexplained connection error for a stream the launcher lists.
    """
    calls = _spy_open(monkeypatch)
    window.open_streams([descriptor(sfreq=0.0)])
    assert calls == []
    assert window.documents == ()
    # a regular descriptor in the same batch is still opened.
    regular = descriptor()
    window.open_streams([descriptor(sfreq=0.0), regular])
    assert len(calls) == 1
    assert calls[0][0] == (regular,)


def test_open_streams_raises_an_open_identity(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that opening an identity which is already open raises its document.

    Without the de-duplication the same stream gets two documents, two channel models
    and two render clocks.

    Two documents over one stream: distinct identities are what this needs, and each
    connection is 1.68 s of liblsl handshake.
    """
    stream, _ = lsl_stream()
    first = _open(window, stream, source_id="unit-1")
    second = _open(window, stream, source_id="unit-2")
    assert window.active_document is second
    calls = _spy_open(monkeypatch)
    window.open_streams(
        [_descriptor_for(first.stream, source_id=first.identity.source_id)]
    )
    assert calls == []
    assert len(window.documents) == 2
    assert window.active_document is first


def test_open_streams_dedups_in_flight(
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that two opens of one identity back to back connect it once.

    A double click on 'Open selected' would otherwise connect the same stream twice, and
    the second stream is then released -- an inlet and its acquisition thread built for
    nothing.
    """
    calls = _spy_open(monkeypatch)
    entry = descriptor()
    window.open_streams([entry])
    window.open_streams([entry])
    assert len(calls) == 1


def test_open_streams_merges_the_batch_in_flight(
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a second Open carries the still-connecting streams with it.

    'Connector.open' bumps its generation counter, i.e. it supersedes the batch in
    flight instead of queueing behind it. Submitting only the new selection therefore
    drops the stream the user asked for first -- and its identity is then in flight
    forever, so it can never be opened again for the life of the process.
    """
    calls = _spy_open(monkeypatch)
    first, second = descriptor(name="first"), descriptor(name="second")
    window.open_streams([first])
    window.open_streams([second])
    assert len(calls) == 2
    assert calls[0][0] == (first,)
    assert calls[1][0] == (first, second)
    assert set(window._connecting) == {first.identity, second.identity}


def test_a_failure_survives_a_sibling_success(
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the success of one stream does not wipe the failure of its sibling.

    The status bar is the only error surface of a failed connection, thus a batch which
    clears it unconditionally when the last outcome comes back tells the user nothing at
    all about the stream which failed.
    """
    _spy_open(monkeypatch)
    stream, _ = lsl_stream()
    good = _descriptor_for(stream)
    bad = descriptor(name="unreachable")
    window.open_streams([bad, good])
    assert window.statusBar().currentMessage() == "Connecting to 2 stream(s)…"
    window._on_failed(bad, "boom")
    window._on_connected(good, stream)
    message = window.statusBar().currentMessage()
    assert "unreachable" in message
    assert "boom" in message
    assert window._connecting == {}
    assert len(window.documents) == 1


def test_a_completed_batch_clears_its_own_message(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a clean batch empties the in-flight map and takes its message away."""
    _spy_open(monkeypatch)
    stream, _ = lsl_stream()
    entry = _descriptor_for(stream)
    window.open_streams([entry])
    assert window.statusBar().currentMessage()
    window._on_connected(entry, stream)
    assert window._connecting == {}
    assert window.statusBar().currentMessage() == ""


def test_on_connected_releases_a_duplicate(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a stream arriving for an identity already open is disconnected.

    The connector transfers ownership with its signal, thus returning without
    disconnecting leaks a live inlet plus its acquisition thread for the whole process,
    invisibly.
    """
    first, _ = lsl_stream()
    late, _ = lsl_stream()
    descriptor = _descriptor_for(first)
    window._on_connected(descriptor, first)
    assert len(window.documents) == 1
    window._on_connected(descriptor, late)
    assert len(window.documents) == 1
    assert not late.connected
    assert first.connected


def test_on_connected_releases_a_document_which_refuses_to_build(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a construction which raises releases the stream it was handed.

    Reachable: the document refuses a stream declaring no sampling rate, and a source
    re-provisioned as an event stream between the discovery pass and this connection
    arrives here as exactly that. The exception escapes into a Qt slot, where the policy
    logs it and carries on -- so without the guard the connected inlet and its
    acquisition thread stay alive, unreferenced, for the life of the process.
    """
    stream, _ = lsl_stream()
    descriptor = _descriptor_for(stream)

    def _raise(*args: object, **kwargs: object) -> None:
        raise ValueError("an irregularly sampled stream cannot be displayed")

    monkeypatch.setattr(_window, "StreamDocument", _raise)
    window._on_connected(descriptor, stream)
    assert window.documents == ()
    assert not stream.connected
    assert "Could not open" in window.statusBar().currentMessage()


def test_documents_tab_together(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that every document lands in one dock area: tabs, not stacked panes.

    Three documents over one stream: three identities are what this needs, and each
    connection is 1.68 s of liblsl handshake.
    """
    stream, _ = lsl_stream()
    docs = [_open(window, stream, source_id=f"unit-{index}") for index in range(3)]
    areas = {doc.dockAreaWidget() for doc in docs}
    assert len(areas) == 1
    assert window._manager.dockAreaCount() == 1


def test_object_names_unique_over_the_process(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a closed document never frees its name for a later one.

    Qt-ADS keeps a closed dock widget in its map forever, thus a counter derived from
    the number of *open* documents reuses a name which is still taken -- measured to
    lose a map entry and to write an ambiguous saved layout.

    Two streams and three documents, and not one stream for all three: closing 'first'
    disconnects the stream it owns, thus the last document would then build a channel
    model over a dead stream.
    """
    first = _open(window, lsl_stream()[0])
    survivor, _ = lsl_stream()
    second = _open(window, survivor, source_id="unit-2")
    first.closeDockWidget()
    third = _open(window, survivor, source_id="unit-3")
    names = {doc.objectName() for doc in (first, second, third)}
    assert len(names) == 3
    assert len(window._manager.dockWidgetsMap()) == 3


# -- the status bar --------------------------------------------------------------------
def test_status_bar_follows_focus(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the status bar retargets to whichever document became active."""
    first = _open(window, lsl_stream(n_channels=8, sfreq=100.0)[0])
    second = _open(window, lsl_stream(n_channels=5, sfreq=250.0)[0])
    window._set_active(first)
    assert "8/8 ch" in window._sb_meta.text()
    assert "100 Hz" in window._sb_meta.text()
    assert first.identity.name in window._sb_identity.text()
    window._set_active(second)
    assert "5/5 ch" in window._sb_meta.text()
    assert "250 Hz" in window._sb_meta.text()
    assert "100 Hz" not in window._sb_meta.text()
    assert second.identity.name in window._sb_identity.text()


def test_status_bar_follows_document_changed(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the shared status bar follows the active document's own changes.

    What it pins is the outcome, not the guard in the handler: the bar is always drawn
    from the active document, thus a background document freezing cannot change it
    either way. Dropping the 'changed' connection altogether is what this catches.

    Two documents over one stream: distinct identities are what this needs, and each
    connection is 1.68 s of liblsl handshake.
    """
    stream, _ = lsl_stream()
    first = _open(window, stream, source_id="unit-1")
    second = _open(window, stream, source_id="unit-2")
    assert window.active_document is second
    second.set_frozen(True)
    assert "Frozen" in window._sb_state.text()
    second.set_frozen(False)
    assert "Live" in window._sb_state.text()
    before = window._sb_state.text()
    first.set_frozen(True)
    assert window._sb_state.text() == before


def test_status_bar_identity_elided_with_tooltip(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a long source ID is shortened in the label and kept in the tooltip.

    The full value has to stay recoverable from the interface: it is half of what makes
    an identity exact.
    """
    source_id = "source-" + "x" * 40
    doc = _open(window, lsl_stream()[0], source_id=source_id)
    assert doc.identity.source_id == source_id
    assert source_id not in window._sb_identity.text()
    assert "…" in window._sb_identity.text()
    assert source_id in window._sb_identity.toolTip()
    assert doc.identity.name in window._sb_identity.toolTip()


def test_progress_reaches_both_surfaces(window: ViewerWindow) -> None:
    """Test that a discovery tag reaches the launcher label and the status bar alike."""
    window._on_progress("checking")
    assert window._landing._progress.text() == PROGRESS_TEXT["checking"]
    assert window.statusBar().currentMessage() == PROGRESS_TEXT["checking"]
    # an unknown tag must not raise from inside a Qt slot.
    window._on_progress("some-new-tag")
    assert window._landing._progress.text() == "some-new-tag"
    assert window.statusBar().currentMessage() == "some-new-tag"


def test_open_action_tracks_selection(
    window: ViewerWindow, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that the Open action follows the selection, and a pass which replaced it.

    A discovery pass rebuilds the table under the selection, thus the action has to be
    re-evaluated with no user interaction at all.

    The row is selected through the selection model, as the page does. 'selectRow' obeys
    the selection *mode*, so under 'MultiSelection' it toggles rather than selects and a
    second call would silently deselect the row again.
    """
    table = window._landing._table
    assert not window._act_open.isEnabled()
    window._on_streams_found([descriptor(name="a"), descriptor(name="b")])
    assert not window._act_open.isEnabled()
    table.selectionModel().select(
        table.model().index(0, 0),
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows,
    )
    assert window._act_open.isEnabled()
    window._on_streams_found([])
    assert not window._act_open.isEnabled()


def test_failed_connection_keeps_documents(
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a failed connection is reported without touching what is open.

    One stable error surface: a status message naming the stream, and no dialog to
    dismiss per failed stream of a batch.
    """
    doc = _open(window, lsl_stream()[0])
    failing = descriptor(name="unreachable")
    window._on_failed(failing, "boom")
    assert window.documents == (doc,)
    assert doc.stream.connected
    message = window.statusBar().currentMessage()
    assert "unreachable" in message
    assert "boom" in message
    assert window.findChildren(QMessageBox) == []


def test_refresh_clears_the_error_surface(
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a refresh takes the previous failure off the status bar.

    A failed connection is shown with no timeout and no dialog, thus the next refresh or
    open is the only thing which ends it: without this the bar goes on naming a stream
    the user has since asked the network for again. The discovery pass itself is stubbed
    out, as a real one costs a second and the close waits for it.
    """
    monkeypatch.setattr(Discovery, "refresh", lambda self: None)
    window._on_failed(descriptor(name="unreachable"), "boom")
    assert "unreachable" in window.statusBar().currentMessage()
    window.refresh()
    assert window.statusBar().currentMessage() == ""


# -- the borrowed-stream path ---------------------------------------------------------
def test_adopt_stream_borrows(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that an adopted stream is borrowed and survives its document.

    This is the 'BaseStream.plot()' contract seen from the window: defaulting the
    ownership to 'True' here would disconnect a stream the caller still owns.
    """
    stream, _ = lsl_stream()
    doc = window.adopt_stream(stream)
    assert not doc.owns_stream
    assert doc.identity == StreamIdentity(stream.name, stream.stype, stream.source_id)
    assert doc.trace.running
    doc.closeDockWidget()
    assert stream.connected
    assert window.documents == ()


def test_adopt_stream_rejects(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that only a connected, regularly sampled LSL stream can be adopted, once.

    The sampling-rate refusal is the 'BaseStream.plot()' path: 'open_streams' skips an
    event source before it can reach a document, thus without the check at the document
    itself the public path opens a viewer which cannot draw the stream it was handed.
    """
    with pytest.raises(TypeError, match="only open a document for an LSL stream"):
        window.adopt_stream(object())
    with pytest.raises(RuntimeError, match="only known once it is connected"):
        window.adopt_stream(StreamLSL(2.0, name="mne-lsl-viewer-absent"))
    assert window.documents == ()
    event_source, _ = lsl_stream(n_channels=2, sfreq=0.0, bufsize=50)
    with pytest.raises(ValueError, match="irregularly sampled"):
        window.adopt_stream(event_source)
    assert window.documents == ()
    stream, _ = lsl_stream()
    doc = window.adopt_stream(stream)
    assert window.adopt_stream(stream) is doc
    assert len(window.documents) == 1


# -- theming --------------------------------------------------------------------------
def test_theme_flip_reskins_and_retints(
    app: QApplication,
    controller: ThemeController,
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a theme flip reaches the toolbar, the dock chrome and every document.

    A 'QIcon' bakes its color at creation, thus the rendered pixmaps are what has to
    change -- for the toolbar actions and for the docking title-bar glyphs alike.

    The docking *glyphs* are what pins the dock re-skin: the style sheet is a static
    string installed by the constructor and never cleared, so asserting on it passes
    with the whole re-skin deleted from the flip handler.
    """
    doc = _open(window, lsl_stream()[0])
    controller.install(app, "light")
    light = [
        action.icon().pixmap(16, 16).toImage() for action, _ in window._toolbar_icons
    ]
    assert len(light) == len(window._toolbar_icons)
    assert all(not image.isNull() for image in light)
    provider = window._manager.iconProvider()
    light_ads = {
        slot: provider.customIcon(getattr(ads.eIcon, slot)).pixmap(16, 16).toImage()
        for slot in _ADS_ICONS
    }
    assert all(not image.isNull() for image in light_ads.values())
    retinted: list[object] = []
    monkeypatch.setattr(
        StreamDocument, "retint_icons", lambda self: retinted.append(self)
    )
    controller.set_mode("dark")
    dark = [
        action.icon().pixmap(16, 16).toImage() for action, _ in window._toolbar_icons
    ]
    assert "ads--CDockWidgetTab" in window._manager.styleSheet()
    for index, (before, after) in enumerate(zip(light, dark, strict=True)):
        assert before != after, index
    for slot, before in light_ads.items():
        after = provider.customIcon(getattr(ads.eIcon, slot)).pixmap(16, 16).toImage()
        assert before != after, slot
    assert retinted == [doc]


def test_theme_toggle_offers_every_mode() -> None:
    """Test that the toggle segments are the theme vocabulary itself, in its order.

    The index of the current setting is looked up in the shared tuple, thus a segment
    list which disagreed with it would raise from the constructor, or leave the
    highlight on a mode which is not the one in effect.
    """
    assert tuple(setting for _, _, setting in _window._THEME_SEGMENTS) == _MODES


def test_closed_window_stops_following_the_theme(
    app: QApplication, controller: ThemeController, window: ViewerWindow
) -> None:
    """Test that a closed window no longer re-themes itself on an OS theme flip.

    The theme controller is a process singleton and 'Viewer' holds its window forever,
    thus a closed one which stayed connected keeps rebuilding the icons of a dead
    toolbar -- and its documents are torn down by then, so nobody can see any of it.
    """
    controller.install(app, "light")
    window.close()
    assert not window._following_theme
    before = [
        action.icon().pixmap(16, 16).toImage() for action, _ in window._toolbar_icons
    ]
    controller.set_mode("dark")
    after = [
        action.icon().pixmap(16, 16).toImage() for action, _ in window._toolbar_icons
    ]
    for index, (closed, flipped) in enumerate(zip(before, after, strict=True)):
        assert closed == flipped, index


def test_reshown_window_follows_the_theme_again(
    app: QApplication, controller: ThemeController, window: ViewerWindow
) -> None:
    """Test that a window shown after a close reconnects and catches the flip it missed.

    The required counterpart of dropping the connection on close: a window put back on
    screen without it would keep the icons of whichever mode was current when it closed,
    for the rest of the process.
    """
    controller.install(app, "light")
    light = window._toolbar_icons[0][0].icon().pixmap(16, 16).toImage()
    window.close()
    controller.set_mode("dark")  # flipped while the window was not listening
    window.show()
    assert window._following_theme
    assert window._toolbar_icons[0][0].icon().pixmap(16, 16).toImage() != light


# -- teardown -------------------------------------------------------------------------
def test_last_document_returns_to_landing(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that closing the last document brings the launcher back."""
    doc = _open(window, lsl_stream()[0])
    assert window._stack.currentWidget() is window._dock_host
    doc.closeDockWidget()
    assert window.documents == ()
    assert window.active_document is None
    assert window._stack.currentWidget() is window._landing
    assert window._sb_state.text() == "Disconnected"
    assert window._sb_meta.text() == ""


def test_closing_the_active_document_follows_the_tab(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the status bar retargets to the tab Qt-ADS promoted, not to a guess.

    Closing the current tab makes the docking system promote another one, and it is not
    the last document opened: with three open it promotes the *first*. A guess therefore
    leaves the shared bar describing a document which is not the one on screen.
    """
    stream, _ = lsl_stream()
    docs = [_open(window, stream, source_id=f"unit-{index}") for index in range(3)]
    assert window.active_document is docs[2]
    docs[2].closeDockWidget()
    promoted = docs[0].dockAreaWidget().currentDockWidget()
    assert promoted in docs[:2]
    assert window.active_document is promoted


def test_close_all_documents(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that every document is closed, and not every other one.

    Iterating the live list while the close handler removes from it skips half of them.

    Three documents over one stream: the per-document assertion is on the render clock,
    which is one object per document, and it is what a list-mutation bug shows up in --
    the connection is shared, thus asserting it thrice would assert one object thrice.
    """
    stream, _ = lsl_stream()
    docs = [_open(window, stream, source_id=f"unit-{index}") for index in range(3)]
    window.close_all_documents()
    assert window.documents == ()
    for doc in docs:
        assert not doc.trace.running
    assert not stream.connected


def test_close_event_closes_documents(
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that closing the window tears its documents down.

    Measured: neither the window nor the dock host closes the dock widgets by itself,
    thus without this loop every clock keeps ticking and every stream stays connected.
    """
    doc = _open(window, lsl_stream()[0])
    window.close()
    assert window.documents == ()
    assert not doc.trace.running
    assert not doc.stream.connected


def test_close_event_stops_the_workers(
    window: ViewerWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that all four workers are stopped by the window closing.

    The 'aboutToQuit' fallback each worker owner installs needs a running event loop,
    i.e. it never fires on the path which merely shows the window, thus this is the only
    teardown there.

    Four and not two: the load-only connector and the prober are the two which are easy
    to forget, because each starts its thread lazily and only a configuration gesture
    ever does. A 'QThread' destroyed while it runs makes Qt abort the process, and only
    sometimes, so a missing stop lands as CI flake rather than as a failure here.
    """
    stopped: list[str] = []
    for cls in (Discovery, Connector, Prober):
        original = cls.stop

        def _stop(self, _original=original, _name=cls.__name__) -> None:
            stopped.append(_name)
            _original(self)

        monkeypatch.setattr(cls, "stop", _stop)
    window.refresh()
    assert window._discovery._thread.isRunning()
    window.close()
    # two 'Connector' entries: the incremental one and the load-only one.
    assert stopped == ["Discovery", "Connector", "Connector", "Prober"]
    for owner in (
        window._discovery,
        window._connector,
        window._loader,
        window._prober,
    ):
        assert not owner._thread.isRunning()


def test_refresh_after_a_close_starts_no_worker(window: ViewerWindow) -> None:
    """Test that a closed window starts no discovery pass, and reports itself as closed.

    'Discovery.refresh' starts its worker thread on demand and 'closeEvent' is the whole
    teardown, thus a pass asked for afterwards leaves a running thread which nothing
    will ever stop.
    """
    assert not window.closed
    window.close()
    assert window.closed
    window.refresh()
    assert not window._discovery._thread.isRunning()


# -- saved configurations: helpers -----------------------------------------------------
def _write_config(
    name: str,
    streams: list[tuple[str, str, str]],
    channels: dict[tuple[str, str, str], list[str]] | None = None,
    presentation: dict | None = None,
) -> None:
    """Write one configuration straight through the persistence layer."""
    save_configuration(
        ViewerConfig(
            name=name,
            streams=list(streams),
            channels={
                channel_key(identity): list(names)
                for identity, names in (channels or {}).items()
            },
            presentation=presentation or {},
        )
    )


def _spy_probe(monkeypatch: pytest.MonkeyPatch) -> list[tuple[StreamDescriptor, ...]]:
    """Record every 'Prober.probe' call instead of probing anything."""
    calls: list[tuple[StreamDescriptor, ...]] = []
    monkeypatch.setattr(
        Prober, "probe", lambda self, descriptors: calls.append(tuple(descriptors))
    )
    return calls


def _stub_text(
    monkeypatch: pytest.MonkeyPatch, value: str, *, accepted: bool = True
) -> list[str]:
    """Answer every name prompt with ``value``; return the prompt titles seen.

    The static helper is patched, which is what the shipped code calls: no dialog is
    ever constructed and no nested event loop is entered, so a prompt this suite forgot
    to stub fails as a missing call rather than hanging with no timeout.
    """
    seen: list[str] = []
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        staticmethod(
            lambda parent, title, *args, **kwargs: (
                seen.append(title),
                (value, accepted),
            )[1]
        ),
    )
    return seen


def _stub_box(monkeypatch: pytest.MonkeyPatch, kind: str, answer: object) -> list[str]:
    """Answer every message box of ``kind`` with ``answer``; return the texts seen.

    Stubs the window's own dialog helper rather than the static ``QMessageBox`` methods,
    because the helper is what the window calls -- it exists to render a name literally
    instead of interpreting it as markup.

    A dialog of an *unstubbed* kind raises instead of opening. That is the point: a real
    modal in a test blocks forever, and ``pytest-timeout``'s signal method cannot
    interrupt one, so the run dies at the session timeout with no useful report.
    Stacking two calls for different kinds works -- each chains to the one before it.
    """
    seen: list[str] = []
    previous = _window._message
    chained = getattr(previous, "_is_stub", False)

    def fake(parent, box_kind: str, title: str, text: str, detail: str = "") -> bool:
        if box_kind == kind:
            # Summary and detail joined for recording only. The dialog shows the summary
            # and folds the exception behind its details control, so both reach the user
            # and a test asserting on either keeps working.
            seen.append(f"{text}\n\n{detail}" if detail else text)
            return answer == QMessageBox.StandardButton.Yes
        if chained:
            return previous(parent, box_kind, title, text, detail)
        raise AssertionError(f"an unstubbed {box_kind} dialog was shown: {text!r}")

    fake._is_stub = True
    monkeypatch.setattr(_window, "_message", fake)
    return seen


def _card(window: ViewerWindow, name: str):
    """Return the configuration card named ``name``."""
    return window._landing._cards[name]


def _drive_load(
    window: ViewerWindow,
    name: str,
    connected: list[tuple[StreamDescriptor, object]],
    failed: list[tuple[StreamDescriptor, str]] = (),
) -> None:
    """Open a configuration with its connections already made.

    'Connector.open' must be spied by the caller: this hands the streams to the load
    path through the very slot the connector's signal reaches, which is what lets every
    load outcome be exercised for the price of the connections the test already has.
    """
    window.open_configuration(name)
    for descriptor, message in failed:
        window._on_load_failed(descriptor, message)
    for descriptor, stream in connected:
        window._on_load_connected(descriptor, stream)


class _DummyStream:
    """Stand-in for a connected stream, recording its own release."""

    def __init__(self) -> None:
        self.connected = True
        self.disconnected = 0

    def disconnect(self) -> None:
        """Record one release."""
        self.disconnected += 1
        self.connected = False


# -- availability: no stream, no network -----------------------------------------------
def test_cards_start_waiting_for_discovery(
    config_home: Path, window: ViewerWindow, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a card before the first discovery pass says it is waiting for one.

    'No matching stream' is a claim the viewer cannot make before a pass has landed, and
    initialising the set of present identities to an empty frozenset instead of 'None'
    is what makes the first paint state it anyway.
    """
    identity = descriptor().identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    window.reload_configurations()
    assert window._present is None
    card = _card(window, "mine")
    assert card.state == STATE_UNAVAILABLE_NO_MATCH
    assert card._reason.text() == "Waiting for discovery…"


def test_identity_match_submits_exactly_the_matching_probes(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that only the identities which are both required and present are probed.

    Three streams take part: one required and on the network, one required and absent,
    one present and required by nothing. A submit list built from the configurations
    instead of from the intersection fires a probe at the absent identity, which burns
    the full resolution timeout once per Refresh for a card already reading 'no matching
    stream'; a list built from the pass alone probes every stream on the network for
    nothing.
    """
    matching = descriptor(name="required-and-present")
    absent = descriptor(name="required-and-absent")
    bystander = descriptor(name="present-and-unwanted")
    _write_config(
        "mine",
        [matching.identity.as_tuple()],
        {matching.identity.as_tuple(): ["Fp1"]},
    )
    _write_config(
        "other", [absent.identity.as_tuple()], {absent.identity.as_tuple(): ["Cz"]}
    )
    calls = _spy_probe(monkeypatch)
    window.reload_configurations()
    assert calls == []  # nothing is probed before a pass has landed
    window._on_streams_found([matching, bystander])
    assert calls == [(matching,)]
    assert _card(window, "mine").state == STATE_CHECKING
    assert _card(window, "mine")._reason.text() == "Checking availability…"
    assert _card(window, "other").state == STATE_UNAVAILABLE_NO_MATCH


def test_no_identity_match_submits_no_probe(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a pass matching no configuration submits no probe batch at all.

    Dropping the guard submits an empty batch, i.e. one round trip through the worker
    per Refresh for nothing -- and a first launch with twenty saved configurations would
    pay a probe per configuration before this guard existed.
    """
    absent = descriptor(name="absent")
    _write_config(
        "mine", [absent.identity.as_tuple()], {absent.identity.as_tuple(): ["Cz"]}
    )
    calls = _spy_probe(monkeypatch)
    window.reload_configurations()
    window._on_streams_found([descriptor(name="unrelated")])
    assert calls == []


def test_probe_result_fans_out_to_every_configuration(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that one probe result settles every configuration naming that stream.

    Storing the result per configuration rather than per stream means N configurations
    cost N probes of the same stream, i.e. N inlets opened against one acquisition
    device.
    """
    shared = descriptor(name="shared")
    identity = shared.identity.as_tuple()
    _write_config("first", [identity], {identity: ["Fp1"]})
    _write_config("second", [identity], {identity: ["Fp1", "Fp2"]})
    calls = _spy_probe(monkeypatch)
    window.reload_configurations()
    window._on_streams_found([shared])
    assert calls == [(shared,)]
    window._on_probed(shared, ["Fp1", "Fp2", "Cz"])
    assert _card(window, "first").state == STATE_AVAILABLE
    assert _card(window, "second").state == STATE_AVAILABLE
    assert _card(window, "first")._reason.text() == "1 stream"


def test_probe_result_is_cached_by_uid(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a second pass re-probes only when the outlet instance changed.

    Dropping the uid from the cache key makes a re-provisioned stream keep its stale
    channel set for the rest of the session; skipping the cache re-probes every stream
    on every Refresh, which opens and destroys an inlet against a live device per click.
    """
    first = descriptor(name="device", source_id="unit-1", uid="uid-a")
    identity = first.identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    calls = _spy_probe(monkeypatch)
    window.reload_configurations()
    window._on_streams_found([first])
    window._on_probed(first, ["Fp1"])
    assert _card(window, "mine").state == STATE_AVAILABLE
    window._on_streams_found([first])  # the same outlet instance: nothing to re-probe
    assert calls == [(first,)]
    assert _card(window, "mine").state == STATE_AVAILABLE
    # re-instantiated under the same identity: the channel set may have changed.
    again = descriptor(name="device", source_id="unit-1", uid="uid-b")
    window._on_streams_found([again])
    assert calls == [(first,), (again,)]
    assert _card(window, "mine").state == STATE_CHECKING


def test_probe_failure_names_the_stream(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a probe which raised reads as unreachable and carries its message.

    Storing the failure as an empty name list makes the card blame the channels of a
    stream the viewer never reached, i.e. the wrong one of the two reasons the whole
    eager probe exists to separate.
    """
    stream = descriptor(name="device", stype="eeg", source_id="unit-1")
    identity = stream.identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    _spy_probe(monkeypatch)
    window.reload_configurations()
    window._on_streams_found([stream])
    window._on_probe_failed(stream, "the inlet did not open")
    card = _card(window, "mine")
    assert card.state == STATE_UNAVAILABLE_NO_MATCH
    assert card._reason.text() == (
        "Could not reach device (eeg/unit-1): the inlet did not open"
    )


def test_refresh_does_not_reset_the_cards(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an available card stays available across a Refresh yet to land.

    Resetting the cards on a Refresh flickers every available configuration to
    unavailable and back on each click; the page-level indicator already carries that
    progress.
    """
    stream = descriptor(name="device")
    identity = stream.identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    _spy_probe(monkeypatch)
    monkeypatch.setattr(Discovery, "refresh", lambda self: None)
    window.reload_configurations()
    window._on_streams_found([stream])
    window._on_probed(stream, ["Fp1"])
    assert _card(window, "mine").state == STATE_AVAILABLE
    window.refresh()
    assert _card(window, "mine").state == STATE_AVAILABLE


def test_refresh_relists_the_configuration_directory(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a Refresh picks up a configuration which appeared or vanished.

    This is the replacement for a filesystem watcher and for the re-list a manage dialog
    would have done: listing only at construction leaves a deleted configuration on
    screen until the window is reopened, and a file written by a second window
    invisible.
    """
    identity = descriptor().identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    monkeypatch.setattr(Discovery, "refresh", lambda self: None)
    window.refresh()
    assert window._landing.configuration_names() == ("mine",)
    _write_config("later", [identity], {identity: ["Fp1"]})
    window.refresh()
    assert window._landing.configuration_names() == ("later", "mine")
    for path in config_home.rglob("mine*.json"):
        path.unlink()
    window.refresh()
    assert window._landing.configuration_names() == ("later",)


# -- saving ----------------------------------------------------------------------------
def test_save_as_writes_the_whole_workspace(
    app: QApplication,
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the envelope, the layout, the window block and the action enablement.

    One workspace and one connection covers all four, because they are one gesture. Two
    documents over **one** stream: a second document needs a second identity, not a
    second connection, and a connection is 1.68 s of liblsl handshake.

    What each half kills. The envelope: an order which disagrees between 'streams' and
    the presentation blocks, or a 'channels' mapping keyed by name, makes the
    availability check compare the wrong channel set. The layout: a regressed
    compression flag turns the value into a blob 'json.dumps' refuses, and a missing
    layout version writes 'UserVersion="0"' which refuses every layout the next release
    saves. The window block: reading 'geometry()' while maximized stores a full-screen
    'normal' size, so the restored window cannot be un-maximized to anything sensible.
    The actions: 'save_configuration' reaches a writer which refuses an empty stream
    list, i.e. raises inside a Qt slot.
    """
    assert not window._act_save.isEnabled()
    assert not window._act_save_as.isEnabled()
    first = _open(window, stream, source_id="unit-1")
    second = _open(window, stream, source_id="unit-2")
    assert window._act_save.isEnabled()
    assert window._act_save_as.isEnabled()
    window.showMaximized()
    app.processEvents()
    assert window.isMaximized()
    assert window.geometry() != window.normalGeometry()
    normal = window.normalGeometry()
    _stub_text(monkeypatch, "my workspace")
    window.save_configuration_as()

    (cfg,) = list_configurations()
    assert cfg.name == "my workspace"
    assert cfg.streams == [first.identity.as_tuple(), second.identity.as_tuple()]
    assert cfg.channels == {
        channel_key(doc.identity.as_tuple()): list(stream.ch_names)
        for doc in (first, second)
    }
    blocks = cfg.presentation["streams"]
    assert [tuple(block["identity"]) for block in blocks] == cfg.streams
    assert [block["slot"] for block in blocks] == [
        first.objectName(),
        second.objectName(),
    ]
    layout = cfg.presentation["layout"]
    assert layout.startswith("<?xml")
    assert f'UserVersion="{_window.LAYOUT_VERSION}"' in layout
    for doc in (first, second):
        assert doc.objectName() in layout
    assert cfg.presentation["window"] == {
        "x": normal.x(),
        "y": normal.y(),
        "width": normal.width(),
        "height": normal.height(),
        "maximized": True,
    }
    assert window._source == "my workspace"
    window.close_all_documents()
    assert not window._act_save.isEnabled()
    assert not window._act_save_as.isEnabled()


def test_save_overwrites_the_source_without_prompting(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a plain Save replaces the configuration this workspace came from.

    A '_source' which is never set makes every Save prompt, so the user accumulates a
    file per save; one which is not cleared when the file disappears makes Save write
    silently to a name no card shows.
    """
    _open(window, stream)
    prompts = _stub_text(monkeypatch, "mine")
    window.save_configuration_as()
    assert len(prompts) == 1
    (path,) = list(config_home.rglob("*.json"))
    window.save_configuration()
    assert len(prompts) == 1  # no second prompt: the name is already the user's
    assert list(config_home.rglob("*.json")) == [path]
    # the configuration disappeared behind the workspace's back.
    path.unlink()
    window.save_configuration()
    assert len(prompts) == 2  # it falls through to Save as


def test_save_as_prompt_policy(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the cancel, the blank name, the 120-character cap and the Replace question.

    Four policies of one prompt over one connection. A dropped cancel check saves a
    configuration the user backed out of; a dropped blank check writes a card with no
    title which can be neither identified nor renamed; a dropped cap carries 300
    characters into every card label and every unavailability reason; and a dropped
    Replace question makes Save as silently overwrite a configuration the user did not
    name.
    """
    _open(window, stream)
    _stub_text(monkeypatch, "cancelled", accepted=False)
    window.save_configuration_as()
    assert list_configurations() == []

    _stub_text(monkeypatch, "   ")
    window.save_configuration_as()
    assert list_configurations() == []

    _stub_text(monkeypatch, "n" * 300)
    window.save_configuration_as()
    (cfg,) = list_configurations()
    assert cfg.name == "n" * 120

    # a colliding name, refused: one question, and the other configuration untouched.
    _write_config("taken", [("a", "b", "c")])
    _stub_text(monkeypatch, "taken")
    asked = _stub_box(monkeypatch, "question", QMessageBox.StandardButton.No)
    window.save_configuration_as()
    assert len(asked) == 1
    assert "taken" in asked[0]
    by_name = {entry.name: entry for entry in list_configurations()}
    assert by_name["taken"].streams == [("a", "b", "c")]

    asked = _stub_box(monkeypatch, "question", QMessageBox.StandardButton.Yes)
    window.save_configuration_as()
    assert len(asked) == 1
    by_name = {entry.name: entry for entry in list_configurations()}
    assert by_name["taken"].streams == [window.documents[0].identity.as_tuple()]


def test_save_reports_a_disconnected_stream(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a document whose stream went away is saved, and said to be down.

    Blocking the save would let a device which will be back tomorrow lose a workspace
    today: the identity and every setting are still known, and a configuration describes
    a *desired* workspace. Dropping the notice instead saves silently and leaves the
    user with no way to tell that a stream was already gone.

    The count is read off the document's *state* and not off 'stream.connected', which
    is why the tick is here: the state moves on a render tick, and the 30 Hz clock has
    delivered thousands of them by the time a user reaches the menu. Reading the stream
    instead would report nothing at all for a stalled document -- which is connected --
    and would call a refused one merely disconnected.
    """
    _open(window, stream)
    stream.disconnect()
    window.documents[0].trace._render()  # one tick, as the render clock provides
    assert window.documents[0].state != LIVE
    _stub_text(monkeypatch, "mine")
    window.save_configuration_as()
    (cfg,) = list_configurations()
    assert cfg.streams
    message = window.statusBar().currentMessage()
    assert "Saved 'mine'." in message
    assert "1 stream currently interrupted." in message


# -- loading ---------------------------------------------------------------------------
def _save_two_document_workspace(
    window: ViewerWindow, stream: StreamLSL, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, list[tuple[StreamDescriptor, StreamLSL]]]:
    """Save a two-document workspace and return its name and the load's connections.

    Two documents over **one** stream, under two source IDs: a document needs a second
    identity and not a second connection. The same stream object is handed back for both
    identities, which is exactly what the connector's slot would do for two streams.

    The workspace which is saved *borrows* the stream, so that closing it leaves the
    stream connected for the load to be handed. A load always owns what it was given,
    thus the documents it builds do disconnect on the way out.
    """
    first = _borrow(window, stream, source_id="unit-1")
    second = _borrow(window, stream, source_id="unit-2")
    _stub_text(monkeypatch, "mine")
    window.save_configuration_as()
    pairs = [
        (_descriptor_for(stream, source_id=doc.identity.source_id), stream)
        for doc in (first, second)
    ]
    window._descriptors = {
        descriptor.identity.as_tuple(): descriptor for descriptor, _ in pairs
    }
    return "mine", pairs


def test_load_end_to_end(
    config_home: Path,
    window: ViewerWindow,
    qtbot: QtBot,
    outlets: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a real save, close and reopen against a real outlet on the network.

    The one load which goes through the loader thread and a real connection: a missing
    loader connection, a slot which never reserves its name, a purge which never runs or
    a document which is never published all surface here. The **state equality** is what
    makes it more than a smoke test -- the reopened document has to report exactly the
    presentation state which was written, slot and identity included.
    """
    _spy_probe(monkeypatch)  # the channel probe is not what this test is about
    descriptor = outlets(n_channels=3, ch_names=["Fp1", "Fp2", "Cz"])
    live = connect_stream(descriptor, 4.0)
    _open(window, live)
    window.documents[0].model.rename(0, "Renamed")
    _stub_text(monkeypatch, "mine")
    window.save_configuration_as()
    (cfg,) = list_configurations()
    saved_block = cfg.presentation["streams"][0]
    saved_layout = cfg.presentation["layout"]

    window.close_all_documents()
    assert window.documents == ()
    assert not live.connected  # the document owned it and disconnected it
    window.reload_configurations()
    window._on_streams_found([descriptor])
    window.open_configuration("mine")
    qtbot.waitUntil(lambda: len(window.documents) == 1, timeout=25000)
    doc = window.documents[0]
    assert doc.identity == descriptor.identity
    assert window._stack.currentWidget() is window._dock_host
    assert window._source == "mine"
    restored = doc.capture_state()
    # The controller width is compared apart on purpose: it is whatever the splitter's
    # own minimum-size negotiation granted in this layout, not a number the document
    # stores, and the saved workspace was laid out in a different one. Everything else
    # has to match exactly, and 'test_apply_state_round_trip' pins the width too, over
    # one layout.
    assert restored.pop("controller")["visible"] == saved_block["controller"]["visible"]
    assert doc.controller_width > 0
    del saved_block["controller"]
    assert restored == saved_block
    assert doc.model.channel(0).name == "Renamed"
    # the slot was reused, thus the saved layout still joins and re-saves identically.
    assert (
        bytes(window._manager.saveState(_window.LAYOUT_VERSION)).decode("utf-8")
        == saved_layout
    )


def test_load_restores_the_slots_and_reserves_them(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a load reuses the saved object names and keeps the counter past them.

    A counter which is not reserved grants a name a saved layout already owns on the
    next incremental Open, and a duplicate object name overwrites the manager's map
    entry and makes every later 'saveState()' name that slot several times -- which
    restores into as many phantom dock areas, survives a save/load cycle and is
    unrecoverable from the interface.
    """
    _spy_open(monkeypatch)
    name, pairs = _save_two_document_workspace(window, stream, monkeypatch)
    slots = [doc.objectName() for doc in window.documents]
    window.close_all_documents()
    _drive_load(window, name, pairs)
    assert [doc.objectName() for doc in window.documents] == slots
    assert window._next_index > max(int(slot.split("-")[1]) for slot in slots)


def test_load_purges_closed_documents_first(
    config_home: Path,
    window: ViewerWindow,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that loading twice in one process leaves one map entry per document.

    Without the purge the second load re-registers the saved slot names as duplicates,
    which -- measured -- overwrites the map entries and makes the next 'saveState()'
    name one slot four times. The byte-identical re-save is the exact assertion, and it
    is only exact because the load rebuilds in the saved order.
    """
    _spy_open(monkeypatch)
    first_stream, _ = lsl_stream()
    name, pairs = _save_two_document_workspace(window, first_stream, monkeypatch)
    saved = bytes(window._manager.saveState(_window.LAYOUT_VERSION)).decode("utf-8")
    window.close_all_documents()
    _drive_load(window, name, pairs)
    assert len(window.documents) == 2
    window.close_all_documents()

    # a fresh connection: the documents of the first load owned and disconnected the one
    # above when they were closed.
    second_stream, _ = lsl_stream()
    again = [(descriptor, second_stream) for descriptor, _ in pairs]
    _drive_load(window, name, again)
    assert len(window.documents) == 2
    assert len(window._manager.dockWidgetsMap()) == 2
    assert (
        bytes(window._manager.saveState(_window.LAYOUT_VERSION)).decode("utf-8")
        == saved
    )


def test_restore_layout_always_leaves_every_document_open(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that no saved layout, however broken, can lose a document.

    Three ways a layout fails, all of them measured. Unparsable XML is refused and
    closes nothing. A layout whose version was bumped is refused the same way. And an
    XML naming only some of the registered widgets is *accepted* -- 'restoreState'
    returns True -- while leaving the others closed, so one document disappears with no
    error at all; an XML naming none of them closes every document and returns True over
    an empty workspace.

    The re-add loop is what catches all three, and none of them is a load failure: all
    or nothing applies to streams and documents, never to widget placement, so the
    outcome is one non-modal note and no dialog.
    """
    _spy_open(monkeypatch)
    name, pairs = _save_two_document_workspace(window, stream, monkeypatch)
    docs = list(window.documents)
    good = bytes(window._manager.saveState(_window.LAYOUT_VERSION)).decode("utf-8")
    bumped = good.replace(
        f'UserVersion="{_window.LAYOUT_VERSION}"',
        f'UserVersion="{_window.LAYOUT_VERSION + 1}"',
    )
    renamed = good.replace(docs[1].objectName(), "stream-does-not-exist")
    for layout in ("not xml at all", bumped, renamed, "", None):
        window.statusBar().clearMessage()
        window._restore_layout(layout, docs)
        for doc in docs:
            assert not doc.isClosed(), layout
        assert window.findChildren(QMessageBox) == []
    # only a layout which was present and refused says so.
    window.statusBar().clearMessage()
    window._restore_layout("not xml at all", docs)
    assert "shown as tabs" in window.statusBar().currentMessage()
    window.statusBar().clearMessage()
    window._restore_layout(None, docs)
    assert window.statusBar().currentMessage() == ""


def test_load_rolls_back_a_failed_connection(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that one failed connection undoes the whole load, with exactly one dialog.

    A rollback which forgets the streams leaks a live inlet and its acquisition thread
    for the life of the process, and a dialog moved into the per-stream failure slot
    opens one modal per stream for a single click.
    """
    _spy_open(monkeypatch)
    name, pairs = _save_two_document_workspace(window, stream, monkeypatch)
    window.close_all_documents()
    shown = _stub_box(monkeypatch, "critical", QMessageBox.StandardButton.Ok)
    refreshed: list[int] = []
    monkeypatch.setattr(Discovery, "refresh", lambda self: refreshed.append(1))
    _drive_load(
        window,
        name,
        connected=[pairs[0]],
        failed=[(pairs[1][0], "0 were found: [].")],
    )
    assert window.documents == ()
    assert window._stack.currentWidget() is window._landing
    assert window._loading is None
    assert not stream.connected  # the sibling which did connect was released
    assert len(shown) == 1
    assert "0 were found" in shown[0]
    assert refreshed == [1]
    assert window._act_refresh.isEnabled()


def test_load_dialog_is_one_per_click_whatever_the_number_of_failures(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that two failed connections still produce exactly one dialog.

    A five-stream configuration whose device is off must not open five modals for one
    click, and the per-stream messages have to survive into the one which is shown.
    """
    _spy_open(monkeypatch)
    name, pairs = _save_two_document_workspace(window, stream, monkeypatch)
    window.close_all_documents()
    shown = _stub_box(monkeypatch, "critical", QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(Discovery, "refresh", lambda self: None)
    _drive_load(
        window,
        name,
        connected=[],
        failed=[(pairs[0][0], "first is down"), (pairs[1][0], "second is down")],
    )
    assert len(shown) == 1
    assert "first is down" in shown[0]
    assert "second is down" in shown[0]


def test_load_rolls_back_a_channel_mismatch(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a shrunk channel set is caught against the *connected* metadata.

    Skipping the check opens the workspace over a stream which no longer carries what
    the configuration described, and the Channels page then shows a different channel
    set with nothing to explain it. The count and a name have to reach the dialog, or
    the user cannot tell which stream to look at.
    """
    _spy_open(monkeypatch)
    name, pairs = _save_two_document_workspace(window, stream, monkeypatch)
    window.close_all_documents()
    # the saved contract names a channel the stream does not publish.
    (cfg,) = list_configurations()
    key = channel_key(pairs[0][0].identity.as_tuple())
    cfg.channels[key] = [*cfg.channels[key], "GoneAway"]
    save_configuration(cfg)
    window.reload_configurations()
    shown = _stub_box(monkeypatch, "critical", QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(Discovery, "refresh", lambda self: None)
    _drive_load(window, name, pairs)
    assert window.documents == ()
    assert not stream.connected
    assert len(shown) == 1
    assert "no longer provides 1 of its saved channels (GoneAway)." in shown[0]


def test_load_releases_a_late_connection(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
) -> None:
    """Test that a connection landing after a rollback is disconnected, not adopted.

    The connector transfers stream ownership with its signal, thus the 'not my attempt'
    branch is the only thing between a cancelled load and one leaked inlet plus its
    acquisition thread, for the life of the process. No connection is made here at all.
    """
    late = _DummyStream()
    assert window._loading is None
    window._on_load_connected(descriptor(), late)
    assert late.disconnected == 1
    assert window.documents == ()


def test_load_refuses_a_vanished_identity_before_connecting(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a stream which left between the card and the click is refused up front.

    The race is one event-loop turn wide, so this is defence -- and it is the difference
    between one dialog and a 'KeyError' raised inside a Qt slot, which aborts the
    process for an embedder. Nothing may be connected either.
    """
    gone = descriptor(name="gone", stype="eeg", source_id="unit-9")
    identity = gone.identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    calls = _spy_open(monkeypatch)
    shown = _stub_box(monkeypatch, "critical", QMessageBox.StandardButton.Ok)
    window.reload_configurations()
    window._on_streams_found([])  # the network reports nothing
    window.open_configuration("mine")
    assert calls == []
    assert window._loading is None
    assert len(shown) == 1
    assert shown[0] == "gone (eeg/unit-9) is no longer on the network."


# -- the restored window geometry ------------------------------------------------------
def test_restore_geometry_applies_an_on_screen_rect(
    app: QApplication, window: ViewerWindow
) -> None:
    """Test that a rectangle inside a real screen is applied verbatim.

    A defensive loop which always falls through to the clamp makes every configuration
    reopen centred at the same size, i.e. the saved geometry means nothing at all.
    """
    available = QGuiApplication.primaryScreen().availableGeometry()
    target = QRect(available.x() + 20, available.y() + 20, 400, 300)
    _window._restore_geometry(
        window,
        {
            "x": target.x(),
            "y": target.y(),
            "width": target.width(),
            "height": target.height(),
            "maximized": False,
        },
    )
    app.processEvents()
    assert window.geometry() == target


def test_restore_geometry_clamps_an_off_screen_rect(
    app: QApplication, window: ViewerWindow
) -> None:
    """Test that a rectangle no screen can show is clamped onto the primary one.

    A monitor which was unplugged restores the window somewhere it cannot be seen and,
    worse, cannot be grabbed by its title bar to be brought back. Clamped silently: an
    unplugged screen is not an error the user has to acknowledge.
    """
    available = QGuiApplication.primaryScreen().availableGeometry()
    _window._restore_geometry(
        window,
        {"x": -9000, "y": -9000, "width": 400, "height": 300, "maximized": False},
    )
    app.processEvents()
    assert available.intersected(window.geometry()).width() >= _window._MIN_VISIBLE_W
    assert available.intersected(window.geometry()).height() >= _window._MIN_VISIBLE_H
    assert window.width() <= available.width()
    assert window.height() <= available.height()


def test_restore_geometry_restores_the_maximized_state(
    app: QApplication, window: ViewerWindow
) -> None:
    """Test that a workspace saved maximized comes back maximized."""
    available = QGuiApplication.primaryScreen().availableGeometry()
    _window._restore_geometry(
        window,
        {
            "x": available.x() + 10,
            "y": available.y() + 10,
            "width": 400,
            "height": 300,
            "maximized": True,
        },
    )
    app.processEvents()
    assert window.isMaximized()


@pytest.mark.parametrize(
    "state",
    [
        "not a mapping",
        {"x": 0, "y": 0, "width": 400},  # a missing field
        {"x": 0, "y": 0, "width": "400", "height": 300},  # a string
        {
            "x": 0,
            "y": 0,
            "width": True,
            "height": 300,
        },  # a bool passes 'isinstance(int)'
        {"x": 0, "y": 0, "width": 0, "height": 300},  # a degenerate size
        {"x": 0, "y": 0, "width": -400, "height": 300},
    ],
)
def test_restore_geometry_skips_an_unusable_state(
    app: QApplication, window: ViewerWindow, state: object
) -> None:
    """Test that a hand-edited window block is skipped whole rather than half-applied.

    Half a rectangle is worse than none, and this is a file the user can edit: applying
    what parses would move the window without resizing it, or resize it to nothing. No
    connection is made in any of these cases.
    """
    before = window.geometry()
    _window._restore_geometry(window, state)
    app.processEvents()
    assert window.geometry() == before


# -- renaming and deleting -------------------------------------------------------------
def test_delete_asks_before_removing(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a delete is confirmed once, and that refusing it keeps the file.

    There is no undo, so an unconfirmed delete destroys a workspace on a misclick; and
    the card must disappear on the confirmed one, or the interface keeps offering a file
    which is gone.
    """
    identity = descriptor().identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    monkeypatch.setattr(Discovery, "refresh", lambda self: None)
    window.reload_configurations()
    refused = _stub_box(monkeypatch, "question", QMessageBox.StandardButton.No)
    window._delete_configuration("mine")
    assert len(refused) == 1
    assert "mine" in refused[0]
    assert [cfg.name for cfg in list_configurations()] == ["mine"]
    _stub_box(monkeypatch, "question", QMessageBox.StandardButton.Yes)
    window._delete_configuration("mine")
    assert list_configurations() == []
    assert window._landing.configuration_names() == ()


def test_rename_follows_the_source_and_reports_a_collision(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the workspace's source name follows a rename, and that a clash warns.

    A source which does not follow makes the next plain Save write a *third* file under
    the old name; a collision reported as anything but a warning would fall through to a
    write which overwrites the configuration the user did not name.
    """
    identity = descriptor().identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    _write_config("other", [identity], {identity: ["Fp1"]})
    monkeypatch.setattr(Discovery, "refresh", lambda self: None)
    window.reload_configurations()
    window._source = "mine"

    _stub_text(monkeypatch, "renamed")
    window._rename_configuration("mine")
    assert window._source == "renamed"
    assert sorted(cfg.name for cfg in list_configurations()) == ["other", "renamed"]
    assert window._landing.configuration_names() == ("other", "renamed")

    warned = _stub_box(monkeypatch, "warning", QMessageBox.StandardButton.Ok)
    _stub_text(monkeypatch, "other")
    window._rename_configuration("renamed")
    assert len(warned) == 1
    assert "already exists" in warned[0]
    assert window._source == "renamed"
    assert sorted(cfg.name for cfg in list_configurations()) == ["other", "renamed"]


def test_rename_cancelled_changes_nothing(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that dismissing the rename prompt writes nothing.

    A missing check on the prompt's own return value renames to whatever text the field
    happened to hold when the user pressed Cancel.
    """
    identity = descriptor().identity.as_tuple()
    _write_config("mine", [identity], {identity: ["Fp1"]})
    _stub_text(monkeypatch, "renamed", accepted=False)
    window._rename_configuration("mine")
    assert [cfg.name for cfg in list_configurations()] == ["mine"]


def test_loading_takes_over_its_own_card_only(
    config_home: Path,
    window: ViewerWindow,
    stream: StreamLSL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the card being opened reads 'Opening' and every sibling goes inert.

    Dropping the substitution leaves the loading card available and clickable, which is
    what lets a second click start a second load attempt -- and the second one strands
    the streams the first has already connected. The state is imposed by the window and
    not by the availability check, which knows nothing about a load being in flight.
    """
    _spy_open(monkeypatch)
    _spy_probe(monkeypatch)
    name, pairs = _save_two_document_workspace(window, stream, monkeypatch)
    window.close_all_documents()
    other = pairs[0][0]
    _write_config(
        "sibling", [other.identity.as_tuple()], {other.identity.as_tuple(): ["ch0"]}
    )
    window.reload_configurations()
    window._on_streams_found([descriptor for descriptor, _ in pairs])
    window._on_probed(other, list(stream.ch_names))
    assert _card(window, "sibling").state == STATE_AVAILABLE

    window.open_configuration(name)
    assert _card(window, name).state == STATE_LOADING
    assert _card(window, name)._reason.text() == "Connecting…"
    assert not _card(window, name).activatable
    assert not _card(window, "sibling").activatable
    assert not window._act_refresh.isEnabled()
    # finish the load, so the fixture tears a settled window down.
    for descriptor, live in pairs:
        window._on_load_connected(descriptor, live)
    assert len(window.documents) == 2
    assert window._act_refresh.isEnabled()


def test_open_configuration_refuses_an_invalid_one(
    config_home: Path, window: ViewerWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a corrupt configuration is never opened, and connects nothing.

    Its card is not activatable, so this is defence -- and the payload of an invalid
    configuration is empty, thus a load which went ahead would connect nothing, publish
    nothing and leave the workspace on a landing page with no explanation at all.

    The four assertions below all hold in the untouched state, which is why the dialog
    is stubbed and asserted on: without the guard the empty payload takes the
    'references no stream' branch, which also connects, publishes and clears nothing, so
    the *only* observable difference is the message box. Unstubbed, that box used to
    make the mutated test hang rather than fail -- a modal blocks forever and
    'pytest-timeout''s signal method cannot interrupt one.
    """
    directory = config_home / ".mne-lsl" / "viewer" / "configurations"
    directory.mkdir(parents=True)
    (directory / "broken.json").write_text("{", encoding="utf-8")
    calls = _spy_open(monkeypatch)
    shown = _stub_box(monkeypatch, "critical", QMessageBox.StandardButton.Ok)
    window.reload_configurations()
    assert _card(window, "broken").state == STATE_INVALID
    window.open_configuration("broken")
    assert calls == []
    assert window._loading is None
    assert window.documents == ()
    assert shown == []


def test_open_configuration_refuses_an_event_only_configuration(
    config_home: Path,
    window: ViewerWindow,
    descriptor: Callable[..., StreamDescriptor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a configuration of event sources alone is refused instead of hanging.

    Only reachable from a hand-edited file -- a saved workspace always holds at least
    one document -- and it has to be refused rather than submitted: an empty batch makes
    the connector emit nothing at all, so the load never finishes and the card stays on
    'Connecting…' with Refresh disabled for the rest of the session.
    """
    markers = descriptor(name="markers", stype="annotations", sfreq=0.0)
    identity = markers.identity.as_tuple()
    _write_config("events only", [identity])
    calls = _spy_open(monkeypatch)
    shown = _stub_box(monkeypatch, "critical", QMessageBox.StandardButton.Ok)
    window.reload_configurations()
    window._on_streams_found([markers])
    assert _card(window, "events only").state == STATE_AVAILABLE
    window.open_configuration("events only")
    assert calls == []
    assert window._loading is None
    assert window._act_refresh.isEnabled()
    assert len(shown) == 1
    assert "no stream this viewer can open" in shown[0]
