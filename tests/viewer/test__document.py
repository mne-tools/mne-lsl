from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne._fiff.constants import FIFF
from qtpy.QtWidgets import QMainWindow

from mne_lsl.viewer import _document
from mne_lsl.viewer._bootstrap import configure_docking, import_ads
from mne_lsl.viewer._document import (
    CLOSED,
    INTERRUPTED,
    LIVE,
    MISMATCHED,
    StreamDocument,
)
from mne_lsl.viewer.backend import (
    RESUME_LIVE,
    RESUME_MISMATCH,
    RESUME_RETRY,
    StreamIdentity,
    reconnect_stream,
)
from mne_lsl.viewer.backend._discovery import _ReconnectSignals
from mne_lsl.viewer.controller import ChannelsPage
from mne_lsl.viewer.display import TraceDisplay
from mne_lsl.viewer.theme import tokens

ads = import_ads()

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pytestqt.qtbot import QtBot
    from qtpy.QtWidgets import QApplication

    from mne_lsl.stream import StreamLSL
    from mne_lsl.viewer.theme import ThemeController

# Channel names whose alphabetical order differs from the acquisition order.
# Load-bearing: on the default 'ch0...ch6 + STI0' names, an alphabetical reorder returns
# the acquisition order and a reorder test then asserts nothing at all.
_SCRAMBLED = ["zeta", "alpha", "mike", "bravo", "yankee", "charlie", "delta", "STI0"]

_FIELDS = frozenset({"state", "channels", "sfreq", "history", "latency"})


@pytest.fixture
def manager(flush_deletes: Callable[..., None]) -> Generator[ads.CDockManager]:
    """Yield a dock manager on its own host window; a document requires one.

    The flags are set before the manager is constructed because the constructor consumes
    them, and they are process-global: this is never written as a flip-and-restore.
    """
    configure_docking()
    host = QMainWindow()
    host.resize(1200, 700)
    built = ads.CDockManager(host)
    yield built
    host.close()
    flush_deletes(host)


@pytest.fixture
def document(
    manager: ads.CDockManager,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    flush_deletes: Callable[..., None],
) -> Generator[Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]]]:
    """Yield a factory building documents over a real stream, torn down afterwards.

    The factory takes the ``lsl_stream`` overrides plus ``owns_stream`` and returns
    ``(document, stream, push)``. The identity is built here from the stream's own
    attributes rather than with 'backend.stream_identity': that helper is code of this
    phase and must not be what validates the rest of it.
    """
    created: list[StreamDocument] = []

    def _make(
        *, owns_stream: bool = True, **kwargs: object
    ) -> tuple[StreamDocument, StreamLSL, Callable[..., None]]:
        """Build one document over a fresh stream and register it for teardown."""
        stream, push = lsl_stream(**kwargs)
        identity = StreamIdentity(stream.name, stream.stype, stream.source_id)
        doc = StreamDocument(manager, stream, identity, owns_stream=owns_stream)
        created.append(doc)
        return doc, stream, push

    yield _make
    for doc in reversed(created):
        doc.teardown()  # idempotent, and a no-op for a document already closed
    flush_deletes(*reversed(created))
    created.clear()


def _dock(manager: ads.CDockManager, doc: StreamDocument, index: int = 0) -> None:
    """Register ``doc`` in ``manager``, as the window's own registration does."""
    doc.setObjectName(f"stream-{index}")
    manager.addDockWidget(ads.DockWidgetArea.CenterDockWidgetArea, doc)


def _stim_row(doc: StreamDocument) -> int:
    """Return the model row of the stim channel of ``doc``."""
    model = doc.model
    return next(
        row for row in range(model.rowCount()) if model.channel(row).ch_type == "stim"
    )


# -- the composition: the model -> display edges --------------------------------------
def test_initial_layout_push(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the document states the layout once, on construction.

    Pinned with a call spy and never with a row count: a fresh model produces every
    channel visible in acquisition order and the display already starts there, thus the
    two initial states are identical and an 'n_rows' assertion passes with the push
    deleted. The push is what keeps the model the single owner of the order and the
    visibility if either default ever moves.
    """
    calls: list[list[int]] = []
    monkeypatch.setattr(
        TraceDisplay,
        "set_channel_layout",
        lambda self, rows: calls.append(list(rows)),
    )
    doc, stream, _ = document()
    assert calls == [list(range(len(stream.ch_names)))]
    assert doc.trace.n_channels == len(stream.ch_names)


def test_layout_edge(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that every visibility change reaches the display, freshly read each time.

    The second mutation is what kills a '_push_layout' which cached the first list: the
    display would then keep the layout of the first change forever.
    """
    doc, _, _ = document()
    total = doc.trace.n_channels
    doc.model.set_visible([0], False)
    assert doc.trace.n_rows == total - 1
    assert doc.trace._rows == doc.model.visible_acq_indices()
    doc.model.set_visible([3], False)
    assert doc.trace.n_rows == total - 2
    assert doc.trace._rows == doc.model.visible_acq_indices()


def test_layout_edge_emits_changed(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a layout change reports itself, so the status bar can follow.

    Without the emission the status bar keeps the channel count it had when the document
    opened -- a plausible-looking number, which is how it survives a look at the screen.
    """
    doc, _, _ = document()
    seen: list[object] = []
    doc.changed.connect(seen.append)
    doc.model.set_visible([0], False)
    assert seen == [doc]
    doc.model.set_visible([1], False)
    assert seen == [doc, doc]


def test_reorder_reorders_without_recolouring(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a reorder moves the rows and recolors nothing.

    The color is seeded by the acquisition index, never by the row, thus the mapping
    from acquisition index to color survives the reorder while the color *of a row*
    changes, because the channel under it changed. A row-seeded color inverts both.
    """
    doc, _, _ = document(ch_names=_SCRAMBLED)
    trace = doc.trace
    rows = range(trace.n_rows)
    before = {trace._rows[row]: trace.color_for(row).name() for row in rows}
    first_row = trace.color_for(0).name()
    doc.model.order_by("alphabetical")
    assert trace._rows != list(range(trace.n_channels))
    assert [trace.channel_name(row) for row in rows] == sorted(
        _SCRAMBLED, key=str.casefold
    )
    after = {trace._rows[row]: trace.color_for(row).name() for row in rows}
    assert after == before
    assert trace.color_for(0).name() != first_row


def test_metadata_edge(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a metadata change reaches the display without touching the layout.

    'is_bad' is asserted rather than the axis label: the ``'X '`` prefix of a bad
    channel is the axis's, not this accessor's.
    """
    doc, _, _ = document()
    trace = doc.trace
    rows, gain = list(trace._rows), list(trace._gain)
    doc.model.set_bad([0], True)
    assert trace.is_bad(0)
    doc.model.rename(0, "RENAMED")
    assert trace.channel_name(0) == "RENAMED"
    doc.model.set_type([0], "ecg")
    assert trace._gain[0] != gain[0]
    assert trace._rows == rows  # a metadata change is not a layout change


def test_hidden_stim_keeps_events(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that hiding the stim channel drops its trace and keeps its event lines.

    The overlays belong to the Events page, not to channel visibility, thus the picks
    may never simply equal the layout.
    """
    doc, _, push = document()
    trace = doc.trace
    row = _stim_row(doc)
    acq = doc.model.channel(row).acq_index
    doc.model.set_visible([row], False)
    assert acq not in trace._rows
    assert acq in trace._picks
    assert trace._event_pos
    push(stim_at=10)
    trace._render()
    assert any(line.isVisible() for line in trace._event_lines)


def test_all_hidden_placeholder(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that hiding every channel shows the placeholder and renders quietly.

    'get_data(picks=[])' raises *and* logs a bug-report line, which at 30 Hz fills the
    terminal, thus the render must return before it.
    """
    doc, _, _ = document()
    rows = range(doc.model.rowCount())
    doc.model.set_visible(rows, False)
    assert doc.trace.n_rows == 0
    assert doc.trace._empty_label.isVisible()
    assert doc.trace._assigned == {}
    doc.trace._render()  # must not raise
    doc.model.set_visible(rows, True)
    assert doc.trace.n_rows == doc.trace.n_channels
    assert not doc.trace._empty_label.isVisible()


def test_model_ownership(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the document owns the model and the page merely borrows it.

    A model parented to the page would be destroyed with it, i.e. whenever the
    controller panel is closed, while the document keeps pushing layouts read from it.
    """
    doc, _, _ = document()
    assert doc.channels.model is doc.model
    assert doc.model.parent() is doc
    assert doc.isAncestorOf(doc.channels)
    assert doc.isAncestorOf(doc.trace)
    assert doc.channels.parent() is not doc.model


def test_single_controller_tab(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the controller panel holds the Channels page and nothing else.

    The Processing and Events pages never call their own 'super().__init__()', thus a
    tab added for either today holds a widget which raises on its first method call.
    """
    doc, _, _ = document()
    assert doc._panel.count() == 1
    assert doc._panel.tabText(0) == "Channels"
    assert doc._panel.widget(0) is doc.channels


def test_construction_starts_the_clock(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a constructed document is live, clock included.

    This object owns the live/frozen state while the clock lives in the display, thus a
    document whose caller has not started the clock yet reports 'Live' over a viewport
    which never advances.
    """
    doc, _, _ = document()
    assert doc.trace.running
    assert not doc.frozen
    assert doc.status_fields()["state"] == "Connected • Live"


def test_document_refuses_an_event_source(
    manager: ads.CDockManager,
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> None:
    """Test that an irregularly sampled stream cannot own a document.

    Refused at the choke point both entry points meet at: the window's own connect path
    skips an event source already, while the 'BaseStream.plot()' path would otherwise
    open a document whose time axis means nothing.
    """
    stream, _ = lsl_stream(n_channels=2, sfreq=0.0, bufsize=50)
    identity = StreamIdentity(stream.name, stream.stype, stream.source_id)
    with pytest.raises(ValueError, match="irregularly sampled"):
        StreamDocument(manager, stream, identity)


# -- freeze and the controller toggle -------------------------------------------------
def test_freeze(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that freezing stops the clock, relabels the button and reports once.

    The button is a pause button: it reads 'Freeze' while live and 'Live' while frozen.
    'changed' emitted twice here is the missing 'blockSignals' guard around the mirror,
    which bounces the programmatic call back through 'toggled'.
    """
    doc, _, _ = document()
    seen: list[object] = []
    doc.changed.connect(seen.append)

    doc.set_frozen(True)
    assert doc.frozen
    assert not doc.trace.running
    assert "Frozen" in doc._indicator.text()
    assert doc._freeze_button.text() == "Live"
    assert doc._freeze_button.isChecked()
    assert seen == [doc]

    doc.set_frozen(False)
    assert not doc.frozen
    assert doc.trace.running
    assert "Live" in doc._indicator.text()
    assert doc._freeze_button.text() == "Freeze"
    assert not doc._freeze_button.isChecked()
    assert seen == [doc, doc]


def test_indicator_stops_claiming_live_when_interrupted(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the indicator follows the connection state, not the freeze alone.

    Its tooltip promises it reports whether the viewport advances, and an interrupted
    viewport does not -- 'MISMATCHED' stops the clock outright. A green '● Live' over a
    viewport frozen on its last frame is the toolbar contradicting the banner directly
    above it, for the rest of the outage.

    The button keeps offering the operator's own toggle, which is a preference and
    applies on the resume: relabelling it here is what would take that away.
    """
    doc, _, _ = document()
    assert doc._indicator.text() == "● Live"

    doc._enter(INTERRUPTED, "Stream lost")
    assert doc._indicator.text() == "■ Interrupted"
    assert doc._freeze_button.text() == "Freeze"

    doc._enter(MISMATCHED, "The channels changed")
    assert doc._indicator.text() == "■ Interrupted"

    doc._go_live(time.monotonic())
    assert doc._indicator.text() == "● Live"


def test_freeze_button_round_trip(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the toolbar button drives the freeze, once per click."""
    doc, _, _ = document()
    seen: list[object] = []
    doc.changed.connect(seen.append)
    doc._freeze_button.click()
    assert doc.frozen
    assert not doc.trace.running
    assert len(seen) == 1
    doc._freeze_button.click()
    assert not doc.frozen
    assert doc.trace.running
    assert len(seen) == 2


def test_frozen_viewport_does_not_advance(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that hiding a channel while frozen leaves the frozen window on screen.

    Freeze stops the render clock and nothing else, thus a layout push used to run a
    fetch of its own and jump the viewport to the newest samples: the frozen document
    silently showed twice the samples it had been frozen on, still labelled 'Frozen'.

    'equal_nan=True' is mandatory on both comparisons: fewer samples are pushed than the
    window holds, so the un-filled prefix is drawn as NaN, and 'np.array_equal' is
    'False' for two *identical* NaN-bearing arrays -- which fails the positive assertion
    and makes the negative one hold whatever the document does.
    """
    doc, _, push = document()
    trace = doc.trace
    push(50)
    trace._render()
    row = trace._rows.index(5)
    before = trace._assigned[row].getData()[1].copy()

    doc.set_frozen(True)
    push(50)  # the acquisition keeps running while the viewport is frozen
    doc.model.set_visible([0], False)
    assert doc.frozen
    assert not trace.running
    row = trace._rows.index(5)
    assert np.array_equal(trace._assigned[row].getData()[1], before, equal_nan=True)

    doc.set_frozen(False)
    trace._render()
    row = trace._rows.index(5)
    assert not np.array_equal(trace._assigned[row].getData()[1], before, equal_nan=True)


def test_frozen_rows_change_keeps_the_traces(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the row stepper repaints a frozen document instead of emptying it.

    The row count rebuilds the curve pool, thus a frozen document left the plot with no
    curves and no placeholder while the status bar went on reporting '4/16 ch'.
    """
    doc, _, push = document(n_channels=16)
    trace = doc.trace
    push(50)
    trace._render()
    doc.set_frozen(True)
    trace.controls.set_rows(4)
    assert doc.status_fields()["channels"] == "4/16 ch"
    assert not trace._empty_label.isVisible()
    assert trace._assigned
    for drawn, curve in trace._assigned.items():
        assert curve.getData()[1].size, drawn


def test_controller_visible(
    app: QApplication,
    manager: ads.CDockManager,
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that hiding the controller collapses it and showing it restores its width.

    The document is docked and the host shown on purpose: a splitter holds no real size
    until it has been laid out.

    The width coming back is the splitter's own memory of a hidden child, measured and
    not assumed -- the document deliberately saves nothing. What this pins is therefore
    the user-visible contract, and an explicit 'setSizes' added to the toggle is what
    would break it.
    """
    doc, _, _ = document()
    assert doc.controller_visible
    _dock(manager, doc)
    manager.window().show()
    app.processEvents()
    before = doc._splitter.sizes()
    assert before[0] > 0

    doc.set_controller_visible(False)
    app.processEvents()
    assert not doc.controller_visible
    assert doc._panel.isHidden()
    assert not doc._controller_button.isChecked()
    assert doc._splitter.sizes()[0] == 0

    doc.set_controller_visible(True)
    app.processEvents()
    assert doc.controller_visible
    assert doc._controller_button.isChecked()
    assert doc._splitter.sizes() == before


def test_controller_width_of_a_collapsed_panel(
    app: QApplication,
    manager: ads.CDockManager,
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a panel dragged to nothing is not what a configuration saves.

    A splitter lets its children collapse, so this one can be dragged to zero while
    staying perfectly *shown* -- the hidden-panel fallback does not cover it. Saving the
    zero writes a configuration which opens with no Channels page in every session which
    loads it, while the collapse it came from lasts as long as the window.
    """
    doc, _, _ = document()
    _dock(manager, doc)
    manager.window().show()
    app.processEvents()
    assert doc._splitter.sizes()[0] > 0

    doc._splitter.setSizes([0, sum(doc._splitter.sizes())])  # dragged onto the edge
    app.processEvents()
    assert doc.controller_visible  # shown, and 0 wide
    assert doc._splitter.sizes()[0] == 0
    assert doc.controller_width == doc._panel_width > 0
    assert doc.capture_state()["controller"] == {
        "visible": True,
        "width": doc._panel_width,
    }


def test_controller_button_round_trip(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that the toolbar button drives the controller visibility both ways."""
    doc, _, _ = document()
    assert doc._controller_button.isChecked()
    doc._controller_button.click()
    assert not doc.controller_visible
    doc._controller_button.click()
    assert doc.controller_visible


# -- the status-bar fields ------------------------------------------------------------
def test_status_fields_live(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test the fields of a live document, counts and formatting included.

    The channel counts are asserted with the displayed count differing from the total
    and then with the viewport smaller than the layout, which is what a swap of the two,
    or a read of either alone, cannot survive.
    """
    doc, _, _ = document(bufsize=6.0)
    fields = doc.status_fields()
    assert set(fields) == _FIELDS
    assert fields["state"] == "Connected • Live"
    assert fields["channels"] == "8/8 ch"
    assert fields["sfreq"] == "100 Hz"
    assert fields["history"] == "6 s history"
    assert fields["latency"] == "No processing • 0 ms"

    doc.model.set_visible([0, 1, 2], False)
    assert doc.status_fields()["channels"] == "5/8 ch"
    doc.trace.controls.set_rows(3)
    assert doc.status_fields()["channels"] == "3/8 ch"

    doc.set_frozen(True)
    assert doc.status_fields()["state"] == "Connected • Frozen"


def test_status_fields_disconnected(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a disconnected stream reports its state without being read.

    Both 'info' and 'n_buffer' raise on a disconnected stream, and this is the path the
    connection-loss handling walks into.
    """
    doc, stream, _ = document()
    stream.disconnect()
    fields = doc.status_fields()
    assert set(fields) == _FIELDS
    assert fields["state"] == "Disconnected"
    for key in ("channels", "sfreq", "history", "latency"):
        assert fields[key] == "—", key


def test_status_fields_unreadable_under_the_read(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a stream reset under the read is reported as the disconnection it is.

    'connected' answers off the private attributes, so it is still True while the
    acquisition thread is part-way through clearing them and the read which follows
    raises anyway -- out of the status bar's own slot. Kills both halves: dropping the
    guard, and reporting the state before the read, which claims 'Connected • Live' next
    to five blanks.
    """
    doc, stream, _ = document()

    def _raise(self) -> None:
        raise RuntimeError("The Stream is not connected.")

    monkeypatch.setattr(type(stream), "info", property(_raise))
    assert stream.connected  # the attributes the property reads are still set
    fields = doc.status_fields()
    assert fields["state"] == "Disconnected"
    for key in ("channels", "sfreq", "history", "latency"):
        assert fields[key] == "—", key


# -- teardown -------------------------------------------------------------------------
def test_teardown_owned(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that tearing an owned document down stops the clock and the stream.

    The second call is not decoration: a second 'disconnect()' raises 'RuntimeError',
    and the window's close loop reaches every document after its own 'closed' handler
    already did. Both children are asserted closed through their theme connection, which
    is what their own 'closeEvent' drops.
    """
    doc, stream, _ = document()
    doc.teardown()
    assert not doc.trace.running
    assert not doc.trace._following_theme
    assert not doc.channels._following_theme
    assert not stream.connected
    doc.teardown()
    assert not stream.connected


def test_teardown_closes_each_child_once(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a second teardown closes neither child again.

    The idempotence guard is load-bearing and no state the other teardown tests read can
    tell: 'TraceDisplay.closeEvent' latches whether it stopped a *running* clock, which
    is what a later 'showEvent' restarts it from, so a second close latches 'False' over
    it and the display comes back frozen on its last frame forever.
    """
    closed: list[str] = []
    for cls, name in ((TraceDisplay, "trace"), (ChannelsPage, "channels")):
        original = cls.close

        def _close(self, _original=original, _name=name) -> bool:
            closed.append(_name)
            return _original(self)

        monkeypatch.setattr(cls, "close", _close)
    doc, _, _ = document()
    doc.teardown()
    assert closed == ["trace", "channels"]
    assert doc.trace._was_running
    doc.teardown()
    assert closed == ["trace", "channels"]
    assert doc.trace._was_running


def test_teardown_closes_the_children_before_the_stream(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the stream is released only once both children are closed.

    The order is the whole point of the sequence: a render tick or a metadata read
    arriving between the disconnect and the display's close would reach a stream going
    away, and 'BaseStream' raises on nearly every attribute there.
    """
    doc, stream, _ = document()
    order: list[str] = []
    for cls, name in ((TraceDisplay, "trace"), (ChannelsPage, "channels")):
        original = cls.close

        def _close(self, _original=original, _name=name) -> bool:
            order.append(_name)
            return _original(self)

        monkeypatch.setattr(cls, "close", _close)
    disconnect = stream.disconnect

    def _disconnect() -> None:
        order.append("disconnect")
        disconnect()

    monkeypatch.setattr(stream, "disconnect", _disconnect)
    doc.teardown()
    assert order == ["trace", "channels", "disconnect"]
    assert not stream.connected


def test_teardown_releases_a_half_connected_inlet(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a stream which reads as disconnected still has its inlet destroyed.

    'connect()' can raise *after* it opened the inlet -- the channel-info read and the
    time correction both follow it -- which leaves a live, subscribed inlet on a stream
    whose 'connected' is 'False'. That is the state a failed reconnection attempt leaves
    behind, so a teardown which asked 'connected' first leaked that inlet and its
    acquisition thread for the life of the process. The state is reproduced here by
    nulling one backing attribute with the acquisition stopped, which is what a partial
    initialization looks like from outside without paying a second connection.
    """
    doc, stream, _ = document()
    stream._executor.shutdown(wait=True, cancel_futures=True)
    stream._buffer = None  # the shape of a 'connect()' which raised after 'open_stream'
    assert not stream.connected
    assert stream._inlet is not None
    doc.teardown()
    assert stream._inlet is None


def test_teardown_borrowed(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a borrowed stream survives the document it was shown in.

    This is the 'BaseStream.plot()' contract: the caller owns the stream and closing the
    viewer must leave it connected.
    """
    doc, stream, _ = document(owns_stream=False)
    assert not doc.owns_stream
    doc.teardown()
    assert not doc.trace.running
    assert stream.connected


def test_teardown_breaks_the_model_edge(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a layout change after the teardown reaches the closed display no more.

    The model outlives the document's widgets, so a push arriving afterwards draws into
    a display nobody can see. Asserted over a *borrowed* stream, the case which is not
    benign: an owned one is disconnected by the teardown and the render returns on that,
    while a borrowed one is still connected and the fetch goes through.
    """
    doc, stream, _ = document(owns_stream=False)
    doc.teardown()
    assert stream.connected
    rows, gain = list(doc.trace._rows), list(doc.trace._gain)
    seen: list[object] = []
    doc.changed.connect(seen.append)
    doc.model.set_visible([0], False)
    doc.model.set_type([1], "ecg")
    assert doc.trace._rows == rows
    assert doc.trace._gain == gain
    assert seen == []


def test_set_frozen_is_inert_after_the_teardown(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that resuming a torn-down document does not restart its render clock.

    Asserted over a *borrowed* stream, the case which is not benign: the teardown leaves
    it connected, so a restarted clock fetches real windows and draws them into a closed
    display -- and nothing is left to stop that clock again.
    """
    doc, stream, _ = document(owns_stream=False)
    doc.teardown()
    assert not doc.trace.running
    seen: list[object] = []
    doc.changed.connect(seen.append)
    doc.set_frozen(False)
    assert not doc.trace.running
    doc.set_frozen(True)
    assert not doc.trace.running
    assert seen == []
    assert stream.connected


def test_teardown_via_closed_signal(
    manager: ads.CDockManager,
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that closing the dock widget tears the document down exactly once.

    Qt-ADS delivers no close event to the content widget, thus nothing but this signal
    stops the render clock; and it emits 'closed' once even for two close calls, so the
    count also pins the guard.
    """
    calls: list[object] = []
    original = StreamDocument.teardown

    def _counting(self: StreamDocument) -> None:
        calls.append(self)
        original(self)

    monkeypatch.setattr(StreamDocument, "teardown", _counting)
    doc, stream, _ = document()
    _dock(manager, doc)
    doc.closeDockWidget()
    doc.closeDockWidget()
    assert len(calls) == 1
    assert doc.isClosed()
    assert not doc.trace.running
    assert not stream.connected


def test_teardown_survives_a_failing_disconnect(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a stream whose 'disconnect()' raises does not abort the teardown.

    A window close tears every document down in one loop, thus an escaping exception
    here would abandon every document after this one -- with its clock still ticking and
    its stream still connected.
    """
    doc, stream, _ = document()

    def _raise() -> None:
        raise RuntimeError("boom on disconnect")

    monkeypatch.setattr(stream, "disconnect", _raise)
    caplog.set_level(logging.WARNING, logger="mne_lsl")
    doc.teardown()
    assert not doc.trace.running
    assert "boom on disconnect" in caplog.text
    # undone here, and not left to the fixture ordering: the stream fixture disconnects
    # what is still connected at teardown, and it must not meet the raising stand-in.
    monkeypatch.undo()
    assert stream.connected


# -- theming --------------------------------------------------------------------------
def test_retint_icons(
    app: QApplication,
    controller: ThemeController,
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a theme flip repaints the indicator and rebuilds the toolbar icons.

    A 'QIcon' bakes its color at creation and 'qtawesome' memoizes on the icon name
    alone, thus the rendered pixmap is what has to change; and the indicator carries the
    mode's own status color rather than the one the other mode left behind.
    """
    doc, _, _ = document()
    controller.install(app, "light")
    doc.retint_icons()
    light_style = doc._indicator.styleSheet()
    light_icons = [
        button.icon().pixmap(16, 16).toImage()
        for button in (doc._freeze_button, doc._controller_button, doc._close_button)
    ]
    controller.set_mode("dark")
    doc.retint_icons()
    dark_style = doc._indicator.styleSheet()
    dark_icons = [
        button.icon().pixmap(16, 16).toImage()
        for button in (doc._freeze_button, doc._controller_button, doc._close_button)
    ]
    assert tokens("light").success in light_style
    assert tokens("dark").success in dark_style
    for index, (before, after) in enumerate(zip(light_icons, dark_icons, strict=True)):
        assert not before.isNull()
        assert before != after, index


# -- the serializable state ------------------------------------------------------------
_STATE_KEYS = frozenset(
    {
        "slot",
        "identity",
        "hidden",
        "renames",
        "types",
        "units",
        "bads",
        "controller",
        "display",
    }
)


def _row_of(doc: StreamDocument, name: str) -> int:
    """Return the display row whose channel was acquired under ``name``."""
    model = doc.model
    return next(
        row for row in range(model.rowCount()) if model.channel(row).orig.name == name
    )


def _edit(doc: StreamDocument) -> None:
    """Apply one edit of every kind a configuration carries to ``doc``."""
    doc.model.set_type([_row_of(doc, "zeta")], "ecg")
    doc.model.set_unit([_row_of(doc, "zeta")], "mV")
    doc.model.rename(_row_of(doc, "alpha"), "Left frontal")
    doc.model.set_bad([_row_of(doc, "mike")], True)
    doc.model.set_visible([_row_of(doc, "bravo")], False)
    doc.model.set_order([3, 0, 7, 1, 6, 2, 5, 4])
    doc.set_controller_width(250)
    doc.set_controller_visible(False)
    bar = doc.trace.controls
    bar.set_rows(7)
    bar.set_window(3.0)
    bar.set_scale(2.0)
    bar._color_combo.setCurrentIndex(1)
    bar._labels_switch.setChecked(False)
    bar._events_switch.setChecked(False)


def test_capture_state_of_a_fresh_document(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test the captured key set and that an unedited document carries only defaults.

    'channel_order' has to be absent: writing it unconditionally puts a 256-name list
    which says nothing into every configuration and breaks the omit-what-is-default rule
    the file's inspectability rests on. The four delta containers have to be empty: a
    diff taken against a constant rather than against the acquisition baseline turns
    each of them into a full table instead.

    The display block is read from the control bar and not from the trace display's own
    copies -- the bar is the single source of truth, and a second one is what a whole
    controller page was removed to avoid.
    """
    doc, _, _ = document()
    doc.trace.controls.set_rows(7)
    state = doc.capture_state()
    assert set(state) == _STATE_KEYS
    assert state["slot"] == doc.objectName()
    assert state["identity"] == list(doc.identity.as_tuple())
    assert state["hidden"] == []
    assert state["renames"] == {}
    assert state["types"] == {}
    assert state["units"] == {}
    assert state["bads"] == []
    assert state["controller"] == {
        "visible": True,
        "width": doc._splitter.sizes()[0],
    }
    assert state["display"] == doc.trace.controls.state
    assert state["display"]["rows"] == 7


def test_capture_state_is_keyed_on_acquisition_names(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that every channel key is the device's name and survives a disconnection.

    Keying on the edited name makes the configuration unmatchable -- the availability
    contract lists the device's names -- and unloadable, since the restore looks the
    device's names up. The disconnection half pins that nothing here reads the stream: a
    document whose stream went away must still be savable, because a configuration
    describes a *desired* workspace.
    """
    doc, stream, _ = document(ch_names=list(_SCRAMBLED))
    _edit(doc)
    state = doc.capture_state()
    assert state["renames"] == {"alpha": "Left frontal"}
    assert state["types"] == {"zeta": "ecg"}
    assert state["units"] == {"zeta": -3}
    assert state["bads"] == ["mike"]
    assert state["hidden"] == ["bravo"]
    # the saved order is the acquisition names in presentation order, never the edited
    # ones: 'alpha' is renamed and still appears under its device name.
    assert state["channel_order"] == [
        "bravo",
        "zeta",
        "STI0",
        "alpha",
        "delta",
        "mike",
        "charlie",
        "yankee",
    ]
    stream.disconnect()
    assert doc.capture_state() == state


def test_capture_state_over_a_hidden_controller(
    app: QApplication,
    manager: ads.CDockManager,
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a hidden controller still saves the width it had.

    The document is docked and the host shown on purpose: a laid-out 'QSplitter' reports
    a hidden child as **0** pixels wide, which is measurable only once it has a
    real size. Reading 'sizes()[0]' unconditionally therefore writes a zero-width panel
    into the configuration, and the restored document then shows nothing at all when the
    user toggles the controller back on.
    """
    doc, _, _ = document()
    _dock(manager, doc)
    manager.window().show()
    app.processEvents()
    doc.set_controller_width(250)
    app.processEvents()
    # never asserted as 250: the splitter clamps the panel to its own minimum, so the
    # number saved is whatever the layout granted rather than whatever was asked for.
    width = doc.capture_state()["controller"]
    assert width == {"visible": True, "width": doc._splitter.sizes()[0]}
    assert width["width"] > 0
    doc.set_controller_visible(False)
    app.processEvents()
    assert doc._splitter.sizes()[0] == 0  # what the naive read would have saved
    assert doc.capture_state()["controller"] == {
        "visible": False,
        "width": width["width"],
    }


def test_apply_state_round_trip(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that save, restore and save again produce the same non-empty deltas.

    The strongest single assertion of the configuration feature: it fails if any step of
    'apply_state' is dropped, reordered or keyed wrongly. The delta containers are
    asserted **non-empty** first, because an equality between two empty dictionaries
    passes for exactly the defect this test exists to catch -- a restore which wrote the
    edits onto the stream, re-baselined the acquisition metadata and made the next save
    discard the whole configuration.

    Two streams, not one: a second document over the *same* stream would read the
    already edited metadata as its baseline, i.e. the failure being ruled out here.
    """
    first, _, _ = document(ch_names=list(_SCRAMBLED))
    _edit(first)
    saved = first.capture_state()
    for key in ("renames", "types", "units", "bads", "hidden", "channel_order"):
        assert saved[key], key
    second, _, _ = document(ch_names=list(_SCRAMBLED))
    second.apply_state(saved)
    restored = second.capture_state()
    # the slot and the identity belong to the document, not to the presentation state.
    for state in (saved, restored):
        del state["slot"]
        del state["identity"]
    assert restored == saved


def test_apply_state_keeps_orig_as_the_device_baseline(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a restore leaves the acquisition baseline the device's own values.

    Writing the saved edits onto the stream before the model exists makes the *edited*
    values that baseline: Reset becomes a no-op for every restored edit, and the next
    save writes empty deltas, i.e. destroys the configuration by saving it. Asserted via
    'reset_metadata', which is the user-visible consequence.
    """
    doc, _, _ = document(ch_names=list(_SCRAMBLED))
    row = _row_of(doc, "alpha")
    doc.apply_state({"renames": {"alpha": "Left frontal"}, "types": {"alpha": "ecg"}})
    channel = doc.model.channel(row)
    assert (channel.name, channel.ch_type) == ("Left frontal", "ecg")
    assert (channel.orig.name, channel.orig.ch_type) == ("alpha", "eeg")
    assert doc.model.reset_metadata([row]) == []
    channel = doc.model.channel(row)
    assert (channel.name, channel.ch_type) == ("alpha", "eeg")


def test_apply_state_types_before_units(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a saved type and multiplier on one channel both land.

    A unit-less channel acquires a physical unit through its *type*, so applying the
    units first offers the multiplier to a channel whose kind has no unit ladder at all:
    it is refused, the type write then resets the multiplier to zero, and the saved unit
    is lost with nothing on screen to explain it.
    """
    doc, _, _ = document(
        n_channels=3,
        n_stim=0,
        ch_names=["A", "B", "C"],
        ch_types=["misc", "eeg", "eeg"],
        ch_units=["none", "uv", "uv"],
    )
    row = _row_of(doc, "A")
    assert doc.model.channel(row).orig.unit_kind == int(FIFF.FIFF_UNIT_NONE)
    doc.apply_state({"types": {"A": "eeg"}, "units": {"A": -3}})
    channel = doc.model.channel(row)
    assert channel.ch_type == "eeg"
    assert channel.unit_mul == -3
    assert channel.unit == "mV"


def test_apply_state_is_best_effort_per_key(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a hand-edited state applies what it can and raises for nothing.

    The file is user-editable and a restore runs after every stream of the configuration
    is already connected: a raise here aborts a load with nothing left to fail at. An
    unknown channel name is the legitimate half of the same property -- a channel set
    may have grown, or the stream may have been re-provisioned -- and one bad name must
    not cost the document every other edit.

    The payload itself is checked too: 'presentation["streams"]' is opaque to the
    persistence layer, thus a hand-edited file can put a list where a block belongs and
    the load path reaches this with no check of its own. One of the values is a *list*,
    which is unhashable: grouping the rows by it before checking its type would raise
    'TypeError' from inside the grouping, and this method may not raise at all.
    """
    doc, _, _ = document(ch_names=list(_SCRAMBLED))
    caplog.set_level(logging.WARNING, logger="mne_lsl")
    display = dict(doc.trace.controls.state)
    order = doc.model.presentation_order()
    before = doc.capture_state()
    doc.apply_state(["not", "a", "mapping"])
    assert doc.capture_state() == before
    assert "not a mapping" in caplog.text
    doc.apply_state(
        {
            "types": {"zeta": "ecg", "absent": "eeg", "alpha": 5, "mike": ["eeg"]},
            "units": {"zeta": -3, "alpha": "mV", "bravo": True},
            "renames": {"mike": "Kept", "delta": None},
            "bads": "mike",
            "channel_order": "reversed",
            "hidden": 3,
            "controller": {"width": -50, "visible": "yes"},
            "display": {
                "rows": "many",
                "window": None,
                "labels": 3,
                "color_mode": "no",
            },
        }
    )
    zeta = doc.model.channel(_row_of(doc, "zeta"))
    assert (zeta.ch_type, zeta.unit_mul) == ("ecg", -3)  # the usable values landed
    assert doc.model.channel(_row_of(doc, "mike")).name == "Kept"
    assert doc.model.presentation_order() == order
    assert doc.model.hidden_channels() == []
    assert doc.capture_state()["bads"] == []
    assert doc.controller_visible
    assert doc.trace.controls.state == display
    assert "absent" in caplog.text


# -- disconnection detection and recovery ---------------------------------------------
def _tick(doc: StreamDocument) -> None:
    """Run one full render tick: the display polls, then reports it."""
    doc.trace._render()


def _spin(doc: StreamDocument, n: int = 1) -> None:
    """Report ``n`` ticks without polling the stream again.

    What lets a test advance the state machine while the window the display holds, and
    therefore 'trace.last_timestamp', stays exactly where it was.
    """
    for _ in range(n):
        doc.trace.polled.emit()


def _spy_reconnect(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Replace 'submit_reconnect' with a spy recording ``(stream, expected, slot)``.

    No test of this block touches the network: the reconnection is never performed,
    and the test feeds the outcome by calling the recorded slot itself.

    A real '_ReconnectSignals' is handed back, connected exactly as 'submit_reconnect'
    connects it, because the handle is not opaque to the document: a teardown re-points
    the in-flight attempt at a receiver which does not need the document to survive, so
    a bare sentinel here would test a document nothing can be re-pointed on.
    """
    calls: list[tuple] = []
    emitters: list[_ReconnectSignals] = []

    def _submit(stream: object, expected: object, on_finished: Callable) -> object:
        signals = _ReconnectSignals()
        signals.finished.connect(on_finished)
        emitters.append(signals)  # held, as the pool's runnable holds the real one
        calls.append((stream, expected, on_finished))
        return signals

    monkeypatch.setattr(_document, "submit_reconnect", _submit)
    return calls


def test_fresh_document_is_live(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a document starts live, with no reason and no notice strip built.

    Kills initialising the state to anything else, and kills building the banner
    eagerly: it costs a widget per document for a state most documents never reach.
    """
    doc, _, _ = document()
    assert doc.state == LIVE
    assert doc.notice == ""
    assert doc._banner is None
    assert doc.status_fields()["state"] == "Connected • Live"


def test_lost_stream_interrupts(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
) -> None:
    """Test that a disconnected stream moves the document and says why.

    Kills dropping the 'connected' branch, and kills not emitting 'changed', which
    leaves the shared status bar showing 'Connected • Live' over a stream which is gone.
    The status fields are asserted here too: the state must name the reason, and the
    live fields must fall back rather than be read off a stream which raises.
    """
    doc, stream, _ = document()
    seen: list[StreamDocument] = []
    doc.changed.connect(seen.append)
    stream.disconnect()
    _tick(doc)
    assert doc.state == INTERRUPTED
    assert doc.notice == "Stream disconnected"
    banner = doc._banner
    assert banner is not None
    assert not banner.isHidden()
    assert "Stream disconnected" in banner._label.text()
    assert "reconnecting" in banner._label.text()
    assert banner._retry_button.isHidden()  # the viewer is already retrying
    assert seen == [doc]
    fields = doc.status_fields()
    assert fields["state"] == "Interrupted • Stream disconnected"
    for key in ("channels", "sfreq", "history", "latency"):
        assert fields[key] == "—", key


def test_stall_interrupts_only_once_armed(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the stall detector: unarmed, all-hidden, expired, and not self-resuming.

    Kills arming the freshness clock in the constructor, which would move every document
    over a silent outlet to 'Interrupted' after five seconds -- an order-dependent flake
    across the whole suite. Kills dropping the 'n_rows' term, which is a false notice
    whenever the operator hides every row. Kills sharing one wording with a lost stream.

    And it kills the one-stage resume: a stall leaves the stream *connected* with its
    stale timestamp above zero, so a tick which resumed on 'connected and ts > 0' would
    flash the notice every 'T_STALL' seconds forever and never reconnect anything.
    """
    doc, _, push = document()
    monkeypatch.setattr(_document, "T_STALL", 0.0)
    _spin(doc, 5)  # nothing was ever pushed: the clock is unarmed
    assert doc.state == LIVE
    push(50)
    _tick(doc)  # the first non-empty window arms it
    assert doc.state == LIVE

    doc.model.set_visible(list(range(doc.model.rowCount())), False)
    assert doc.trace.n_rows == 0
    _spin(doc, 5)
    assert doc.state == LIVE
    doc.model.set_visible(list(range(doc.model.rowCount())), True)
    _spin(doc, 2)
    assert doc.state == INTERRUPTED
    assert doc.notice == "No data"
    assert doc.status_fields()["state"] == "Interrupted • No data"
    # still connected, so the live fields keep coming from the stream
    assert doc.status_fields()["channels"] == "8/8 ch"

    refreshes: list[int] = []
    monkeypatch.setattr(doc.model, "refresh", lambda: refreshes.append(1))
    calls = _spy_reconnect(monkeypatch)
    _spin(doc, 30)
    assert doc.state == INTERRUPTED
    assert doc.notice == "No data"
    assert refreshes == []  # no tick may re-apply the settings on its own
    assert calls == []  # and the first retry deadline has not passed yet


def test_attempt_is_submitted_once_and_stops_the_clock(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
) -> None:
    """Test the retry deadline, the single-flight guard, the clock and a user freeze.

    Kills dropping the deadline or the in-flight guard, which would submit a
    reconnection 30 times a second. Kills dropping the '_attempt' term from the clock
    rule: a running clock across an attempt landing on a narrower stream raises
    'IndexError' at 30 Hz. Kills dropping the '_frozen' term, i.e. a reconnection which
    silently unfreezes the viewport the operator froze.
    """
    doc, stream, _ = document()
    monkeypatch.setattr(_document, "_BACKOFF", (0.05, 0.1, 0.2, 0.4))
    calls = _spy_reconnect(monkeypatch)
    stream.disconnect()
    _tick(doc)
    assert doc.state == INTERRUPTED
    _spin(doc, 5)
    assert calls == []  # the deadline is 50 ms away and no time has passed

    qtbot.waitUntil(lambda: len(calls) == 1, timeout=5000)
    assert doc._attempt is not None
    assert not doc.trace.running  # stopped for the whole attempt
    _spin(doc, 5)
    assert len(calls) == 1  # never two at once

    doc.set_frozen(True)
    calls[0][2](RESUME_RETRY, "still gone")
    assert not doc.trace.running  # the freeze survived the outcome
    doc.set_frozen(False)
    assert doc.trace.running


def test_retry_climbs_the_backoff_and_reuses_the_banner(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a failed reconnection climbs the ladder and re-texts one banner.

    Kills resetting the ladder on a failure, which re-resolves the network every second
    for the whole outage. Kills building a second banner per attempt, which stacks a
    notice strip per retry down the document.
    """
    doc, stream, _ = document()
    calls = _spy_reconnect(monkeypatch)
    stream.disconnect()
    _tick(doc)
    banner = doc._banner
    assert doc._backoff == 0
    assert doc._next_attempt - time.monotonic() == pytest.approx(1.0, abs=0.1)

    intervals: list[float] = []
    for _ in range(5):
        doc._submit_attempt()
        before = time.monotonic()
        calls[-1][2](RESUME_RETRY, "still gone")
        intervals.append(round(doc._next_attempt - before, 1))
        assert doc._banner is banner
    assert intervals == [2.0, 5.0, 10.0, 10.0, 10.0]
    assert doc.state == INTERRUPTED


def test_mismatch_is_terminal_and_retry_leaves_it(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the refused-resume state, its affordance, and the verb which leaves it.

    Kills leaving the retry deadline armed in a terminal state, i.e. re-resolving the
    network forever for a stream which will never be accepted. Kills routing a refusal
    through the resume, which would re-baseline the model against the wrong stream.
    Kills not wiring the banner's Retry, and kills not resetting the ladder when the
    operator asks for the attempt.

    And it kills dropping the state from the clock rule: neither other stop reason holds
    here, so the 30 Hz clock would run for the life of the document over a stream which
    was released -- and the advertised "frozen on the last frame" would be an accident
    of the fetch's disconnected early return rather than a decision.
    """
    doc, stream, _ = document()
    calls = _spy_reconnect(monkeypatch)
    refreshes: list[int] = []
    monkeypatch.setattr(doc.model, "refresh", lambda: refreshes.append(1))
    stream.disconnect()
    _tick(doc)
    doc._submit_attempt()
    doc._backoff = 2  # a ladder already climbed, so the reset below is observable
    reason = "the channel count changed from 8 to 3"
    calls[-1][2](RESUME_MISMATCH, reason)
    assert doc.state == MISMATCHED
    assert doc.notice == reason
    assert doc.status_fields()["state"] == f"Interrupted • {reason}"
    banner = doc._banner
    assert not banner.isHidden()
    assert reason in banner._label.text()
    assert "reconnecting" not in banner._label.text()
    assert not banner._retry_button.isHidden()
    assert doc._next_attempt is None
    assert refreshes == []
    # the clock is stopped by the state, with neither other reason holding
    assert not doc.trace.running
    assert not doc.frozen
    assert doc._attempt is None
    submitted = len(calls)
    _spin(doc, 30)
    assert len(calls) == submitted  # terminal: nothing retries on its own

    banner._retry_button.click()
    assert doc.state == INTERRUPTED
    assert doc._backoff == 0
    assert len(calls) == submitted + 1
    assert doc._banner is banner


def test_resume_reapplies_then_waits_for_data(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the two-stage resume, its order, and that the user's edits survive it.

    The stream is genuinely reconnected here, on the still-live outlet, because the
    re-apply reads it. Kills reordering 'refresh' and 'apply_state', which makes the
    re-apply a no-op against the model's own cache. Kills declaring the document live on
    the outcome instead of on the first non-empty window, which flashes a live viewport
    for a source which returns and vanishes again.

    And it kills any path which lets 'refresh()' rebuild: a rebuild re-baselines
    'Channel.orig', which makes the next 'capture_state' produce empty deltas, i.e.
    destroys the configuration by the act of saving it.

    The signature the attempt carries is asserted here rather than in a test of its own,
    because this is the one test whose document has been renamed: the expected channel
    names are the **acquisition** names, so a signature built from the model's current
    names would refuse every renamed document its own stream. And 'changed' is asserted
    on the resume, without which the status bar keeps reading 'Interrupted' over a live
    viewport.
    """
    doc, stream, push = document(ch_names=list(_SCRAMBLED))
    _edit(doc)
    before = doc.capture_state()
    for key in ("channel_order", "hidden", "renames", "types", "units", "bads"):
        assert before[key], key

    order: list[str] = []
    original = (doc.capture_state, doc.model.refresh, doc.apply_state)

    def _capture() -> dict:
        order.append("capture")
        return original[0]()

    def _refresh() -> None:
        order.append("refresh")
        original[1]()

    def _apply(state: object) -> None:
        order.append("apply")
        original[2](state)

    monkeypatch.setattr(doc, "capture_state", _capture)
    monkeypatch.setattr(doc.model, "refresh", _refresh)
    monkeypatch.setattr(doc, "apply_state", _apply)

    calls = _spy_reconnect(monkeypatch)
    stream.disconnect()
    _tick(doc)
    assert doc.state == INTERRUPTED
    doc._submit_attempt()
    expected = calls[-1][1]
    assert expected.identity == doc.identity
    assert expected.ch_names == tuple(_SCRAMBLED)  # the wire names, not the edited ones
    assert expected.sfreq == 100.0
    assert expected.dtype == "float32"
    reconnect_stream(stream)  # what the worker does before it reports a match
    calls[-1][2](RESUME_LIVE, "")
    assert order == ["capture", "refresh", "apply"]
    assert doc.state == INTERRUPTED  # not live until one window arrives
    assert not doc._banner.isHidden()
    assert doc.trace.running

    seen: list[StreamDocument] = []
    doc.changed.connect(seen.append)
    push(50)
    _tick(doc)
    assert doc.state == LIVE
    assert doc.notice == ""
    assert doc._banner.isHidden()
    assert seen == [doc]  # the status bar is told the document is live again
    monkeypatch.setattr(doc, "capture_state", original[0])
    assert doc.capture_state() == before


def test_flap_rewords_and_keeps_the_backoff(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a loss shortly after a resume is re-worded and keeps the ladder.

    A source which comes and goes must not restart the ladder at one second per attempt
    every time, and the operator has to be told the connection is unstable rather than
    shown the same notice as a first outage. Kills resetting the ladder on every resume.
    """
    doc, _, push = document()
    monkeypatch.setattr(_document, "T_STALL", 0.0)
    calls = _spy_reconnect(monkeypatch)
    push(50)
    _tick(doc)
    _spin(doc, 2)
    assert doc.state == INTERRUPTED
    assert doc.notice == "No data"
    banner = doc._banner

    doc._submit_attempt()
    calls[-1][2](RESUME_RETRY, "still gone")
    assert doc._backoff == 1
    # the source recovered: the stream never left, so a match is what the worker reports
    doc._submit_attempt()
    calls[-1][2](RESUME_LIVE, "")
    push(50)
    _tick(doc)
    assert doc.state == LIVE
    assert doc._resumed_at is not None

    _spin(doc, 2)  # 'T_STALL' is 0, so the very next tick loses it again
    assert doc.state == INTERRUPTED
    assert doc.notice == "Connection unstable"
    assert doc._backoff == 1  # the ladder kept climbing
    assert doc._banner is banner


def test_teardown_closes_and_releases_a_late_resume(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a torn document stops listening and releases a stream which came back.

    Kills dropping the torn guard of the outcome handler: the worker transfers a
    connected stream, so a document which went away while an attempt was in flight leaks
    a live inlet plus its acquisition thread for the life of the process.
    """
    doc, stream, _ = document()
    calls = _spy_reconnect(monkeypatch)
    stream.disconnect()
    _tick(doc)
    doc._submit_attempt()
    on_finished = calls[-1][2]

    doc.teardown()
    assert doc.state == CLOSED
    with pytest.raises((TypeError, RuntimeError)):
        doc.trace.polled.disconnect(doc._on_polled)  # already dropped

    reconnect_stream(stream)  # the worker got the stream back
    assert stream.connected
    on_finished(RESUME_LIVE, "")
    assert not stream.connected


def test_awaiting_data_has_a_deadline_and_a_hidden_layout_confirms(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the two ways the wait for the first post-resume window used to never end.

    Both are a source which answered the identity and then delivered nothing. Kills
    dropping the confirmation deadline: a hung sender is what 'recover=False' leaves
    liblsl unable to report, so without a deadline of its own the document waits forever
    for a window which never comes, repainting an all-NaN viewport at 30 Hz, and the
    attempt count stays at one for the rest of the session.

    And it kills dropping the 'n_rows' term: an all-hidden display fetches nothing at
    all, so 'last_timestamp' is frozen at whatever it held before the outage -- here the
    0.0 of a stream which never delivered a sample, which no positive test can ever
    satisfy. There is no live viewport to flash either, which is what the same term
    exists for eleven lines above in the stall branch.

    One stream and one reconnection for both halves: the second stage needs a stream the
    worker really did reconnect, because the resume re-applies the document state and
    that reads the stream.
    """
    doc, stream, _ = document()
    monkeypatch.setattr(_document, "T_STALL", 0.0)
    monkeypatch.setattr(_document, "_BACKOFF", (0.0, 0.0, 0.0, 0.0))
    calls = _spy_reconnect(monkeypatch)
    stream.disconnect()
    _tick(doc)
    assert doc.state == INTERRUPTED
    reconnect_stream(stream)  # what the worker does before it reports a match
    doc._submit_attempt()
    calls[-1][2](RESUME_LIVE, "")
    assert doc._awaiting_data
    assert doc._next_attempt is not None  # the confirmation carries its own deadline

    _tick(doc)  # the source answered and then said nothing
    assert doc.state == INTERRUPTED
    assert doc.notice == "No data"
    assert not doc._awaiting_data
    assert doc._backoff == 1
    submitted = len(calls)
    _spin(doc, 1)
    assert len(calls) == submitted + 1  # a second attempt, not a wedge

    doc.model.set_visible(list(range(doc.model.rowCount())), False)
    assert doc.trace.n_rows == 0
    assert not doc.trace.last_timestamp  # 0.0: nothing was ever acquired
    calls[-1][2](RESUME_LIVE, "")
    _tick(doc)
    assert doc.state == LIVE
    assert doc.notice == ""
    _spin(doc, 5)
    assert doc.state == LIVE  # and the all-hidden layout does not stall it either


def test_lost_again_while_awaiting_data_retries_and_says_so(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the one failure path which reached neither the banner nor the status bar.

    A stream lost again before the first post-resume window is the only failure which
    did not go through '_enter': the ladder was climbed and a deadline armed in place,
    so the notice strip kept the text of the attempt before it and 'changed' was never
    emitted -- the shared status bar therefore never learned that the resume had failed.
    Kills climbing the ladder anywhere but through the one helper.
    """
    doc, stream, push = document()
    monkeypatch.setattr(_document, "T_STALL", 0.0)
    calls = _spy_reconnect(monkeypatch)
    push(50)
    _tick(doc)  # the first non-empty window arms the freshness clock
    _spin(doc, 2)
    assert doc.state == INTERRUPTED
    assert doc.notice == "No data"
    doc._submit_attempt()
    calls[-1][2](RESUME_LIVE, "")  # the stream never left: the worker reports a match
    assert doc._awaiting_data
    banner = doc._banner
    # a sentinel, so that the re-texting below is a change and not a coincidence.
    banner.set_notice("sentinel")
    seen: list[StreamDocument] = []
    doc.changed.connect(seen.append)

    stream.disconnect()
    _tick(doc)
    assert not doc._awaiting_data
    assert doc.state == INTERRUPTED
    assert doc._backoff == 1  # the ladder climbed
    assert doc._next_attempt is not None  # and a deadline is armed
    assert seen == [doc]  # the status bar is told
    assert "No data" in banner._label.text()
    assert "reconnecting" in banner._label.text()


def test_a_failing_reapply_retries_instead_of_killing_the_document(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a re-apply which raises leaves a document which still retries.

    The re-apply reads the stream, and a source lost again in the ~1 ms between the
    worker's comparison and this slot makes it raise -- a flapping source, which is what
    the ladder and the stability window exist for. The raise used to escape past
    '_apply_clock', the handler's last line, leaving the clock stopped, no attempt in
    flight and no deadline armed: a document dead for the rest of the session, with a
    banner still promising a reconnection.
    """
    doc, stream, _ = document()
    calls = _spy_reconnect(monkeypatch)
    stream.disconnect()
    _tick(doc)
    doc._submit_attempt()

    def _raise() -> None:
        raise RuntimeError("the stream is not connected")

    monkeypatch.setattr(doc, "_reapply", _raise)
    caplog.set_level(logging.WARNING, logger="mne_lsl")
    calls[-1][2](RESUME_LIVE, "")
    assert doc.state == INTERRUPTED
    assert not doc._awaiting_data
    assert doc._backoff == 1
    assert doc._next_attempt is not None
    assert doc._attempt is None
    assert doc.trace.running  # the clock came back
    assert not doc.trace._suspended
    assert "Could not re-apply the settings" in caplog.text


def test_an_interaction_never_touches_a_suspended_or_frozen_stream(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an interaction reaches the display and never reports a render tick.

    A stopped clock is not what keeps this document off its stream: a layout change, a
    scroll and a row step all repaint by themselves, and a repaint which finds nothing
    it can reuse re-reads the window. During an attempt that fetch runs against a stream
    the worker has just reconnected -- possibly narrower -- which raises 'IndexError'
    from inside whichever interaction triggered it. And it used to *emit* the render
    tick as well, so one visibility click advanced the connection state machine.

    Every layout change here **widens** the layout, which is the load-bearing detail: a
    narrower one is a subset of the retained window, so the repaint reindexes it and
    reaches no fetch at all. A test built on hiding a channel passes with both
    invariants deleted -- it is the retained frame doing the work, the same trap as
    asserting 'not trace.running' instead of the invariant.
    """
    doc, stream, push = document()
    monkeypatch.setattr(_document, "_BACKOFF", (0.0, 0.0, 0.0, 0.0))
    calls = _spy_reconnect(monkeypatch)
    doc.model.set_visible([0], False)  # so the retained window will not carry that row
    push(50)
    _tick(doc)
    assert doc.state == LIVE
    fetches: list[int] = []
    ticks: list[int] = []
    original = stream.get_data

    def _record(winsize=None, picks=None, exclude="bads"):
        fetches.append(1)
        return original(winsize, picks, exclude)

    monkeypatch.setattr(stream, "get_data", _record)
    doc.trace.polled.connect(lambda: ticks.append(1))

    # (a) live and running: the interaction may fetch, and may not report a tick
    doc.model.set_visible([0], True)
    assert fetches  # the new row's samples have to come from somewhere
    assert ticks == []

    # (b) during an attempt the stream is off limits, repaint or no repaint
    doc.model.set_visible([0], False)
    doc._submit_attempt()
    assert doc.trace._suspended
    fetches.clear()
    doc.model.set_visible([0], True)  # a pick the retained window does not carry
    doc.trace.scroll_by(1)
    doc.trace.controls.set_rows(4)
    assert fetches == []
    assert ticks == []
    calls[-1][2](RESUME_RETRY, "still gone")
    assert not doc.trace._suspended
    assert doc.state == INTERRUPTED
    submitted = len(calls)

    # (c) frozen: the new row still has to come from somewhere, so a fetch is legitimate
    # -- but the machine must not advance, though the retry deadline has passed
    doc.set_frozen(True)
    doc.model.set_visible([0], False)
    doc.model.set_visible([0], True)
    assert ticks == []
    assert len(calls) == submitted
    doc.set_frozen(False)
    _spin(doc, 1)
    assert len(calls) == submitted + 1  # and the clock still drives one


def test_a_borrowed_stream_is_reconnected_only_on_request(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the viewer never reconnects, nor releases, a stream it does not own.

    A reconnection replaces the object's inlet and its buffer, which drops the filters,
    the callbacks, the acquisition delay, the reference channels and the processing
    flags its owner set on it -- and the stall branch fires on a source which never went
    away at all, merely one which went quiet. So a borrowed document offers Retry rather
    than arming a deadline, and a resume which lands after the document is gone leaves
    the caller's stream connected.
    """
    doc, stream, _ = document(owns_stream=False)
    monkeypatch.setattr(_document, "_BACKOFF", (0.0, 0.0, 0.0, 0.0))
    calls = _spy_reconnect(monkeypatch)
    released: list[object] = []
    monkeypatch.setattr(_document, "release_stream", released.append)
    stream.disconnect()
    _tick(doc)
    assert doc.state == INTERRUPTED
    assert doc._next_attempt is None  # nothing is retried on a timer
    banner = doc._banner
    assert not banner._retry_button.isHidden()  # the operator's verb instead
    assert "reconnecting" not in banner._label.text()
    _spin(doc, 30)
    assert calls == []

    banner._retry_button.click()
    assert len(calls) == 1
    doc.retry()
    assert len(calls) == 1  # single-flight: an attempt is already in flight
    doc.teardown()
    calls[-1][2](RESUME_LIVE, "")
    assert released == []


def test_retry_while_live_is_a_no_op(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the recovery verb does nothing at all to a healthy document.

    Reachable programmatically, and the notice strip's own button is only ever shown by
    an interrupted state. Kills dropping the state guard, which would let a caller
    disconnect and re-resolve a stream which is delivering.
    """
    doc, _, _ = document()
    calls = _spy_reconnect(monkeypatch)
    doc.retry()
    assert calls == []
    assert doc.state == LIVE
    assert doc._banner is None
    assert doc.trace.running


def test_the_stability_window_closes_and_resets_the_ladder(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a resume which held resets the ladder, and that a fresh one does not.

    'T_STABLE' is the second read of the resume timestamp, and without it the timestamp
    is a latch rather than a window: the ladder never resets, so a source which flapped
    once hours ago is still retried at ten-second intervals, and every later notice
    keeps the 'Connection unstable' wording instead of the reason to act on.
    """
    doc, _, push = document()
    monkeypatch.setattr(_document, "T_STALL", 0.0)
    calls = _spy_reconnect(monkeypatch)
    push(50)
    _tick(doc)
    _spin(doc, 2)
    assert doc.state == INTERRUPTED
    doc._submit_attempt()
    calls[-1][2](RESUME_RETRY, "still gone")
    assert doc._backoff == 1
    doc._submit_attempt()
    calls[-1][2](RESUME_LIVE, "")
    push(50)
    _tick(doc)
    assert doc.state == LIVE
    assert doc._resumed_at is not None

    monkeypatch.setattr(_document, "T_STALL", 100.0)  # no stall for the rest of this
    _spin(doc, 2)
    assert doc._backoff == 1  # inside the window: the resume has not held yet
    assert doc._resumed_at is not None
    monkeypatch.setattr(_document, "T_STABLE", 0.0)
    _spin(doc, 1)
    assert doc._backoff == 0  # the resume held
    assert doc._resumed_at is None


def test_a_mismatch_inside_the_stability_window_keeps_its_reason(
    document: Callable[..., tuple[StreamDocument, StreamLSL, Callable[..., None]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the anti-flap wording never overwrites a refusal.

    The re-wording belongs to 'INTERRUPTED' alone. A refusal names the field which
    changed, and that is the only thing on screen telling the operator that the stream
    which came back is not theirs -- 'Connection unstable' over it would describe a
    source which is in fact answering perfectly.
    """
    doc, stream, _ = document()
    calls = _spy_reconnect(monkeypatch)
    stream.disconnect()
    _tick(doc)
    doc._resumed_at = time.monotonic()  # a resume which landed a moment ago
    doc._submit_attempt()
    reason = "the sampling rate changed from 100 Hz to 200 Hz"
    calls[-1][2](RESUME_MISMATCH, reason)
    assert doc.state == MISMATCHED
    assert doc.notice == reason
    assert reason in doc._banner._label.text()


# -- the same, end to end over a real source which goes away --------------------------
# The only tests of this module which start a subprocess. Every assertion is on document
# state and never on an escaping exception: the acquisition thread re-raises into a
# discarded 'Future', so nothing reaches the caller's thread.
_MIXED_PICKS = ["Fpz", "Fp2", "ECG", "TRIGGER"]
# A subset which keeps the channel *type* set mixed. Load-bearing: 'PlayerLSL' publishes
# the single channel type of its file, or '' for a mixed one, so an EEG-only pick
# changes the published 'stype' and therefore the identity -- and the mismatch test
# below would then silently exercise the retry path instead of the match rule.


def _has_data(doc: StreamDocument) -> bool:
    """Return whether the display has fetched a window carrying a real sample."""
    return bool(doc.trace.last_timestamp)


def _open(
    manager: ads.CDockManager, handle: object, index: int = 0
) -> tuple[StreamDocument, StreamLSL]:
    """Connect to a running player and open a document over it, as the window does."""
    from mne_lsl.viewer.backend import connect_stream, resolve_descriptors

    identity = (handle.name, "", handle.source_id)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        for descriptor in resolve_descriptors(1.0):
            if descriptor.identity.as_tuple() == identity:
                stream = connect_stream(descriptor, 4.0)
                doc = StreamDocument(manager, stream, descriptor.identity)
                _dock(manager, doc, index)
                return doc, stream
    pytest.fail(f"The player {identity} never became resolvable.")


@pytest.mark.slow
def test_lost_source_is_detected_per_document(
    manager: ads.CDockManager,
    player: Callable[..., object],
    qtbot: QtBot,
    flush_deletes: Callable[..., None],
) -> None:
    """Test that one lost source interrupts its own document and no other.

    Kills any coupling of two documents' connection state: the detection is per document
    by construction, and a shared flag would freeze every open stream on one outage.
    """
    first, second = player(), player()
    doc_a, stream_a = _open(manager, first, 0)
    doc_b, stream_b = _open(manager, second, 1)
    try:
        qtbot.waitUntil(lambda: _has_data(doc_b), timeout=20000)
        first.kill()
        qtbot.waitUntil(lambda: doc_a.state == INTERRUPTED, timeout=20000)
        assert doc_a.notice == "Stream lost"
        assert doc_a._banner is not None
        assert not doc_a._banner.isHidden()
        assert doc_b.state == LIVE
        assert doc_b._banner is None
        advanced = doc_b.trace.last_timestamp
        qtbot.waitUntil(
            lambda: bool(doc_b.trace.last_timestamp > advanced), timeout=20000
        )
        assert doc_b.state == LIVE
    finally:
        doc_a.teardown()
        doc_b.teardown()
        flush_deletes(doc_a, doc_b)


@pytest.mark.slow
def test_source_which_returns_resumes_with_a_gap(
    manager: ads.CDockManager,
    player: Callable[..., object],
    qtbot: QtBot,
    flush_deletes: Callable[..., None],
) -> None:
    """Test that a source which comes back unchanged resumes, leaving a gap on screen.

    The NaN assertion is what pins the gap requirement end to end: the reconnection
    allocates a fresh buffer, and the un-filled part of it must be drawn as a break in
    the curve rather than joined to the first real sample. Kills the whole resume path,
    and kills the un-filled window rule.
    """
    handle = player()
    doc, stream = _open(manager, handle)
    try:
        qtbot.waitUntil(lambda: _has_data(doc), timeout=20000)
        handle.kill()
        qtbot.waitUntil(lambda: doc.state == INTERRUPTED, timeout=20000)
        handle.start()
        qtbot.waitUntil(lambda: doc.state == LIVE, timeout=60000)
        assert doc.notice == ""
        assert doc._banner.isHidden()
        assert stream.connected
        assert np.isnan(doc.trace._frame[1]).any()  # the fetched window
        row = min(doc.trace._assigned)
        assert np.isnan(doc.trace._assigned[row].getData()[1]).any()  # and on screen
    finally:
        doc.teardown()
        flush_deletes(doc)


@pytest.mark.slow
def test_source_which_returns_narrower_is_refused(
    manager: ads.CDockManager,
    player: Callable[..., object],
    qtbot: QtBot,
    flush_deletes: Callable[..., None],
) -> None:
    """Test that a source which came back with other channels is refused, not adopted.

    Every piece of stream-side state the viewer holds is an integer index, so adopting a
    narrower stream would draw one channel's samples under another's label. Kills the
    match rule end to end, and kills forgetting the release on the worker: the stream
    was connected to be compared and must not stay open.
    """
    handle = player()
    doc, stream = _open(manager, handle)
    try:
        qtbot.waitUntil(lambda: _has_data(doc), timeout=20000)
        frozen = doc.trace._frame[1].copy()
        handle.kill()
        qtbot.waitUntil(lambda: doc.state == INTERRUPTED, timeout=20000)
        handle.start(picks=list(_MIXED_PICKS))
        qtbot.waitUntil(lambda: doc.state == MISMATCHED, timeout=60000)
        assert "channel count changed from 67 to 4" in doc.notice
        assert not doc._banner._retry_button.isHidden()
        assert not stream.connected  # released by the worker
        assert np.array_equal(doc.trace._frame[1], frozen, equal_nan=True)
    finally:
        doc.teardown()
        flush_deletes(doc)


@pytest.mark.slow
def test_flapping_source_reuses_one_notice(
    manager: ads.CDockManager,
    player: Callable[..., object],
    qtbot: QtBot,
    flush_deletes: Callable[..., None],
) -> None:
    """Test three real outages in a row: one notice strip, and the unstable wording.

    Kills building a notice per outage, which stacks one strip per retry down the
    document. Kills not recording the resume timestamp, which is what tells a first
    outage from a connection which cannot hold. The retry interval is *not* asserted
    here: whether the first attempt of a cycle lands before or after the restart is a
    race by construction, and the ladder itself is pinned without a subprocess above.
    """
    handle = player()
    doc, stream = _open(manager, handle)
    banners: set[int] = set()
    notices: list[str] = []
    doc.changed.connect(lambda d: notices.append(d.notice))
    try:
        for _ in range(3):
            qtbot.waitUntil(lambda: doc.state == LIVE, timeout=60000)
            qtbot.waitUntil(lambda: _has_data(doc), timeout=20000)
            handle.kill()
            qtbot.waitUntil(lambda: doc.state == INTERRUPTED, timeout=20000)
            banners.add(id(doc._banner))
            handle.start()
        qtbot.waitUntil(lambda: doc.state == LIVE, timeout=60000)
        assert len(banners) == 1  # one strip, ever
        assert "Connection unstable" in notices
    finally:
        doc.teardown()
        flush_deletes(doc)
