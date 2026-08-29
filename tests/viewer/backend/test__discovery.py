from __future__ import annotations

import logging
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from mne_lsl.viewer.backend import (
    RESUME_LIVE,
    RESUME_MISMATCH,
    RESUME_RETRY,
    Connector,
    Discovery,
    Prober,
    StreamIdentity,
    StreamSignature,
    _discovery,
    release_stream,
)
from mne_lsl.viewer.backend._discovery import (
    _PROBE_WORKERS,
    _ConnectorWorker,
    _DiscoveryWorker,
    _ProbeWorker,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType

    from pytestqt.qtbot import QtBot
    from qtpy.QtWidgets import QApplication

    from mne_lsl.stream import BaseStream
    from mne_lsl.viewer.backend import StreamDescriptor


class _DummyStream:
    """Stand-in for a connected stream, recording its own release."""

    def __init__(self) -> None:
        self.disconnected = 0

    def disconnect(self) -> None:
        """Record one release."""
        self.disconnected += 1


@pytest.fixture
def discovery(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Generator[Discovery, None, None]:
    """Yield a discovery object, stopped at teardown.

    Both parameters are requested and unused. 'app' because an application must exist
    for a QThread to run. 'monkeypatch' because fixtures are finalized in reverse order
    of instantiation, thus depending on it here guarantees that this teardown -- the one
    which stops the worker thread -- runs *before* a test's monkeypatched resolver is
    put back, rather than after, when the worker could still be calling it.
    """
    obj = Discovery()
    yield obj
    obj.stop()


@pytest.fixture
def connector(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Generator[Connector, None, None]:
    """Yield a connector object, stopped at teardown; see 'discovery' for the args."""
    obj = Connector()
    yield obj
    obj.stop()


@pytest.fixture
def prober(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Generator[Prober, None, None]:
    """Yield a prober object, stopped at teardown; see 'discovery' for the args."""
    obj = Prober()
    yield obj
    obj.stop()


# -- import rules ----------------------------------------------------------------------
def test_no_lsl_import(
    module_scan: Callable[[ModuleType], tuple[set[str], set[str]]],
) -> None:
    """Test that '_discovery.py' imports nothing from LSL and never names 'StreamLSL'.

    The transport is Qt only: the two blocking calls it drives live in '_source.py',
    which is what keeps the seam of a future non-LSL protocol to a single module.

    Both sets are asserted, because either one alone has a blind spot: 'from ... import
    lsl' binds the module without putting 'lsl' anywhere in an import *path*, while
    'import mne_lsl' followed by 'mne_lsl.lsl.resolve_streams(...)' reaches the module
    through an attribute and imports nothing forbidden at all.
    """
    imports, identifiers = module_scan(_discovery)
    for name in imports:
        assert "lsl" not in name.split("."), name
    assert not {"lsl", "StreamLSL"} & identifiers


# -- workers, no thread and no network ----------------------------------------------
def test_discovery_worker_done(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a successful pass echoes its generation and its descriptors.

    The worker is a plain QObject, thus 'run' is called directly on the main thread: the
    logic is tested with no thread and no network involved at all.
    """
    descriptors = [descriptor()]
    monkeypatch.setattr(_discovery, "resolve_descriptors", lambda *args: descriptors)
    events: list[tuple] = []
    worker = _DiscoveryWorker()
    worker.generation = 7  # what 'Discovery.refresh' mirrors onto the worker
    worker.done.connect(lambda *args: events.append(("done", *args)))
    worker.failed.connect(lambda *args: events.append(("failed", *args)))
    worker.run(7)
    assert events == [("done", 7, descriptors)]


def test_discovery_worker_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a raising resolution is reported instead of escaping the thread."""

    def _raise(*args: object) -> None:
        raise RuntimeError("the network is on fire")

    monkeypatch.setattr(_discovery, "resolve_descriptors", _raise)
    events: list[tuple] = []
    worker = _DiscoveryWorker()
    worker.generation = 3
    worker.done.connect(lambda *args: events.append(("done", *args)))
    worker.failed.connect(lambda *args: events.append(("failed", *args)))
    worker.run(3)
    assert events == [("failed", 3, "the network is on fire")]


def test_discovery_worker_skips_a_superseded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a request which was superseded before it ran resolves nothing.

    'QThread.quit()' leaves the requests already posted to the worker thread in its
    queue, and a restarted thread drains them from the front, so a request which is no
    longer current has to cancel itself or every 'refresh()' a 'stop()' cancelled is
    replayed later, one full resolution each.
    """
    calls: list[object] = []
    monkeypatch.setattr(
        _discovery, "resolve_descriptors", lambda *args: calls.append(1)
    )
    events: list[tuple] = []
    worker = _DiscoveryWorker()
    worker.generation = 4  # the request below was superseded 3 generations ago
    worker.done.connect(lambda *args: events.append(args))
    worker.failed.connect(lambda *args: events.append(args))
    worker.run(1)
    assert calls == []
    assert events == []


def test_connector_worker_reports_every_outcome(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that one failure never stops the others and that the order is preserved."""
    first, second, third = descriptor("a"), descriptor("b"), descriptor("c")
    streams = {first: _DummyStream(), third: _DummyStream()}

    def _connect(item: StreamDescriptor, bufsize: float) -> _DummyStream:
        if item is second:
            raise RuntimeError("0 were found")
        return streams[item]

    monkeypatch.setattr(_discovery, "connect_stream", _connect)
    events: list[tuple] = []
    worker = _ConnectorWorker()
    worker.generation = 1  # what 'Connector.open' mirrors onto the worker
    worker.connected.connect(lambda *args: events.append(("connected", *args)))
    worker.failed.connect(lambda *args: events.append(("failed", *args)))
    worker.run(1, (first, second, third), 4.0)
    assert events == [
        ("connected", 1, first, streams[first]),
        ("failed", 1, second, "0 were found"),
        ("connected", 1, third, streams[third]),
    ]


def test_connector_worker_skips_a_cancelled_item(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a batch cancelled before an item never starts that connection."""
    calls: list[StreamDescriptor] = []

    def _connect(item: StreamDescriptor, bufsize: float) -> _DummyStream:
        calls.append(item)
        return _DummyStream()

    monkeypatch.setattr(_discovery, "connect_stream", _connect)
    worker = _ConnectorWorker()
    worker.generation = 2  # the batch below was already replaced
    worker.run(1, (descriptor(),), 4.0)
    assert calls == []


def test_connector_worker_releases_a_cancelled_stream(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a stream connected during a cancellation is disconnected.

    A connection takes about a second and cannot be interrupted, so it can complete
    after the batch was replaced. Dropping the result would leak a live inlet and its
    acquisition thread for the life of the process -- the one silent failure mode here.
    """
    dummy = _DummyStream()
    worker = _ConnectorWorker()
    worker.generation = 1

    def _connect(item: StreamDescriptor, bufsize: float) -> _DummyStream:
        worker.generation = 2  # cancelled *during* this connection
        return dummy

    monkeypatch.setattr(_discovery, "connect_stream", _connect)
    events: list[tuple] = []
    worker.connected.connect(lambda *args: events.append(args))
    worker.run(1, (descriptor(), descriptor()), 4.0)
    assert events == []
    assert dummy.disconnected == 1


def test_release_swallows_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a stream which refuses to disconnect is logged, not raised."""

    class _Stubborn:
        def disconnect(self) -> None:
            raise RuntimeError("not connected")

    caplog.set_level(logging.WARNING, logger="mne_lsl")
    release_stream(_Stubborn())  # the failure is swallowed, not propagated
    assert "not connected" in caplog.text


# -- Discovery -------------------------------------------------------------------------
def test_discovery_empty(
    discovery: Discovery, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that an empty network clears the table and says so."""
    monkeypatch.setattr(_discovery, "resolve_descriptors", lambda *args: [])
    tags: list[str] = []
    found: list[object] = []
    discovery.progress.connect(tags.append)
    discovery.streams_found.connect(found.append)
    with qtbot.waitSignal(discovery.streams_found, timeout=10000):
        discovery.refresh()
    assert tags == ["checking", "empty"]
    assert found == [[]]


def test_discovery_failed(
    discovery: Discovery,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a failed pass leaves the last known good list on screen.

    'streams_found' is deliberately not emitted: flickering a good state away because
    one resolution failed is worse than showing a slightly stale one.
    """

    def _raise(*args: object) -> None:
        raise RuntimeError("resolution exploded")

    monkeypatch.setattr(_discovery, "resolve_descriptors", _raise)
    caplog.set_level(logging.WARNING, logger="mne_lsl")
    tags: list[str] = []
    found: list[object] = []
    discovery.progress.connect(tags.append)
    discovery.streams_found.connect(found.append)
    # 'refresh' emits 'progress' synchronously, thus 'waitSignal' on it would return at
    # once on the 'checking' tag; the second tag is the one under test.
    discovery.refresh()
    qtbot.waitUntil(lambda: len(tags) == 2, timeout=10000)
    assert tags == ["checking", "failed"]
    assert found == []
    assert "resolution exploded" in caplog.text


def test_discovery_drops_a_stale_pass(
    discovery: Discovery, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that only the result of the newest pass is published.

    Cancellation is stale-work rejection, not interruption: the first pass runs to
    completion on the worker thread and its result is discarded. The second 'refresh' is
    issued only once the worker is provably inside the first resolution, which is what
    makes the first pass an *in-flight* one rather than a request the worker could have
    cancelled before starting it.
    """
    entered = threading.Event()
    passes: list[list[str]] = []

    def _resolve(*args: object) -> list[str]:
        entered.set()
        time.sleep(0.2)
        result = [f"pass-{len(passes)}"]
        passes.append(result)
        return result

    monkeypatch.setattr(_discovery, "resolve_descriptors", _resolve)
    found: list[object] = []
    discovery.streams_found.connect(found.append)
    discovery.refresh()
    assert entered.wait(10), "the worker never started the first pass"
    discovery.refresh()
    qtbot.waitUntil(lambda: len(passes) == 2, timeout=10000)
    qtbot.wait(200)  # leave both results time to be delivered
    assert found == [["pass-1"]]


def test_discovery_stop_is_idempotent_and_restartable(
    discovery: Discovery, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that stopping twice is a no-op and that a later refresh restarts the worker.

    A finished QThread restarts cleanly and its worker keeps its affinity to it, which
    is what makes one persistent thread per object viable instead of one per pass.
    """
    monkeypatch.setattr(_discovery, "resolve_descriptors", lambda *args: [])
    with qtbot.waitSignal(discovery.streams_found, timeout=10000):
        discovery.refresh()
    discovery.stop()
    assert not discovery._thread.isRunning()
    discovery.stop()
    assert not discovery._thread.isRunning()
    with qtbot.waitSignal(discovery.streams_found, timeout=10000):
        discovery.refresh()


def test_discovery_stop_drops_a_pending_result(
    discovery: Discovery, app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a result which lands after 'stop' is never published.

    The wait on 'entered' is what gives the test its teeth: the pass has to be in flight
    before 'stop' bumps the generation, so that the worker really does emit its result
    and the GUI-side generation check is the only thing dropping it. Without it, the
    worker cancels the request before starting it and the assertion holds for free.
    """
    entered = threading.Event()

    def _resolve(*args: object) -> list[str]:
        entered.set()
        time.sleep(0.1)
        return ["late"]

    monkeypatch.setattr(_discovery, "resolve_descriptors", _resolve)
    found: list[object] = []
    discovery.streams_found.connect(found.append)
    discovery.refresh()
    assert entered.wait(10), "the worker never started the pass"
    discovery.stop()  # bumps the generation, then waits for the worker
    app.processEvents()
    assert found == []


def test_stop_leaves_a_stuck_worker_running(
    discovery: Discovery,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a worker which does not stop in time is leaked, never destroyed.

    A blocking liblsl call cannot be interrupted and 'QThread.terminate()' is out of the
    question, thus the only correct behaviour left is to let the thread finish on its
    own: it stays in '_discovery._RUNNING', which holds the only reference to it and to
    its worker, and destroying a running QThread is what would abort the whole process.
    """
    entered = threading.Event()

    def _resolve(*args: object) -> list[str]:
        entered.set()
        time.sleep(1.0)
        return []

    monkeypatch.setattr(_discovery, "resolve_descriptors", _resolve)
    monkeypatch.setattr(_discovery, "_STOP_TIMEOUT_MS", 50)  # forces the timeout path
    caplog.set_level(logging.WARNING, logger="mne_lsl")
    discovery.refresh()
    assert entered.wait(10), "the worker never started the pass"
    discovery.stop()
    assert "did not stop" in caplog.text
    assert discovery._thread.isRunning()
    assert discovery._thread in _discovery._RUNNING
    # it finishes on its own and unregisters itself, on the GUI thread.
    assert discovery._thread.wait(10000)
    app.processEvents()
    assert discovery._thread not in _discovery._RUNNING


def test_discovery_end_to_end(
    discovery: Discovery, qtbot: QtBot, outlets: Callable[..., StreamDescriptor]
) -> None:
    """Test that a real outlet is found through the worker thread."""
    descriptor = outlets()
    tags: list[str] = []
    discovery.progress.connect(tags.append)
    with qtbot.waitSignal(discovery.streams_found, timeout=15000) as blocker:
        discovery.refresh()
    (descriptors,) = blocker.args
    assert descriptor in descriptors
    assert tags == ["checking", "updated"]


# -- Connector -------------------------------------------------------------------------
def test_connector_releases_a_stale_stream(
    connector: Connector, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a stream published after its batch was replaced is disconnected.

    The worker-side check cannot see a 'stop' which happens after it emits, thus the
    GUI-side check is not redundant with it: each covers a different window.
    """
    dummy = _DummyStream()
    received: list[tuple] = []
    connector.connected.connect(lambda *args: received.append(args))
    connector._on_connected(42, descriptor(), dummy)
    assert received == []
    assert dummy.disconnected == 1


def test_connector_end_to_end(
    connector: Connector,
    qtbot: QtBot,
    outlets: Callable[..., StreamDescriptor],
    streams: list[BaseStream],
) -> None:
    """Test that a real outlet is connected through the worker thread."""
    descriptor = outlets(n_channels=3, ch_names=["Fp1", "Fp2", "Cz"])
    with qtbot.waitSignal(connector.connected, timeout=15000) as blocker:
        connector.open([descriptor], 2.0)
    published, stream = blocker.args
    streams.append(stream)  # disconnected by the fixture, whatever the assertions do
    assert published == descriptor
    assert stream.connected
    assert stream.ch_names == ["Fp1", "Fp2", "Cz"]


def test_connector_failure_isolation(
    connector: Connector,
    qtbot: QtBot,
    outlets: Callable[..., StreamDescriptor],
    descriptor: Callable[..., StreamDescriptor],
    streams: list[BaseStream],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a failing descriptor does not abort the ones which follow it."""
    absent = descriptor()  # its uuid4 'source_id' cannot be on the network
    present = outlets()
    caplog.set_level(logging.WARNING, logger="mne_lsl")
    events: list[tuple] = []
    connector.failed.connect(lambda *args: events.append(("failed", *args)))
    connector.connected.connect(lambda *args: events.append(("connected", *args)))
    with qtbot.waitSignal(connector.connected, timeout=20000):
        connector.open([absent, present], 2.0)
    streams.append(events[-1][2])
    assert [event[0] for event in events] == ["failed", "connected"]
    assert events[0][1] == absent
    assert events[1][1] == present
    assert "Could not connect" in caplog.text


def test_connector_stop_cancels_the_pending_connections(
    connector: Connector,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    descriptor: Callable[..., StreamDescriptor],
) -> None:
    """Test that 'stop' cancels the connections which have not started yet.

    The connection in flight cannot be interrupted, thus 'cancelled' means: it
    completes, its stream is released, and the batch stops there.
    """
    entered = threading.Event()
    dummies: list[_DummyStream] = []

    def _connect(item: StreamDescriptor, bufsize: float) -> _DummyStream:
        entered.set()
        time.sleep(0.3)
        dummy = _DummyStream()
        dummies.append(dummy)
        return dummy

    monkeypatch.setattr(_discovery, "connect_stream", _connect)
    received: list[tuple] = []
    connector.connected.connect(lambda *args: received.append(args))
    connector.open([descriptor("a"), descriptor("b"), descriptor("c")], 4.0)
    assert entered.wait(10), "the worker never started the first connection"
    connector.stop()
    app.processEvents()
    assert len(dummies) == 1, "the batch continued past the cancellation"
    assert dummies[0].disconnected == 1
    assert received == []


# -- Prober ----------------------------------------------------------------------------
def test_probe_worker_reports_every_outcome(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that one raising probe is reported without abandoning its siblings.

    A raise escaping a pool slot is invisible beyond a log line and would leave every
    sibling of the batch unreported, i.e. its cards checking forever. Run directly on
    the main thread: the worker is a plain QObject, so no thread and no network is used.
    """
    good, bad = descriptor(name="good"), descriptor(name="bad")

    def _probe(item: StreamDescriptor) -> list[str]:
        if item is bad:
            raise RuntimeError("the inlet did not open")
        return ["Fp1", "Cz"]

    monkeypatch.setattr(_discovery, "probe_channels", _probe)
    events: list[tuple] = []
    worker = _ProbeWorker()
    worker.generation = 5  # what 'Prober.probe' mirrors onto the worker
    worker.resolved.connect(lambda *args: events.append(("resolved", *args)))
    worker.failed.connect(lambda *args: events.append(("failed", *args)))
    worker.run(5, (good, bad))
    assert sorted(events) == sorted(
        [
            ("resolved", 5, good, ["Fp1", "Cz"]),
            ("failed", 5, bad, "the inlet did not open"),
        ]
    )


def test_probe_worker_skips_a_superseded_request(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a request whose generation is stale probes nothing at all.

    'QThread.quit' leaves the requests already posted in the thread's queue and a
    restarted thread drains them from the front, thus without this check every batch a
    'stop' cancelled would be replayed in full by the next 'probe'.
    """
    calls: list[StreamDescriptor] = []
    monkeypatch.setattr(_discovery, "probe_channels", lambda item: calls.append(item))
    events: list[tuple] = []
    worker = _ProbeWorker()
    worker.generation = 9
    worker.resolved.connect(lambda *args: events.append(args))
    worker.failed.connect(lambda *args: events.append(args))
    worker.run(8, (descriptor(),))
    assert calls == []
    assert events == []


def test_probe_worker_cancels_the_rest_on_a_generation_bump(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a batch superseded while it runs emits nothing more.

    The first probe to complete bumps the generation itself, so the check inside the
    completion loop is the only thing between a stale batch and six results written into
    the probe cache under uids the interface has already replaced.
    """
    submitted = tuple(descriptor(name=f"s{k}") for k in range(6))
    worker = _ProbeWorker()
    worker.generation = 1
    calls: list[StreamDescriptor] = []

    def _probe(item: StreamDescriptor) -> list[str]:
        calls.append(item)
        worker.generation = 2  # the pass is superseded from under the batch
        return ["Cz"]

    monkeypatch.setattr(_discovery, "probe_channels", _probe)
    events: list[tuple] = []
    worker.resolved.connect(lambda *args: events.append(args))
    worker.failed.connect(lambda *args: events.append(args))
    worker.run(1, submitted)
    assert calls, "the batch never started"
    assert events == []


def test_probe_worker_uses_the_configured_pool_size(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that the pool is sized by the module constant.

    Weak on its own, and kept because this is the single place the number appears: a
    dropped 'max_workers' makes six distinct streams cost three seconds instead of one,
    with nothing failing.
    """
    seen: list[int | None] = []

    class _Spy(ThreadPoolExecutor):
        """Executor recording the worker count it was built with."""

        def __init__(self, max_workers: int | None = None, **kwargs: object) -> None:
            seen.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(_discovery, "ThreadPoolExecutor", _Spy)
    monkeypatch.setattr(_discovery, "probe_channels", lambda item: ["Cz"])
    worker = _ProbeWorker()
    worker.generation = 1
    worker.run(1, (descriptor(),))
    assert seen == [_PROBE_WORKERS]
    assert _PROBE_WORKERS == 4


def test_prober_leaves_the_warning_filters_alone(
    monkeypatch: pytest.MonkeyPatch, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that the probe's warning suppression does not outlive its batch.

    The suppression is installed inside a 'catch_warnings' block. Dropping that block --
    or installing the filters permanently -- silently disables warnings-as-errors for
    every 'RuntimeWarning' of the rest of the session, which would hide unrelated
    defects in every later test rather than failing this one.
    """
    monkeypatch.setattr(_discovery, "probe_channels", lambda item: ["Cz"])
    worker = _ProbeWorker()
    worker.generation = 1
    worker.run(1, (descriptor(),))
    with pytest.raises(RuntimeWarning, match=_discovery._PROBE_NOTICE):
        warnings.warn(f"{_discovery._PROBE_NOTICE}.", RuntimeWarning, stacklevel=1)


def test_prober_drops_a_stale_result(
    prober: Prober, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a result from a superseded batch is never published.

    The worker-side check cannot see a 'stop' which happens after it emits, thus the
    owner-side check is not redundant with it: each covers a different window, and a
    result slipping through lands in the interface's probe cache under a stale uid.
    """
    received: list[tuple] = []
    prober.resolved.connect(lambda *args: received.append(args))
    prober.failed.connect(lambda *args: received.append(args))
    prober._on_resolved(42, descriptor(), ["Cz"])
    prober._on_failed(42, descriptor(), "boom")
    assert received == []


def test_prober_resolves_a_real_stream(
    prober: Prober, qtbot: QtBot, outlets: Callable[..., StreamDescriptor]
) -> None:
    """Test that a real outlet is probed through the worker thread.

    The **submitted descriptor** must come back, not its identity: the cache is keyed on
    '(identity, uid)', so an identity-only payload would degrade the cache to
    'cache forever' and a re-provisioned stream keeps its stale channel set.
    """
    descriptor = outlets(n_channels=3, ch_names=["Fp1", "Fp2", "Cz"])
    with qtbot.waitSignal(prober.resolved, timeout=15000) as blocker:
        prober.probe([descriptor])
    published, names = blocker.args
    assert published == descriptor
    assert published.uid == descriptor.uid
    assert names == ["Fp1", "Fp2", "Cz"]


def test_prober_resolves_a_nameless_stream(
    prober: Prober, qtbot: QtBot, outlets: Callable[..., StreamDescriptor]
) -> None:
    """Test that a stream publishing no channel description still probes successfully.

    The reader falls back to channel IDs and warns about it, and a warning is an error
    here, so without the worker's suppression the batch reports 'failed' and the card
    reads 'could not reach' for a stream which is present and perfectly matchable. Those
    fallback names are also what a *connection* to the same stream in the same process
    reports, which is what makes the two lists comparable. It additionally fails if the
    names are read from 'get_channel_names()', which returns nothing at all here.
    """
    descriptor = outlets(n_channels=3, ch_names=[])
    with qtbot.waitSignal(prober.resolved, timeout=15000) as blocker:
        prober.probe([descriptor])
    published, names = blocker.args
    assert published == descriptor
    assert names == ["0", "1", "2"]


def test_prober_resolves_a_duplicate_name_stream(
    prober: Prober, qtbot: QtBot, outlets: Callable[..., StreamDescriptor]
) -> None:
    """Test that a duplicate-name stream probes to whatever a connection would report.

    The property that matters is agreement, not which list wins: the probe decides
    whether a saved configuration is loadable, and the connection decides whether the
    load then succeeds, so the two reading one description differently is a card saying
    available above a load that refuses.

    Run with this suite's warnings-as-errors in force and **no ambient relaxation**,
    which is the whole point. The worker suppresses exactly one notice, the outer one
    about the description as a whole; the inner duplicate-name warning still reaches the
    reader, which catches it itself and falls back to channel identifiers. Widening the
    suppression to the inner warning would let the de-duplication through here and
    produce ``['Cz-0', 'Cz-1', 'Fp1']`` from the probe while a connection in the same
    process still reported ``['0', '1', '2']`` -- so this fails under exactly that
    mutation, which the previous version of it could not, having relaxed the filter that
    tells them apart.

    Not relaxing the filter also keeps the worker's own ``catch_warnings`` from
    overlapping one on this thread: both save and restore the process-global filter
    list, and whichever leaves last wins.
    """
    descriptor = outlets(n_channels=3, ch_names=["Cz", "Cz", "Fp1"])
    with qtbot.waitSignal(prober.resolved, timeout=15000) as blocker:
        prober.probe([descriptor])
    published, names = blocker.args
    assert published == descriptor
    assert names == ["0", "1", "2"]


def test_prober_stop_is_idempotent_and_restartable(
    prober: Prober, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that stopping twice is a no-op and that a probe restarts the worker."""
    monkeypatch.setattr(_discovery, "probe_channels", lambda item: ["Cz"])
    with qtbot.waitSignal(prober.resolved, timeout=10000):
        prober.probe([object()])
    prober.stop()
    assert not prober._thread.isRunning()
    prober.stop()
    assert not prober._thread.isRunning()
    with qtbot.waitSignal(prober.resolved, timeout=10000):
        prober.probe([object()])


# -- submit_reconnect ------------------------------------------------------------------
def _signature(**kwargs) -> StreamSignature:
    """Return a signature built of plain values, with every field defaulted."""
    fields = dict(
        identity=StreamIdentity(name="Polar", stype="eeg", source_id="src-1"),
        sfreq=100.0,
        dtype="float32",
        ch_names=("Fp1", "Fpz", "ECG", "TRIGGER"),
    )
    fields.update(kwargs)
    return StreamSignature(**fields)


@pytest.mark.parametrize(
    ("actual", "outcome", "detail", "released"),
    [
        pytest.param(_signature(), RESUME_LIVE, "", 0, id="live"),
        pytest.param(
            _signature(sfreq=200.0),
            RESUME_MISMATCH,
            "sampling rate changed",
            1,
            id="mismatch",
        ),
    ],
)
def test_submit_reconnect(
    app: QApplication,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    actual: StreamSignature,
    outcome: str,
    detail: str,
    released: int,
) -> None:
    """Test the two outcomes of a reconnection which did reconnect.

    A refused stream must be released on the worker: dropping it leaks a live inlet and
    its acquisition thread for the life of the process. Kills collapsing either branch,
    and kills forgetting the release.
    """
    stream = _DummyStream()
    monkeypatch.setattr(_discovery, "reconnect_stream", lambda item: None)
    monkeypatch.setattr(_discovery, "stream_signature", lambda item: actual)
    seen: list[tuple[str, str]] = []
    _discovery.submit_reconnect(stream, _signature(), lambda *args: seen.append(args))
    qtbot.waitUntil(lambda: len(seen) == 1, timeout=10000)
    assert seen[0][0] == outcome
    assert detail in seen[0][1]
    assert stream.disconnected == released


def test_submit_reconnect_retries_when_the_reconnection_raises(
    app: QApplication, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a reconnection which raised reports a retry with the reason.

    An absent identity is the normal case while the source is still down, so it must be
    an outcome and never an exception escaping the runnable, where it would be invisible
    beyond a log line. Kills dropping the 'except'.
    """

    def _raise(item: object) -> None:
        raise RuntimeError("do not uniquely identify an LSL stream")

    monkeypatch.setattr(_discovery, "reconnect_stream", _raise)
    seen: list[tuple[str, str]] = []
    _discovery.submit_reconnect(
        _DummyStream(), _signature(), lambda *args: seen.append(args)
    )
    qtbot.waitUntil(lambda: len(seen) == 1, timeout=10000)
    assert seen == [(RESUME_RETRY, "do not uniquely identify an LSL stream")]


def test_submit_reconnect_retries_when_the_signature_raises(
    app: QApplication, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a stream lost again mid-comparison is released and retried.

    'stream_signature' reads 'stream.info', which raises on a stream the acquisition
    thread has just reset, and the reconnection did open an inlet before that. Kills
    dropping the second 'except', which would leak that inlet.
    """
    stream = _DummyStream()

    def _raise(item: object) -> None:
        raise RuntimeError("The Stream is not connected")

    monkeypatch.setattr(_discovery, "reconnect_stream", lambda item: None)
    monkeypatch.setattr(_discovery, "stream_signature", _raise)
    seen: list[tuple[str, str]] = []
    _discovery.submit_reconnect(stream, _signature(), lambda *args: seen.append(args))
    qtbot.waitUntil(lambda: len(seen) == 1, timeout=10000)
    assert seen[0][0] == RESUME_RETRY
    assert stream.disconnected == 1
