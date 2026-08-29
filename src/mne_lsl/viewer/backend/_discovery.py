"""Background stream discovery and connection.

Both objects run their blocking work off the GUI thread and report through Qt signals,
which is the transport boundary: nothing outside this module touches a worker thread.

Notes
-----
This module names no LSL object and imports nothing from :mod:`mne_lsl.lsl`: the two
blocking calls it drives live in :mod:`~mne_lsl.viewer.backend._source`, and everything
which crosses a thread boundary here is either plain Python data or a
:class:`~mne_lsl.stream.BaseStream`.

Handing a stream across the boundary is safe, and is worth stating: a
:class:`~mne_lsl.stream.StreamLSL` has no Qt thread affinity -- its acquisition loop is
a :mod:`concurrent.futures` executor, and reading its buffer from a thread other than
the one which connected it is the normal contract of the class. A
:class:`~mne_lsl.lsl.StreamInfo` is precisely what this is *not*, which is why one never
leaves :mod:`~mne_lsl.viewer.backend._source`.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from qtpy.QtCore import (
    QCoreApplication,
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    Signal,
)

from ...utils.logs import logger
from ._identity import signature_mismatch
from ._source import (
    connect_stream,
    probe_channels,
    reconnect_stream,
    resolve_descriptors,
    stream_signature,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...stream import BaseStream
    from ._identity import StreamDescriptor, StreamSignature

# Outcome of one reconnection attempt, as the worker reports it to the document. Bare
# strings, so that they cross a 'Signal(str, str)' unchanged. Prefixed, and never a bare
# 'live': the document's own 'LIVE' state constant is the string 'live' and it imports
# both, so 'outcome == LIVE' reads as correct while comparing the wrong two constants.
RESUME_LIVE = "resume-live"
RESUME_RETRY = "resume-retry"
RESUME_MISMATCH = "resume-mismatch"

# Bounded wait on a worker at shutdown, in milliseconds. A blocking liblsl call cannot
# be interrupted, thus the wait has to cover the longest one which can be in flight, and
# that is one connection: 'StreamLSL.connect' applies its 'timeout' -- 2 s by default --
# once to the stream resolution, once to the opening of the inlet, which liblsl follows
# with a fixed 0.5 s sleep, and once to the time correction, i.e. ~6.5 s worst case.
# 10 s leaves a margin for a loaded host. Never 'QThread.terminate()', which Qt
# documents as able to stop a thread while it holds a lock.
_STOP_TIMEOUT_MS = 10000

# Threads which are running, mapped to the worker moved onto them. Destroying a running
# 'QThread' makes Qt call 'qFatal' and takes the whole process down, thus a thread and
# its worker have to outlive whatever owns them: both are parentless and this registry
# is what keeps them alive. It is what 'QThread(owner)' is not -- that ties the thread's
# lifetime to an object the GUI is free to drop, and dropping one while a resolution or
# a connection is in flight aborts the process. An entry is removed once its thread
# reports 'finished', and an owner which was itself dropped mid-pass leaves its entry
# behind: a deliberately leaked thread, which is always preferable to an abort.
_RUNNING: dict[QThread, QObject] = {}

# Channel probes run in parallel: one is ~0.5 s of hardcoded sleep inside
# 'StreamInlet.open_stream', thus they scale near-linearly, and four covers the handful
# of distinct streams a set of configurations names. Not a tuning knob.
_PROBE_WORKERS = 4

# The one upstream notice a channel probe deliberately accepts, matched on the start of
# its message. A stream publishing duplicate, blank or no channel names makes the reader
# fall back to channel IDs and warn about it -- and that fallback is exactly what a
# *connection* to the same stream in the same process reports, which is what makes the
# two name lists comparable at all. A consumer running with warnings as errors would
# otherwise get an unreachable stream for both degenerate descriptions, i.e. precisely
# the outcome the match-on-interpreted-names rule exists to prevent.
#
# Narrowed to this one message on purpose, and never widened to the 'Channel names are
# not unique' notice which the reader itself catches: suppressing that one would make
# the probe report MNE's de-duplicated names while a connection in the same process
# still fell back to channel IDs, and the two lists would then disagree for a stream
# which was perfectly matchable. A probe reproduces a connection, it does not improve
# on it.
_PROBE_NOTICE = "Something went wrong while reading the channel description"


def _ensure_running(thread: QThread, worker: QObject) -> None:
    """Start ``thread`` if it is not running, and hold it while it runs.

    Parameters
    ----------
    thread : QThread
        Thread carrying ``worker``.
    worker : QObject
        Worker which was moved to ``thread``.

    Notes
    -----
    Started lazily on the first request rather than in ``__init__``, and restarted here
    after a stop: a finished :class:`~qtpy.QtCore.QThread` restarts cleanly, and its
    worker keeps its affinity to it.
    """
    if thread.isRunning():
        return
    _RUNNING[thread] = worker  # see '_RUNNING'
    thread.start()


def _stop_thread(thread: QThread, kind: str) -> None:
    """Ask ``thread`` to leave its event loop and wait for it, bounded.

    Parameters
    ----------
    thread : QThread
        Thread to stop; a thread which is not running is a no-op.
    kind : str
        Which worker this thread carries, for the warning message.

    Notes
    -----
    A thread which does not stop within :data:`_STOP_TIMEOUT_MS` is deliberately leaked:
    it stays registered in ``_RUNNING`` and keeps running until its blocking call
    returns on its own, at which point it unregisters itself. Both alternatives are
    worse -- :meth:`~qtpy.QtCore.QThread.terminate` can stop a thread while it holds a
    lock, and destroying it aborts the process -- while the leak costs one thread stack,
    and one inlet at worst, in a session which is already shutting down.
    """
    if not thread.isRunning():
        return
    thread.quit()
    if not thread.wait(_STOP_TIMEOUT_MS):
        logger.warning(
            "The stream %s worker did not stop within %.1f s; it is left running until "
            "its blocking call returns.",
            kind,
            _STOP_TIMEOUT_MS / 1000,
        )


def wait_for_reconnects() -> None:
    """Wait for the reconnection tasks still on the global thread pool; bounded.

    Notes
    -----
    The counterpart, at shutdown, of the ``stop()`` each worker owner offers: a
    reconnection is a :class:`~qtpy.QtCore.QRunnable` on the global pool, so there is no
    thread of its own to stop and nothing in a window's teardown reaches it. Without
    this the pool is drained by its own destructor instead -- measured, ~3.2 s of a
    process which had already returned from ``closeEvent``, with the emitter's C++
    object already destroyed, so the outcome is delivered to nobody and a stream the
    task connected stays open with its acquisition thread for the life of the process.

    Bounded by :data:`_STOP_TIMEOUT_MS` for the reason :func:`_stop_thread` records: a
    blocking liblsl call cannot be interrupted, and one connection is the longest one
    which can be in flight.

    ponytail: the *global* pool, so this also waits out unrelated work an embedder's
    host application submitted to it. The upgrade is a pool dedicated to reconnections,
    which would additionally stop a saturated global pool from starving a queued
    document; nothing else in-process uses it today.
    """
    QThreadPool.globalInstance().waitForDone(_STOP_TIMEOUT_MS)


def release_stream(stream: BaseStream) -> None:
    """Disconnect a stream nobody will ever receive.

    Parameters
    ----------
    stream : BaseStream
        A connected stream whose result was rejected as stale.

    Notes
    -----
    Logged and swallowed: this runs on the failure path of a cancelled pass, where
    raising would replace a leaked inlet with an unhandled exception. Not disconnecting
    is not an option -- a dropped connected stream leaks a live inlet *and* its
    acquisition thread for the life of the process, and it is the one silent failure
    mode of this module.

    Public because a reconnection outcome may reach a document which has been torn down
    in the meantime, and that document then has the same stream to release.
    """
    try:
        stream.disconnect()
    except Exception as error:  # deliberately broad, see the note above
        logger.warning("Could not release a stale stream: %s", error)


class _DiscoveryWorker(QObject):
    """Resolve the streams on the network, on a worker thread.

    A plain :class:`~qtpy.QtCore.QObject`, so that :meth:`run` is directly callable on
    the main thread in the tests, with no thread involved at all.

    Attributes
    ----------
    done : Signal
        Emitted with ``(generation, list[StreamDescriptor])`` on success.
    failed : Signal
        Emitted with ``(generation, message)`` on failure.
    """

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the worker."""
        super().__init__(parent)
        # Written by the GUI thread, read by this thread when it picks a request up: it
        # is what makes a request superseded before it ran self-cancel. A plain 'int'
        # write and read needs no lock under the GIL, and no stronger mechanism would
        # help: the resolution in flight cannot be interrupted.
        self.generation = 0

    def run(self, generation: int) -> None:
        """Run one discovery pass and report its outcome.

        Parameters
        ----------
        generation : int
            Generation this pass was issued with, echoed back so that the owner can drop
            a stale result.

        Notes
        -----
        A request whose generation is no longer the current one returns without
        resolving. :meth:`~qtpy.QtCore.QThread.quit` leaves the requests already posted
        to this thread in its queue, and a restarted thread drains them from the front,
        so without this check every ``refresh()`` which a ``stop()`` cancelled would be
        replayed -- one full resolution each -- by the next ``refresh()``.
        """
        if self.generation != generation:
            return  # superseded before this pass was started
        try:
            descriptors = resolve_descriptors()
        except Exception as error:
            # Broad on purpose: this is a thread boundary. An escaping exception would
            # leave the interface stuck on 'Checking...' forever, with nothing but a log
            # line from the process-wide thread hook to explain it.
            self.failed.emit(generation, str(error))
            return
        self.done.emit(generation, descriptors)


class Discovery(QObject):
    """Resolve the streams present on the network, without blocking the GUI.

    Attributes
    ----------
    progress : Signal
        Emitted with a state tag, one of ``'checking'``, ``'updated'``, ``'failed'`` or
        ``'empty'``.
    streams_found : Signal
        Emitted with the ``list`` of :class:`~mne_lsl.viewer.backend.StreamDescriptor`
        found by the last pass, regular and irregular streams alike.

    Notes
    -----
    Cancellation is stale-work rejection, not interruption: a monotonic generation
    counter lives on the GUI side, every request carries the generation it was issued
    with, and a result whose generation no longer matches is dropped. The counter is
    mirrored onto the worker as well, which additionally lets a request the worker has
    not picked up yet cancel itself instead of running a pointless pass, see
    :meth:`_DiscoveryWorker.run`. The resolution *in flight* is still not interrupted,
    because a blocking liblsl call cannot be.

    ``streams_found`` is emitted with an empty list on ``'empty'``, so that the table
    clears, and is deliberately *not* emitted on ``'failed'``, so that a transient
    network failure leaves the last known good list on screen rather than flickering it
    away.
    """

    progress = Signal(str)
    streams_found = Signal(object)
    # Private, and typed 'int': emitted by the GUI thread and delivered to the worker's
    # own thread, where the connection's affinity makes it queued without any explicit
    # 'Qt.QueuedConnection'.
    _request = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the discovery object."""
        super().__init__(parent)
        self._generation = 0
        self._thread = QThread()  # parentless, held by '_RUNNING' while it runs
        self._thread.setObjectName("mne-lsl-viewer-discovery")
        # Parentless too, for another reason: 'moveToThread' refuses an object with a
        # parent.
        self._worker = _DiscoveryWorker()
        self._worker.moveToThread(self._thread)
        self._request.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.finished.connect(self._on_thread_finished)
        # Belt and braces: 'ViewerWindow.closeEvent' is expected to stop this object,
        # and this covers a shutdown which does not go through it. 'stop' is idempotent,
        # thus both paths running is a no-op.
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    def refresh(self) -> None:
        """Start one discovery pass, replacing a pass which is still running."""
        self._generation += 1  # any in-flight result is now stale
        self._worker.generation = self._generation  # cancels a superseded request
        # emitted synchronously, so the progress label updates before the work starts.
        self.progress.emit("checking")
        _ensure_running(self._thread, self._worker)
        self._request.emit(self._generation)
        # ponytail: one resolve per Refresh click. Two rapid clicks run two passes
        # back-to-back on the one worker thread and the first result is discarded;
        # coalesce with a pending flag if Refresh-mashing becomes a real complaint.

    def stop(self) -> None:
        """Stop the running pass and wait for its worker; idempotent."""
        self._generation += 1  # reject the result of a pass which is already in flight
        self._worker.generation = self._generation  # cancel the requests still queued
        _stop_thread(self._thread, "discovery")

    def _on_thread_finished(self) -> None:
        """Release the worker thread from ``_RUNNING`` once it has finished.

        Connected to a bound method of this object rather than to a free function on
        purpose: this object lives on the GUI thread, thus the connection is queued and
        the entry is dropped there. Dropping it from the default context of that signal
        -- the worker thread, which is emitting ``finished`` and is not finished yet --
        could destroy the thread from within itself, which is one more way to abort the
        process. A thread which has been restarted by the time this arrives keeps its
        entry.
        """
        if not self._thread.isRunning():
            _RUNNING.pop(self._thread, None)

    def _on_done(self, generation: int, descriptors: object) -> None:
        """Publish the descriptors of a pass which is still current.

        Left undecorated, as every slot of this module: a mis-specified ``@Slot`` fails
        silently at connect time, while a plain Python callable is correct on both
        bindings.

        Parameters
        ----------
        generation : int
            Generation the pass was issued with.
        descriptors : list of StreamDescriptor
            The descriptors found by the pass.
        """
        if generation != self._generation:
            return  # stale pass, dropped
        self.progress.emit("empty" if len(descriptors) == 0 else "updated")
        self.streams_found.emit(descriptors)

    def _on_failed(self, generation: int, message: str) -> None:
        """Report the failure of a pass which is still current.

        Parameters
        ----------
        generation : int
            Generation the pass was issued with.
        message : str
            Text of the exception which was raised.
        """
        if generation != self._generation:
            return
        logger.warning("Stream discovery failed: %s", message)
        self.progress.emit("failed")


class _ConnectorWorker(QObject):
    """Connect to a batch of streams sequentially, on a worker thread.

    Attributes
    ----------
    connected : Signal
        Emitted with ``(generation, descriptor, stream)`` per connected stream.
    failed : Signal
        Emitted with ``(generation, descriptor, message)`` per failed connection.
    """

    connected = Signal(int, object, object)
    failed = Signal(int, object, str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the worker."""
        super().__init__(parent)
        # Written by the GUI thread, read by this thread between two connections: it is
        # what makes 'Connector.stop()' able to cancel the connections it has not
        # started yet. A plain 'int' write and read needs no lock under the GIL, and no
        # stronger mechanism would help: the connection in flight cannot be interrupted.
        self.generation = 0

    def run(self, generation: int, descriptors: object, bufsize: float) -> None:
        """Connect to every descriptor in order, reporting each outcome.

        Parameters
        ----------
        generation : int
            Generation this batch was issued with.
        descriptors : sequence of StreamDescriptor
            Descriptors of the streams to connect to, in order.
        bufsize : float
            Size of the stream buffers, in seconds.

        Notes
        -----
        Sequential, so that a failure is unambiguously attributable to one stream, and a
        failure never stops the others: the all-or-nothing rollback of a configuration
        load is the caller's decision, not this loop's.

        ``# ponytail: sequential connect, parallelise if >4-stream configurations become
        common.``
        """
        for descriptor in descriptors:
            if self.generation != generation:
                return  # cancelled before this connection was started
            try:
                stream = connect_stream(descriptor, bufsize)
            except Exception as error:
                # As in '_DiscoveryWorker.run': an exception escaping a worker slot is
                # invisible beyond a log line, and here it would additionally abandon
                # every descriptor which comes after this one.
                self.failed.emit(generation, descriptor, str(error))
                continue
            if self.generation != generation:
                # cancelled *during* the connection, which takes about a second: the
                # stream is live and nobody is listening for it any more.
                release_stream(stream)
                return
            self.connected.emit(generation, descriptor, stream)


class Connector(QObject):
    """Connect to streams in the background, one document at a time.

    A failed connection is reported and leaves the already connected streams untouched;
    the all-or-nothing rollback of a configuration load is the caller's decision.

    Attributes
    ----------
    connected : Signal
        Emitted with ``(descriptor, stream)`` for every stream which connected, the
        stream being a :class:`~mne_lsl.stream.BaseStream`.
    failed : Signal
        Emitted with ``(descriptor, message)`` when a connection failed.

    Notes
    -----
    The generation counter is mirrored onto the worker, as :class:`Discovery`'s is, but
    it buys more here: a batch of connections is a loop with one cancellation point per
    item, thus :meth:`stop` really does cancel the connections which have not started
    yet, while a discovery pass is one atomic uninterruptible call and its mirror can
    only cancel a request the worker has not picked up yet.

    The thread lifecycle below duplicates :class:`Discovery`'s handful of lines rather
    than sharing a base class with it: the two request signatures differ and so do the
    two cancellation models, so a common base would exist only to be parameterised by
    both. What the two do share -- starting a thread and stopping it -- is factored out
    into :func:`_ensure_running` and :func:`_stop_thread`.
    """

    connected = Signal(object, object)
    failed = Signal(object, str)
    _request = Signal(int, object, float)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the connector object."""
        super().__init__(parent)
        self._generation = 0
        self._thread = QThread()  # parentless, see 'Discovery.__init__' and '_RUNNING'
        self._thread.setObjectName("mne-lsl-viewer-connector")
        self._worker = _ConnectorWorker()  # parentless too, see 'Discovery.__init__'
        self._worker.moveToThread(self._thread)
        self._request.connect(self._worker.run)
        self._worker.connected.connect(self._on_connected)
        self._worker.failed.connect(self._on_failed)
        self._thread.finished.connect(self._on_thread_finished)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    def open(self, descriptors: Sequence[StreamDescriptor], bufsize: float) -> None:
        """Connect to every stream of ``descriptors`` in the background.

        Parameters
        ----------
        descriptors : sequence of StreamDescriptor
            Descriptors of the streams to connect to.
        bufsize : float
            Size of the stream buffers, in seconds.
        """
        self._generation += 1
        self._worker.generation = self._generation  # visible to the worker's loop
        _ensure_running(self._thread, self._worker)
        # copied to a tuple, so that a caller mutating its own sequence afterwards
        # cannot change the work which was already submitted.
        self._request.emit(self._generation, tuple(descriptors), float(bufsize))

    def stop(self) -> None:
        """Cancel the pending connections and wait for the worker; idempotent."""
        self._generation += 1
        self._worker.generation = self._generation
        _stop_thread(self._thread, "connector")

    def _on_thread_finished(self) -> None:
        """Release the worker thread from ``_RUNNING``; see 'Discovery'."""
        if not self._thread.isRunning():
            _RUNNING.pop(self._thread, None)

    def _on_connected(
        self, generation: int, descriptor: object, stream: object
    ) -> None:
        """Publish a stream which connected during the current batch.

        Parameters
        ----------
        generation : int
            Generation the batch was issued with.
        descriptor : StreamDescriptor
            Descriptor of the stream which connected.
        stream : BaseStream
            The connected stream.
        """
        if generation != self._generation:
            # The batch was replaced or stopped after the worker's own check and before
            # this slot ran; the worker could not see that. Nobody will hear about this
            # stream, thus it must be released here or it leaks.
            release_stream(stream)
            return
        self.connected.emit(descriptor, stream)

    def _on_failed(self, generation: int, descriptor: object, message: str) -> None:
        """Report a connection which failed during the current batch.

        Parameters
        ----------
        generation : int
            Generation the batch was issued with.
        descriptor : StreamDescriptor
            Descriptor of the stream which failed to connect.
        message : str
            Text of the exception which was raised.
        """
        if generation != self._generation:
            return
        logger.warning(
            "Could not connect to the stream %s: %s",
            descriptor.identity.as_tuple(),
            message,
        )
        self.failed.emit(descriptor, message)


class _ProbeWorker(QObject):
    """Read the channel names of a batch of streams, on a pool of worker threads.

    Attributes
    ----------
    resolved : Signal
        Emitted with ``(generation, descriptor, list[str])`` per probed stream.
    failed : Signal
        Emitted with ``(generation, descriptor, message)`` per probe which raised.
    """

    resolved = Signal(int, object, object)
    failed = Signal(int, object, str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the worker."""
        super().__init__(parent)
        # Written by the GUI thread, read by this thread before every emission: see
        # '_ConnectorWorker.__init__' for why a plain 'int' is enough.
        self.generation = 0

    def run(self, generation: int, descriptors: object) -> None:
        """Probe every descriptor in parallel, reporting each outcome as it lands.

        Parameters
        ----------
        generation : int
            Generation this batch was issued with.
        descriptors : sequence of StreamDescriptor
            Descriptors of the streams to probe.

        Notes
        -----
        Nothing but plain data crosses out: the stream info a probe creates is created,
        read and destroyed inside one call on a pool thread, and what comes back is a
        ``list`` of names. The pool threads emit these signals themselves, which is safe
        precisely because of that -- the connection to the owner is cross-thread and
        therefore queued.

        Cancellation is stale-work rejection, never interruption. The generation is
        re-read before every emission and a superseded batch cancels the futures which
        have not started, which is all :meth:`~concurrent.futures.Future.cancel` can do.
        The ``with`` block then joins, so a superseded pass still waits out the probes
        already in flight: a blocking liblsl call cannot be interrupted, thus a watchdog
        would be theatre.

        The warning suppression covers exactly the upstream notice of
        :data:`_PROBE_NOTICE` and lives *here* rather than inside the probe itself.
        :func:`warnings.catch_warnings` saves and restores the **global** filter list
        with no thread isolation, so entering and leaving it on four pool threads at
        once would leak the ignore filter permanently and silently disable
        warnings-as-errors for :class:`RuntimeWarning` for the rest of the process. One
        ``catch_warnings`` on this single worker thread, joined by the ``with`` before
        the next batch is picked up, is the only placement with no race **among the pool
        threads**.

        It does not remove the race against other threads, and cannot: whichever of two
        overlapping blocks leaves last restores the list it captured, so a
        ``catch_warnings`` entered elsewhere while a batch is in flight can resurrect a
        filter that thread had removed, or drop one it had installed. The GUI thread
        does enter one, measured: pyqtgraph's ``boundingRect``/``dataBounds`` enter
        ``catch_warnings`` on the **render path**, 48--50 times per paint, so a document
        drawing at 30 Hz overlaps any batch which is in flight. The alternatives are
        nonetheless worse -- a permanent filter installed at import is a module-level
        side effect, which this package does not allow, and it would silence the notice
        for an embedder's own calls -- so the real fix belongs upstream, in the reader
        which warns even when its de-duplication produced usable names.

        The suppression covers only the outer notice. The inner duplicate-name warning
        still reaches the reader, which catches it itself and falls back to channel
        identifiers -- the same list a connection reports, which is the agreement the
        availability check depends on.
        """
        if self.generation != generation:
            return  # superseded before this batch was started
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=_PROBE_NOTICE, category=RuntimeWarning
            )
            with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
                futures = {
                    pool.submit(probe_channels, descriptor): descriptor
                    for descriptor in descriptors
                }
                for future in as_completed(futures):
                    if self.generation != generation:
                        for pending in futures:
                            pending.cancel()
                        return
                    descriptor = futures[future]
                    try:
                        names = future.result()
                    except Exception as error:
                        # As in the two other workers: an exception escaping a worker
                        # slot is invisible beyond a log line, and here it would leave
                        # every sibling of this batch unreported as well.
                        self.failed.emit(generation, descriptor, str(error))
                        continue
                    self.resolved.emit(generation, descriptor, names)


class Prober(QObject):
    """Read the channel names of streams in the background, in parallel.

    Discovery reports the channel *count* but not the names, which need an inlet. This
    is the transport of that second round trip, kept separate from :class:`Discovery` so
    that the interface can show the check as it happens: folding the two into one worker
    call would make 'checking availability' a state no user ever sees.

    Attributes
    ----------
    resolved : Signal
        Emitted with ``(descriptor, list[str])`` for every stream which was probed. The
        submitted descriptor comes back, not its identity, because the caller keys its
        cache on the descriptor's ``uid`` as well.
    failed : Signal
        Emitted with ``(descriptor, message)`` when a probe raised.

    Notes
    -----
    A generation counter of its own, mirrored onto the worker exactly as
    :class:`Discovery`'s and :class:`Connector`'s are. Deliberately **not** shared with
    the discovery counter: the two passes are independent, and a shared counter would
    let a refresh which superseded a discovery invalidate an unrelated probe batch.

    This object knows nothing about configurations: it does not de-duplicate -- the
    caller submits distinct descriptors -- and it does not cache, because the cache is
    keyed on ``(identity, uid)`` and read by the interface before a batch is submitted.
    """

    resolved = Signal(object, object)
    failed = Signal(object, str)
    _request = Signal(int, object)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the prober object."""
        super().__init__(parent)
        self._generation = 0
        self._thread = QThread()  # parentless, see 'Discovery.__init__' and '_RUNNING'
        self._thread.setObjectName("mne-lsl-viewer-prober")
        self._worker = _ProbeWorker()  # parentless too, see 'Discovery.__init__'
        self._worker.moveToThread(self._thread)
        self._request.connect(self._worker.run)
        self._worker.resolved.connect(self._on_resolved)
        self._worker.failed.connect(self._on_failed)
        self._thread.finished.connect(self._on_thread_finished)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

    def probe(self, descriptors: Sequence[StreamDescriptor]) -> None:
        """Read the channel names of every stream of ``descriptors``, in the background.

        Parameters
        ----------
        descriptors : sequence of StreamDescriptor
            Descriptors of the streams to probe. An empty sequence still starts a batch,
            which reports nothing.
        """
        self._generation += 1
        self._worker.generation = self._generation  # visible to the worker's loop
        _ensure_running(self._thread, self._worker)
        # copied to a tuple, as in 'Connector.open': a caller mutating its own sequence
        # afterwards cannot change the work which was already submitted.
        self._request.emit(self._generation, tuple(descriptors))

    def stop(self) -> None:
        """Cancel the pending probes and wait for the worker; idempotent."""
        self._generation += 1
        self._worker.generation = self._generation
        _stop_thread(self._thread, "prober")

    def _on_thread_finished(self) -> None:
        """Release the worker thread from ``_RUNNING``; see 'Discovery'."""
        if not self._thread.isRunning():
            _RUNNING.pop(self._thread, None)

    def _on_resolved(self, generation: int, descriptor: object, names: object) -> None:
        """Publish the channel names of a probe which belongs to the current batch.

        Parameters
        ----------
        generation : int
            Generation the batch was issued with.
        descriptor : StreamDescriptor
            Descriptor of the stream which was probed.
        names : list of str
            The channel names, in acquisition order.
        """
        if generation != self._generation:
            return  # stale batch: nothing was acquired, thus nothing to release either
        self.resolved.emit(descriptor, names)

    def _on_failed(self, generation: int, descriptor: object, message: str) -> None:
        """Report a probe which failed during the current batch.

        Parameters
        ----------
        generation : int
            Generation the batch was issued with.
        descriptor : StreamDescriptor
            Descriptor of the stream which could not be probed.
        message : str
            Text of the exception which was raised, shown by the interface.
        """
        if generation != self._generation:
            return
        logger.debug(
            "Could not probe the stream %s: %s", descriptor.identity.as_tuple(), message
        )
        self.failed.emit(descriptor, message)


class _ReconnectSignals(QObject):
    """Emitter carrying the outcome of one reconnection back to the GUI thread.

    Attributes
    ----------
    finished : Signal
        Emitted once with ``(outcome, detail)``, where ``outcome`` is one of
        :data:`RESUME_LIVE`, :data:`RESUME_MISMATCH` and :data:`RESUME_RETRY`.

    Notes
    -----
    A separate object rather than the document which asked for the reconnection: a
    document is a dock widget whose C++ object the docking framework may already have
    destroyed, and emitting on a destroyed object from a worker thread raises there.

    What makes a *late* outcome safe is not this object being collected with its owner:
    it is not: :class:`_ReconnectTask` holds a strong reference to it for the whole of
    ``run()``, which is precisely the window in question. It is that reference, plus Qt
    severing a connection whose receiver was destroyed, so the emission reaches nobody
    instead of a dangling document. The emitter's own C++ object can still go away
    first, at shutdown, which is why :meth:`_ReconnectTask._emit` treats a dead emitter
    as a stream to release rather than as an error.
    """

    finished = Signal(str, str)


class _ReconnectTask(QRunnable):
    """Reconnect one stream in place and evaluate whether it may be resumed.

    Parameters
    ----------
    stream : BaseStream
        The stream to reconnect, connected or not.
    expected : StreamSignature
        Signature recorded while the document was live.
    signals : _ReconnectSignals
        Emitter to report the outcome through.

    Notes
    -----
    Composition rather than ``class _ReconnectTask(QObject, QRunnable)``: multiple
    inheritance from both is a documented PySide6 hazard.

    The match is evaluated here, on the worker, rather than by the document: refusing a
    stream means disconnecting it, which takes about half a second, and doing that on
    the GUI thread is a visible freeze. The document still owns the state machine --
    this computes a fact and hands over a reason string.
    """

    def __init__(
        self,
        stream: BaseStream,
        expected: StreamSignature,
        signals: _ReconnectSignals,
    ) -> None:
        super().__init__()
        self._stream = stream
        self._expected = expected
        self._signals = signals

    def run(self) -> None:
        """Reconnect, compare, and report exactly one outcome."""
        try:
            reconnect_stream(self._stream)
        except Exception as error:
            # Deliberately broad: an absent identity, a refused inlet and a stream which
            # came back as a string stream are all "not back yet", and an exception
            # escaping a runnable is invisible beyond a log line.
            self._emit(RESUME_RETRY, str(error))
            return
        try:
            reason = signature_mismatch(self._expected, stream_signature(self._stream))
        except Exception as error:  # a stream which was lost again mid-comparison
            release_stream(self._stream)
            self._emit(RESUME_RETRY, str(error))
            return
        if reason is None:
            self._emit(RESUME_LIVE, "")
            return
        # refused: nobody will draw it, so it must not stay open
        release_stream(self._stream)
        self._emit(RESUME_MISMATCH, reason)

    def _emit(self, outcome: str, detail: str) -> None:
        """Report one outcome, releasing the stream if nobody can hear it.

        Parameters
        ----------
        outcome : str
            One of :data:`RESUME_LIVE`, :data:`RESUME_MISMATCH`, :data:`RESUME_RETRY`.
        detail : str
            The refusal reason, or the text of the exception which failed the attempt.

        Notes
        -----
        The emitter's C++ object can be destroyed while this task is still running: a
        shutdown which tears the window down and returns leaves this pool thread inside
        a blocking call, and the emission then raises ``RuntimeError: wrapped C/C++
        object ... has been deleted`` -- on a pool thread, outside every ``try``, which
        loses a stream this task had just connected, and its acquisition thread with it.
        This is the last chance to release that stream, so the failure is handled here
        rather than left to escape ``run()``, where nothing but a log line would see it.
        """
        try:
            self._signals.finished.emit(outcome, detail)
        except RuntimeError as error:  # the emitter went with the shutdown
            logger.warning("Could not report a reconnection outcome: %s", error)
            if outcome == RESUME_LIVE:
                release_stream(self._stream)


def submit_reconnect(
    stream: BaseStream, expected: StreamSignature, on_finished: Callable
) -> _ReconnectSignals:
    """Reconnect ``stream`` on the global thread pool and report the outcome.

    Parameters
    ----------
    stream : BaseStream
        The stream to reconnect, connected or not.
    expected : StreamSignature
        Signature recorded while the document was live.
    on_finished : Callable
        Slot called once with ``(outcome, detail)``, on the GUI thread.

    Returns
    -------
    signals : _ReconnectSignals
        Emitter whose ``finished`` signal ``on_finished`` is connected to. Returned so
        that a caller can hold it as the handle of an attempt in flight; it is not what
        the connection is made through.

    Notes
    -----
    ``on_finished`` is an argument rather than something the caller connects to the
    returned emitter, because Qt resolves the receivers of a signal at emit time: a
    reconnection which fails fast -- an identity absent from the network raises as soon
    as the resolution times out, and the pool may run the task before this function has
    even returned -- would emit into no receiver at all and the attempt would then hang
    forever with nothing in flight.

    A :class:`~qtpy.QtCore.QRunnable` on the global pool rather than a fourth
    thread-and-worker pair: a reconnection is per-document and one-shot, so a facade
    owning a :class:`~qtpy.QtCore.QThread` would mean one idle thread per open document
    plus a ``stop()`` to wire into every teardown path, and destroying a running
    ``QThread`` aborts the process.
    """
    signals = _ReconnectSignals()
    signals.finished.connect(on_finished)  # before 'start', see the note above
    QThreadPool.globalInstance().start(_ReconnectTask(stream, expected, signals))
    return signals
