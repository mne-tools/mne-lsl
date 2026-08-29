from __future__ import annotations

import ast
import multiprocessing as mp
import os
import sysconfig
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mne_lsl.lsl import StreamInfo, StreamOutlet
from mne_lsl.stream import StreamLSL

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType

    from qtpy.QtWidgets import QApplication

    from mne_lsl.viewer._window import ViewerWindow
    from mne_lsl.viewer.backend import StreamDescriptor
    from mne_lsl.viewer.theme import ThemeController
    from mne_lsl.viewer.widgets import EditableReadout

# render offscreen unless the environment asks for something else, so the tests run on a
# headless machine without a display server.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Gate the entire package: the viewer is not free-threading safe, and 'mne_lsl.viewer'
# raises on such a build, while a missing Qt binding makes every test unrunnable.
#
# 'collect_ignore_glob' is the right tool: pytest's own default 'pytest_ignore_collect'
# hook implementation (see '_pytest/main.py') is *driven by* this list, so a custom hook
# here would just reimplement it. Per that same source, a file matched by
# 'collect_ignore_glob' is skipped strictly before 'pytest_collect_file' -- the hook
# that imports the module -- ever runs. That ordering is what makes plain, module-level
# 'qtpy'/'pyqtgraph' imports in 'tests/viewer/test_*.py' safe: those modules are only
# ever imported once this conftest has already confirmed Qt is importable.
#
# 'pytest.importorskip("qtpy")' is not usable here, for two independent reasons. First,
# qtpy imports successfully and only then raises 'QtBindingsNotFoundError' (an
# 'ImportError' subclass, but not a 'ModuleNotFoundError') when no binding is installed;
# pytest's 'importorskip' defaults its 'exc_type' to 'ModuleNotFoundError' since 9.1, so
# it no longer catches that and re-raises instead of skipping (passing
# 'exc_type=ImportError' would catch it, but see the second reason). Second, and more
# fundamentally, 'importorskip' skips one test module/function by raising 'Skipped';
# calling it here, at conftest module level, would abort loading this conftest instead
# of quietly excluding the directory, which is what 'collect_ignore_glob' does instead.
collect_ignore_glob: list[str] = []
if sysconfig.get_config_var("Py_GIL_DISABLED"):
    collect_ignore_glob = ["*"]
else:
    try:
        import qtpy  # noqa: F401
    except ImportError:
        collect_ignore_glob = ["*"]

# Bound on the wait for a pushed chunk to reach the stream buffer. Generous, as it is
# only ever reached when the local link is broken; a local outlet delivers in ~10 ms.
_PUSH_DEADLINE = 10.0
# Acquisition period of the connected stream, in seconds. Short, so that a pushed chunk
# is picked up without the test having to wait a default 1 ms x n cycles.
_ACQUISITION_DELAY = 0.01
# Bound on the wait for a freshly created outlet to become resolvable. Generous, as it
# is only ever reached when discovery is broken; a local outlet answers instantly.
_RESOLVE_DEADLINE = 10.0


@pytest.fixture(scope="session")
def app() -> Generator[QApplication, None, None]:
    """Yield the session-wide offscreen QApplication."""
    # Nested deliberately: this conftest's body runs even when the gate above excludes
    # every test, and importing 'mne_lsl.viewer' raises without a Qt binding. Hoisting
    # this would fail the whole suite at collection on a binding-less machine.
    from mne_lsl.viewer._bootstrap import ensure_application

    application = ensure_application("mne-lsl viewer (tests)")
    yield application
    application.processEvents()


@pytest.fixture
def config_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point 'Path.home()' at a temporary directory and return it.

    The configuration directory is computed from 'Path.home()' on every call precisely
    so that it can be redirected here, instead of being frozen into a module constant at
    import time.

    This fixture lives here rather than in 'backend/' because the window, the launcher
    and the persistence layer all need it.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def lsl_stream(
    request: pytest.FixtureRequest,
) -> Generator[Callable[..., tuple[StreamLSL, Callable[..., None]]]]:
    """Yield a factory creating a connected stream fed by a real outlet.

    Rendering, the stim edge detection and the x-mapping all need actual samples, thus
    the silent outlet of 'tests/viewer/backend/' is no longer enough here -- but a
    'PlayerLSL', a subprocess and a testing dataset still are not needed: an outlet and
    'push_chunk' cover every assertion and are far less flaky.

    The factory returns the connected stream and a ``push`` callable. Teardown
    disconnects the stream and destroys the outlet, in the fixture rather than in the
    test bodies, so that a failing assertion still tears both down.

    This fixture lives here rather than in one subpackage's conftest because both the
    trace display and the channel model are built over a connected stream.
    """
    created: list[tuple[StreamLSL, StreamOutlet]] = []

    def _start(
        n_channels: int = 8,
        sfreq: float = 100.0,
        n_stim: int = 1,
        units: str = "uv",
        bufsize: float = 2.0,
        ch_types: list[str] | None = None,
        ch_names: list[str] | None = None,
        ch_units: list[str] | None = None,
        dtype: str = "float32",
    ) -> tuple[StreamLSL, Callable[..., None]]:
        """Create one outlet, connect a stream to it and return ``(stream, push)``.

        The channel names are explicit and unique: MNE's 'create_info' warns on a dup
        names during 'connect()', and this suite turns warnings into errors. The last
        'n_stim' channels are stim channels, unless ``ch_types`` overrides the whole
        list; ``ch_names`` and ``ch_units`` override theirs the same way, which is what
        lets a test publish a mixed-type stream or a duplicate name on purpose.

        ``dtype`` is the channel format published on the wire, e.g. ``'int32'``: an
        integer stream is a legal LSL stream and the display draws one, which is what
        makes the NaN rule of an un-filled window dtype-dependent.

        The LSL name is suffixed with the creation count, not only with the test name:
        a test asking the factory twice would otherwise publish two outlets under one
        name, and 'StreamLSL.connect()' resolves on the name and the type before the
        'source_id', so the second connection could land on the first outlet.

        'bufsize' carries the mne-lsl semantics: seconds for a regularly sampled stream,
        samples when 'sfreq' is 0, i.e. for the irregular stream the render loop has its
        own branch for.
        """
        name = f"mne-lsl-viewer-{request.node.name}-{len(created)}"
        source_id = str(uuid.uuid4())
        sinfo = StreamInfo(name, "eeg", n_channels, sfreq, dtype, source_id)
        n_data = n_channels - n_stim
        if ch_names is None:
            ch_names = [f"ch{k}" for k in range(n_data)]
            ch_names += [f"STI{k}" for k in range(n_stim)]
        sinfo.set_channel_names(ch_names)
        sinfo.set_channel_types(ch_types or ["eeg"] * n_data + ["stim"] * n_stim)
        sinfo.set_channel_units(ch_units or [units] * n_data + ["none"] * n_stim)
        outlet = StreamOutlet(sinfo)
        stream = StreamLSL(
            bufsize,
            name=name,
            stype="eeg",
            source_id=source_id,
        ).connect(acquisition_delay=_ACQUISITION_DELAY, timeout=10.0)
        created.append((stream, outlet))

        def _push(n_samples: int = 50, stim_at: int | None = None) -> None:
            """Push a synthesized block and block until it reached the buffer."""
            rng = np.random.default_rng(101)
            # an irregular stream declares 'sfreq == 0': the sample index is the time.
            times = np.arange(n_samples) / sfreq if sfreq else np.arange(n_samples)
            data = np.empty((n_samples, n_channels), dtype=np.float64)
            for k in range(n_data):
                data[:, k] = np.sin(
                    2 * np.pi * (5 + k) * times
                ) + 0.05 * rng.standard_normal(n_samples)
            data[:, n_data:] = 0.0
            if stim_at is not None:
                data[stim_at, n_data:] = 3.0
            # scaled before the cast so that an integer stream carries a real waveform
            # rather than a column of zeros and ones.
            if not np.issubdtype(np.dtype(dtype), np.floating):
                data *= 1000.0
            outlet.push_chunk(data.astype(dtype))
            deadline = time.monotonic() + _PUSH_DEADLINE
            while time.monotonic() < deadline:
                if stream.n_new_samples > 0:
                    return
                time.sleep(0.01)
            pytest.fail(f"The chunk pushed to {name} never reached the buffer.")

        return stream, _push

    yield _start
    for stream, outlet in reversed(created):
        if stream.connected:  # a test may have disconnected it on purpose
            stream.disconnect()
        outlet._del()
    created.clear()


@pytest.fixture
def default_stream(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
) -> tuple[StreamLSL, Callable[..., None]]:
    """Return the default 8-channel stream and its push callable."""
    return lsl_stream()


def _player_target(
    fname: Path,
    chunk_size: int,
    name: str,
    source_id: str,
    picks: list[str] | None,
    status: object,
) -> None:
    """Run one player until its status flag is cleared.

    A module-level function, because the default start method on macOS is ``spawn`` and
    a nested target cannot be pickled. 'PlayerLSL' and the file are both loaded in the
    child for the same reason.
    """
    from mne.io import read_raw_fif

    from mne_lsl.player import PlayerLSL

    raw = read_raw_fif(fname, preload=True)
    if picks is not None:
        raw.pick(picks)
    player = PlayerLSL(raw, chunk_size=chunk_size, name=name, source_id=source_id)
    player.start()
    status.value = 1
    while status.value:
        time.sleep(0.05)
    player.stop()


class _PlayerHandle:
    """Handle on a player subprocess, restartable under one identity.

    Parameters
    ----------
    fname : Path
        File the player streams.
    name : str
        LSL name of the published stream.
    source_id : str
        LSL source ID of the published stream, a uuid4 so that concurrent jobs sharing
        the link cannot collide.
    manager : multiprocessing.Manager
        Manager owning the status value, created by the fixture rather than at import
        time.

    Notes
    -----
    ``start`` returns only once the child has published, which is the handshake a poll
    on discovery would otherwise have to do. ``kill`` is what an outage looks like: the
    process goes away without stopping its outlet, so the inlet of a consumer raises
    rather than reading an empty stream forever.

    There is deliberately no clean ``stop``: every test which needs a source to go away
    needs it to go away *without* closing its outlet, which is the only case liblsl
    reports at all, and a clean shutdown is already covered without a subprocess.
    """

    _CHUNK_SIZE = 200
    _START_DEADLINE = 60.0

    def __init__(self, fname: Path, name: str, source_id: str, manager: object) -> None:
        self._fname = fname
        self.name = name
        self.source_id = source_id
        self._manager = manager
        self._status = None
        self._process = None

    def start(self, picks: list[str] | None = None) -> None:
        """Start a player, optionally over a subset of the channels."""
        assert self._process is None  # one process per handle at a time
        self._status = self._manager.Value("i", 0)
        self._process = mp.Process(
            target=_player_target,
            args=(
                self._fname,
                self._CHUNK_SIZE,
                self.name,
                self.source_id,
                picks,
                self._status,
            ),
        )
        self._process.start()
        deadline = time.monotonic() + self._START_DEADLINE
        while self._status.value != 1:
            # A child which died is reported at once and with its exit code: waiting the
            # full deadline out turns a bad 'picks' into a minute of silence.
            if not self._process.is_alive():
                code = self._process.exitcode
                self._process = None
                pytest.fail(f"The player {self.name} exited with code {code}.")
            if deadline < time.monotonic():
                self.kill()
                pytest.fail(f"The player {self.name} never started.")
            time.sleep(0.01)

    def kill(self) -> None:
        """Kill the player without letting it close its outlet: an outage."""
        if self._process is None:
            return
        self._process.kill()
        self._process.join(timeout=5)
        self._process = None


@pytest.fixture
def player(
    request: pytest.FixtureRequest, fname: Path
) -> Generator[Callable[..., _PlayerHandle]]:
    """Yield a factory creating restartable player subprocesses.

    The only fixture of this package which starts a subprocess, and deliberately so: an
    outlet answers every assertion which needs a stream to *exist*, while a player is
    needed only where a source has to genuinely go away and come back. Every handle is
    killed at teardown, whatever the assertions did.
    """
    manager = mp.Manager()
    handles: list[_PlayerHandle] = []

    def _make(picks: list[str] | None = None) -> _PlayerHandle:
        """Create one started player under a fresh identity."""
        name = f"mne-lsl-viewer-{request.node.name}-{len(handles)}"
        handle = _PlayerHandle(fname, name, str(uuid.uuid4()), manager)
        handles.append(handle)
        handle.start(picks)
        return handle

    yield _make
    for handle in reversed(handles):
        handle.kill()
    handles.clear()
    manager.shutdown()


@pytest.fixture
def stream(default_stream: tuple[StreamLSL, Callable[..., None]]) -> StreamLSL:
    """Return the default 8-channel stream, without any sample pushed yet."""
    return default_stream[0]


@pytest.fixture
def descriptor() -> Callable[..., StreamDescriptor]:
    """Return a factory building a descriptor out of plain values.

    No outlet is created, and the default 'source_id' is a uuid4, thus the identity is
    never on the network: this is the fixture for the logic which only moves descriptors
    around -- the identity dataclasses, the generation checks of the workers, the
    launcher table, the failure path of a connection -- while 'outlets' is the one to
    use when the stream has to actually answer.
    """
    # Nested for the same reason as in 'app': this conftest is imported even on a host
    # with no Qt binding, where 'mne_lsl.viewer' cannot be imported at all.
    from mne_lsl.viewer.backend import StreamDescriptor, StreamIdentity

    def _make(
        name: str = "mne-lsl-viewer-absent",
        stype: str = "eeg",
        source_id: str | None = None,
        n_channels: int = 4,
        sfreq: float = 100.0,
        hostname: str = "host-1",
        dtype: str = "float32",
        uid: str | None = None,
    ) -> StreamDescriptor:
        """Return one descriptor; every field has a default a test can override."""
        return StreamDescriptor(
            identity=StreamIdentity(
                name=name,
                stype=stype,
                source_id=str(uuid.uuid4()) if source_id is None else source_id,
            ),
            n_channels=n_channels,
            sfreq=sfreq,
            hostname=hostname,
            dtype=dtype,
            # a uuid4 by default, as liblsl's own is per outlet *instance*: two
            # descriptors built by this factory must never share a probe-cache key.
            uid=str(uuid.uuid4()) if uid is None else uid,
        )

    return _make


@pytest.fixture
def outlets(
    request: pytest.FixtureRequest,
) -> Generator[Callable[..., StreamDescriptor], None, None]:
    """Yield a factory creating resolvable LSL outlets, destroyed at teardown.

    The outlets never push a sample: discovery, the channel probe and
    'StreamLSL.connect()' all need an outlet to exist and to answer, not to produce
    data, so a silent outlet covers every assertion which needs one without a player
    subprocess or a testing dataset.

    Each outlet is named after the requesting test and gets a uuid4 'source_id', so
    concurrent jobs sharing the link can never collide, and the factory returns only
    once 'resolve_descriptors' actually sees the identity -- the discovery equivalent of
    the player fixture's status handshake.
    """
    # Nested for the same reason as in 'app', see the comment there.
    from mne_lsl.viewer.backend import resolve_descriptors

    created: list[tuple[StreamOutlet, StreamInfo]] = []

    def _start(
        n_channels: int = 4,
        sfreq: float = 100.0,
        stype: str = "eeg",
        ch_names: list[str] | None = None,
        name: str | None = None,
        source_id: str | None = None,
    ) -> StreamDescriptor:
        """Create one outlet and return its descriptor, once it is resolvable.

        'ch_names' defaults to generated names; an **empty** list publishes no channel
        description at all, which is the degenerate case a probe must still handle.
        """
        name = name if name is not None else f"mne-lsl-viewer-{request.node.name}"
        source_id = source_id if source_id is not None else str(uuid.uuid4())
        sinfo = StreamInfo(name, stype, n_channels, sfreq, "float32", source_id)
        if ch_names is None:
            ch_names = [f"ch{k}" for k in range(n_channels)]
        if len(ch_names) != 0:
            sinfo.set_channel_names(ch_names)
        outlet = StreamOutlet(sinfo)
        created.append((outlet, sinfo))
        identity = (name, stype, source_id)
        deadline = time.monotonic() + _RESOLVE_DEADLINE
        while time.monotonic() < deadline:
            for descriptor in resolve_descriptors(1.0):
                if descriptor.identity.as_tuple() == identity:
                    return descriptor
        pytest.fail(f"The outlet {identity} never became resolvable.")

    yield _start
    # teardown in the fixture rather than in the test bodies, so a failing assertion
    # still destroys the outlets instead of leaving them on the network.
    for outlet, _ in reversed(created):
        outlet._del()
    created.clear()


@pytest.fixture
def controller(app: QApplication) -> Generator[ThemeController, None, None]:
    """Yield the module-singleton ThemeController, restoring its state afterwards.

    The name refers to the *theme* controller, not to the 'controller/' subpackage.

    The singleton is instantiated at import time and 'pytest-randomly' shuffles the test
    order, thus any test which installs it or flips its mode must put it back. Only the
    3 privates below are restored, deliberately not the application palette / style
    sheet / pyqtgraph configuration: no test may assert a *default* application look,
    which would be order-dependent by construction. '_following' is not reset either, as
    the connection genuinely persists for the process and resetting it would let the
    next 'install' add a duplicate connection.
    """
    # Nested for the same reason as in 'app': this conftest is imported even on a host
    # with no Qt binding, where 'mne_lsl.viewer' cannot be imported at all.
    from mne_lsl.viewer.theme import theme_controller

    # 'app' is requested, not used: an application must exist before anything is themed.
    state = (theme_controller._app, theme_controller._setting, theme_controller._mode)
    yield theme_controller
    (
        theme_controller._app,
        theme_controller._setting,
        theme_controller._mode,
    ) = state


@pytest.fixture
def flush_deletes(app: QApplication) -> Callable[..., None]:
    """Return a callable deleting Qt objects and running their C++ destruction.

    Returns
    -------
    flush : callable
        Called with the objects to delete, in the order they must be destroyed.

    Notes
    -----
    ``deleteLater`` posts a ``DeferredDelete`` event, which ``processEvents`` does *not*
    deliver outside a running event loop: without the explicit ``sendPostedEvents`` an
    object is only ever freed by refcounting and the C++ destruction path -- the one
    which surfaces a use-after-delete -- never runs at all. That is a three-line idiom
    which every fixture of this package was repeating, and which several of them got
    wrong by omitting the flush, so it lives here once.
    """
    # Nested for the same reason as in 'app': this conftest is imported even on a host
    # with no Qt binding, where qtpy may not be importable at module level.
    from qtpy.QtCore import QEvent

    def _flush(*objects: object) -> None:
        """Delete ``objects``, then deliver the deferred deletions."""
        for obj in objects:
            obj.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()

    return _flush


@pytest.fixture
def window(
    app: QApplication, flush_deletes: Callable[..., None]
) -> Generator[ViewerWindow, None, None]:
    """Yield a viewer window, closed and deleted afterwards.

    The 'close()' is what stops the two worker threads and tears every open document
    down, as nothing else does on a path which never enters an event loop.
    """
    # Nested for the same reason as in 'app', see the comment there; and this conftest
    # may not import qtpy at module level either.
    from mne_lsl.viewer._window import ViewerWindow

    built = ViewerWindow()
    built.resize(1200, 700)
    yield built
    built.close()
    flush_deletes(built)


@pytest.fixture
def module_scan() -> Callable[[ModuleType], tuple[set[str], set[str]]]:
    """Return a factory parsing a module's source into its imports and identifiers.

    The import rules of a viewer subpackage cannot be checked through 'sys.modules':
    importing 'mne_lsl.viewer.backend._config' necessarily imports
    'mne_lsl.viewer.__init__', which imports qtpy, and 'mne_lsl.__init__', which imports
    'mne_lsl.lsl'. The rule is a source-level one, thus it is checked statically, on the
    module's own source only.

    Identifiers come from the syntax tree and not from a text search, so that a
    docstring mentioning a forbidden name -- documentation, not a dependency -- does not
    trip the check.

    An 'ImportFrom' is recorded as the dotted path of every name it binds, not as its
    module alone, which is what catches 'from ... import lsl': its 'node.module' is
    'None', so the module path of that form carries no segment to check at all. The
    leading dots are stripped, so that '...lsl' and 'mne_lsl.lsl' are both caught by the
    same segment check. Attribute access is what makes the 'identifiers' set worth
    asserting on as well: 'import mne_lsl' followed by 'mne_lsl.lsl.resolve_streams()'
    imports nothing forbidden and reaches the forbidden module anyway.
    """

    def _scan(module: ModuleType) -> tuple[set[str], set[str]]:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imports: set[str] = set()
        identifiers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
                identifiers.update(alias.name.split(".")[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = (node.module or "").lstrip(".")
                imports.update(
                    f"{prefix}.{alias.name}" if prefix else alias.name
                    for alias in node.names
                )
                identifiers.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
        return imports, identifiers

    return _scan


@pytest.fixture
def finish_edit() -> Callable[[EditableReadout, str], None]:
    """Return a helper typing a value in a read-out editor and committing it.

    'EditableReadout' lives in 'widgets/' and is consumed by the display control bar,
    thus the helper for driving one belongs here rather than in either subdirectory.
    """

    def _finish(readout: EditableReadout, text: str) -> None:
        """Open the editor, type ``text`` and commit it, as the Enter key does."""
        readout.begin_edit()
        readout._edit.setText(text)
        readout._edit.editingFinished.emit()

    return _finish
