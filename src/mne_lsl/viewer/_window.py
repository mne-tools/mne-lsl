"""Main window: the application toolbar, the document area and the status bar."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import qtpy
from qtpy.QtCore import QByteArray, QRect, QSize, Qt, qVersion
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)

from .._version import __version__
from ..utils.logs import logger
from ._bootstrap import configure_docking, import_ads
from ._document import LIVE, StreamDocument
from ._launcher import EmptyStatePage, progress_text
from .backend import (
    STATE_LOADING,
    ConfigurationState,
    Connector,
    Discovery,
    Prober,
    StreamIdentity,
    ViewerConfig,
    channel_key,
    channels_reason,
    delete_configuration,
    derive_bufsize,
    evaluate_state,
    identity_text,
    list_configurations,
    missing_channels,
    release_stream,
    rename_configuration,
    save_configuration,
    stream_identity,
    wait_for_reconnects,
)
from .display import WINDOW_RANGE
from .theme import (
    _ADS_ICONS,
    _ADS_QSS,
    _ICON_PX,
    _MODES,
    follow_theme,
    icon,
    theme_controller,
)
from .widgets import AnimatedSegmentedControl

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from qtpy.QtGui import QAction, QCloseEvent, QShowEvent
    from qtpy.QtWidgets import QToolBar

    from ..stream import BaseStream
    from .backend import StreamDescriptor

ads = import_ads()

# Version stamped into the saved Qt-ADS layout as its 'UserVersion'. A mismatch makes
# 'restoreState' return False, which is a free version guard, and it is deliberately
# separate from the configuration schema version: adding a display field must not
# invalidate every saved layout. Bumped only when the dock-widget naming or structure
# changes.
LAYOUT_VERSION = 1

# Interface policy of the Save-as and Rename prompts. No character is rejected -- the
# slug function neutralizes them all -- but a 300-character name would carry into every
# card label and every unavailability reason.
_MAX_NAME_CHARS = 120

# Smallest intersection with a screen, in pixels, which still makes a restored window
# both visible and grabbable by its title bar.
_MIN_VISIBLE_W = 200
_MIN_VISIBLE_H = 100

# Title of the one dialog a failed configuration load shows.
_LOAD_FAILURE_TITLE = "Could not open the configuration"

# Characters kept of the source ID in the status bar before eliding; the full value
# stays in the tooltip. Character-based on purpose: a status-bar segment has no measured
# rectangle to elide against, and a character count is what a test can assert.
_SOURCE_ID_CHARS = 14
# Theme toggle segments, as (label, tooltip, user setting). The settings are the shared
# 'theme._MODES', in its order: the index lookup below relies on it, and a test pins it.
_THEME_SEGMENTS = (
    ("Auto", "Follow the operating system theme", "auto"),
    ("Light", "Force the light theme", "light"),
    ("Dark", "Force the dark theme", "dark"),
)


def _message(
    parent: QWidget,
    kind: str,
    title: str,
    text: str,
    detail: str = "",
) -> bool:
    """Show one modal message box rendering ``text`` literally, and return the answer.

    Parameters
    ----------
    parent : QWidget
        Parent of the dialog.
    kind : str
        One of ``'question'``, ``'warning'`` or ``'critical'``. A question offers Yes
        and No; the other two offer only Ok.
    title : str
        Window title, never user-supplied.
    text : str
        The message. May contain arbitrary user text.
    detail : str
        Optional text folded behind the dialog's details control.

    Returns
    -------
    accepted : bool
        ``True`` when a question was answered Yes, ``False`` otherwise. Always ``False``
        for a warning or a critical notice, which offer nothing to accept.

    Notes
    -----
    Every message this viewer shows carries a name it did not choose -- a configuration
    name typed by the user, or a stream name that arrived over the network -- and the
    static ``QMessageBox`` helpers render with ``AutoText``, which interprets anything
    that looks like markup. A name of ``<img src=...>`` would embed a local file in the
    dialog. Setting the format explicitly is the whole reason this function exists, so
    prefer it over the static helpers even for a message which looks safe today.

    Escaping the text instead would be wrong: with no markup left to detect, the box
    falls back to plain rendering and an ordinary name like ``EEG & EMG`` would be shown
    with its escape sequences visible.
    """
    icons = {
        "question": QMessageBox.Icon.Question,
        "warning": QMessageBox.Icon.Warning,
        "critical": QMessageBox.Icon.Critical,
    }
    box = QMessageBox(parent)
    box.setIcon(icons[kind])
    box.setWindowTitle(title)
    box.setText(text)
    box.setTextFormat(Qt.TextFormat.PlainText)
    if detail:
        box.setDetailedText(detail)
    if kind == "question":
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
    else:
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
    return box.exec() == QMessageBox.StandardButton.Yes


def _clean_name(text: str) -> str:
    """Return a typed configuration name, stripped and capped.

    Parameters
    ----------
    text : str
        The raw text of a Save-as or Rename prompt.

    Returns
    -------
    name : str
        The name to store, empty when nothing usable was typed.

    Notes
    -----
    Stripped **twice**, once around the cap: cutting at the limit can land on whitespace
    and reintroduce a trailing space. That matters beyond tidiness because the
    persistence layer strips again, so a name normalized only once is remembered here in
    a form no file on disk carries -- after which a plain Save stops recognizing its own
    source and silently becomes a Save-as, writing a new file on every press.
    """
    return text.strip()[:_MAX_NAME_CHARS].strip()


def _elide(text: str) -> str:
    """Return ``text`` cut to ``_SOURCE_ID_CHARS`` characters, with an ellipsis."""
    if len(text) <= _SOURCE_ID_CHARS:
        return text
    return f"{text[:_SOURCE_ID_CHARS]}…"


def _slot_index(name: str) -> int | None:
    """Return the counter value ``name`` was minted from, or ``None``.

    Parameters
    ----------
    name : str
        Object name of a dock widget, saved or freshly assigned.

    Returns
    -------
    index : int | None
        The integer of a ``stream-<int>`` name, ``None`` for any other shape.
    """
    prefix = "stream-"
    # 'isdecimal' and not 'isdigit': the latter accepts characters 'int' refuses, so
    # 'stream-²' would pass the guard and raise on conversion -- turning a cosmetic slot
    # name in a hand-edited file into a failed load rather than the documented fallback.
    if not name.startswith(prefix) or not name[len(prefix) :].isdecimal():
        return None
    return int(name[len(prefix) :])


def _slot_for(block: Mapping, taken: set[str]) -> str | None:
    """Return the saved slot name of ``block`` when it is free, else ``None``.

    Parameters
    ----------
    block : dict
        One saved per-document presentation block.
    taken : set of str
        Object names which are already registered in the dock manager, updated in place
        with the name this call hands out.

    Returns
    -------
    slot : str | None
        The saved object name, or ``None`` for 'mint a fresh one'.

    Notes
    -----
    Reusing the saved name is what makes the saved layout XML join the documents at all
    -- the XML stores object names and nothing else. A name which is somehow still taken
    after the purge falls back to a fresh one, which loses the placement and degrades to
    plain tabs: **never** a duplicate, which overwrites the manager's map entry and
    corrupts every later save of the layout.
    """
    slot = block.get("slot")
    if not isinstance(slot, str) or not slot or slot in taken:
        return None
    taken.add(slot)
    return slot


def _presentation_blocks(cfg: ViewerConfig) -> dict[tuple[str, str, str], Mapping]:
    """Return the per-document presentation blocks of ``cfg``, keyed by identity.

    Parameters
    ----------
    cfg : ViewerConfig
        The configuration being loaded.

    Returns
    -------
    blocks : dict
        One block per identity the payload names. A payload of the wrong shape, or an
        entry carrying no usable identity, yields no entry rather than raising: the
        presentation payload is opaque to the persistence layer, thus a hand-edited file
        reaches this with any shape at all.
    """
    blocks: dict[tuple[str, str, str], Mapping] = {}
    streams = cfg.presentation.get("streams")
    if not isinstance(streams, (list, tuple)):
        return blocks
    for block in streams:
        if not isinstance(block, Mapping):
            continue
        identity = block.get("identity")
        if isinstance(identity, (list, tuple)) and len(identity) == 3:
            blocks[tuple(identity)] = block
    return blocks


def _restore_geometry(window: QMainWindow, window_state: object) -> None:
    """Move and resize ``window`` onto a saved rectangle, defensively.

    Parameters
    ----------
    window : QMainWindow
        The window to place.
    window_state : dict
        The saved ``x``, ``y``, ``width``, ``height`` and ``maximized`` values. An
        unusable value skips the whole step, since half a rectangle is worse than none.

    Notes
    -----
    A module-level function and not a method, so that the vanished-screen and
    too-small-intersection branches can be exercised against fabricated screen
    rectangles without performing a configuration load.

    An explicit rectangle rather than ``restoreGeometry``: the saved file stays
    inspectable and this defensive rule stays auditable, which the internal heuristics
    of ``restoreGeometry`` are not. A rectangle whose intersection with every available
    screen is too small to see or to grab is clamped **silently** onto the primary
    screen -- a monitor which was unplugged is not an error the user has to acknowledge.

    The maximized state is set through ``setWindowState`` and not ``showMaximized``,
    which would additionally *show* a window the caller has not shown yet.
    """
    if not isinstance(window_state, Mapping):
        return
    values: list[int] = []
    for key in ("x", "y", "width", "height"):
        value = window_state.get(key)
        # a bool passes 'isinstance(True, int)' and cannot be a coordinate.
        if isinstance(value, bool) or not isinstance(value, int):
            logger.warning(
                "Ignoring the saved window geometry: %r is not an integer %r.",
                value,
                key,
            )
            return
        values.append(value)
    x, y, width, height = values
    if width <= 0 or height <= 0:
        logger.warning(
            "Ignoring the saved window geometry: %dx%d is not a usable size.",
            width,
            height,
        )
        return
    primary = QGuiApplication.primaryScreen()
    if primary is None:  # no screen at all: nothing to place the window against
        return
    target = QRect(x, y, width, height)
    for screen in QGuiApplication.screens():
        visible = screen.availableGeometry().intersected(target)
        if visible.width() >= _MIN_VISIBLE_W and visible.height() >= _MIN_VISIBLE_H:
            window.setGeometry(target)
            break
    else:
        available = primary.availableGeometry()
        window.resize(min(width, available.width()), min(height, available.height()))
        window.move(available.center() - window.rect().center())
    if window_state.get("maximized") is True:
        window.setWindowState(window.windowState() | Qt.WindowState.WindowMaximized)


class _LoadAttempt:
    """The streams one configuration load has connected so far.

    Parameters
    ----------
    cfg : ViewerConfig
        The configuration being opened.
    descriptors : sequence of StreamDescriptor
        Descriptors of the streams which are being connected, event sources excluded.

    Notes
    -----
    :attr:`pending` is the **only** record of what is in flight: nothing counts it,
    nothing mirrors it and nothing clamps it. A second copy of in-flight state is what
    stranded an identity for the life of the process in the incremental open path.
    """

    def __init__(
        self, cfg: ViewerConfig, descriptors: Sequence[StreamDescriptor]
    ) -> None:
        self.cfg = cfg
        self.pending: dict[StreamIdentity, StreamDescriptor] = {
            descriptor.identity: descriptor for descriptor in descriptors
        }
        self.streams: dict[StreamIdentity, BaseStream] = {}
        self.errors: list[str] = []


class ViewerWindow(QMainWindow):
    """The one window owning the complete experience.

    The central widget stacks the landing page and the document host, whose central
    widget is a Qt-ADS ``CDockManager``. Documents are ``CDockWidget`` instances: they
    tab together with IDE-style top tabs and split the space in two when dragged to a
    side.
    The shared status bar follows the focused document through
    ``focusedDockWidgetChanged``.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the window, its dock manager, toolbar and status bar."""
        super().__init__(parent)
        self.setWindowTitle("MNE-LSL viewer")
        self._documents: list[StreamDocument] = []
        self._active: StreamDocument | None = None
        self._closed = False
        self._dock_area = None
        # Monotonic, and never 'len(self._documents)': a closed document keeps its entry
        # in the manager's dock-widget map for the life of the process, thus a reused
        # name loses a map entry and writes an ambiguous saved layout.
        self._next_index = 0
        # The batch in flight, and the message it wrote. Descriptors and not identities:
        # the connector supersedes a batch rather than queueing behind it, so a new one
        # has to carry the still-connecting descriptors with it.
        self._connecting: dict[StreamIdentity, StreamDescriptor] = {}
        self._batch_message = ""
        self._toolbar_icons: list[tuple[QAction, str]] = []

        # -- the saved configurations, all of it GUI-thread state ----------------------
        self._configs: list[ViewerConfig] = []
        # The identities the last discovery pass found, and 'None' -- never an empty set
        # -- until the first one has landed: before then, 'no matching stream' is a
        # claim the viewer cannot make.
        self._present: frozenset[tuple[str, str, str]] | None = None
        self._uids: dict[tuple[str, str, str], str] = {}
        self._descriptors: dict[tuple[str, str, str], StreamDescriptor] = {}
        # THE probe cache, keyed '((name, stype, source_id), uid)' and holding either
        # the probed channel names or the probe's error message. Read and written on the
        # GUI thread only, thus it needs no lock, and never pruned: a uid is one short
        # string and the bound is the number of distinct outlet instances seen this
        # session. The uid is what makes it sound rather than a heuristic -- it changes
        # whenever an outlet is re-instantiated, i.e. whenever the channel set may have
        # changed.
        self._probes: dict[tuple[tuple[str, str, str], str], list[str]] = {}
        # Successes only, above. A failure is not a property of the channel set, so it
        # is kept here and dropped whenever a new discovery pass lands: cached under the
        # key above it would make Refresh -- the only recovery gesture there is --
        # unable to re-probe for the life of the outlet instance, since a uid survives a
        # timeout.
        self._probe_errors: dict[tuple[str, str, str], str] = {}
        # What has been asked for and not yet answered. Without it an identity already
        # being probed is submitted again, and 'Prober.probe' supersedes rather than
        # queues, so a Refresh faster than the probe returns discards the batch in
        # flight and re-opens an inlet against a live device while the card never
        # settles.
        self._probing: set[tuple[tuple[str, str, str], str]] = set()
        # The configuration this workspace came from, so a plain Save replaces it.
        self._source: str | None = None
        self._loading: _LoadAttempt | None = None

        # Connected before anything can call 'open()': a 'Connector' hands stream
        # ownership over with its signal, thus a batch started before the receiver
        # exists would drop a live stream and leak its inlet.
        self._discovery = Discovery(parent=self)
        self._connector = Connector(parent=self)
        # A second connector, load-only. One instance with a mode flag would put the
        # branch inside a slot which already carries three responsibilities, and every
        # defect found in that slot so far came from state living in two places. The
        # cost is one more idle thread, started lazily and stopped in 'closeEvent'.
        self._loader = Connector(parent=self)
        self._prober = Prober(parent=self)
        self._discovery.progress.connect(self._on_progress)
        self._discovery.streams_found.connect(self._on_streams_found)
        self._connector.connected.connect(self._on_connected)
        self._connector.failed.connect(self._on_failed)
        self._loader.connected.connect(self._on_load_connected)
        self._loader.failed.connect(self._on_load_failed)
        self._prober.resolved.connect(self._on_probed)
        self._prober.failed.connect(self._on_probe_failed)

        self._landing = EmptyStatePage(self)
        self._landing.selection_changed.connect(self._update_open_action)
        self._landing.open_requested.connect(self.open_streams)
        self._landing.open_configuration_requested.connect(self.open_configuration)
        self._landing.rename_configuration_requested.connect(self._rename_configuration)
        self._landing.delete_configuration_requested.connect(self._delete_configuration)

        # The docking flags are static and consumed by the 'CDockManager' constructor:
        # setting one afterwards crashes the process on the next 'addDockWidget'.
        configure_docking()
        self._dock_host = QMainWindow()
        self._manager = ads.CDockManager(self._dock_host)
        self._manager.focusedDockWidgetChanged.connect(self._on_focus_changed)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._landing)
        self._stack.addWidget(self._dock_host)
        self.setCentralWidget(self._stack)

        self._build_toolbar()
        self._build_status_bar()
        self._apply_ads_theme()
        self._following_theme = False
        follow_theme(self, self._on_theme_changed, True)
        self._update_open_action()
        self._update_save_actions()
        self._update_status_bar()
        # No discovery pass here: 'Viewer.show()' starts the first one. A pass costs
        # about a second and the 'stop()' of a close waits for the one in flight, which
        # every test of this window would otherwise pay.

    # -- construction ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        """Build the application toolbar."""
        bar = self.addToolBar("Main")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bar.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self._act_refresh = self._add_action(
            bar, "mdi6.refresh", "Refresh", "Look for the streams on the network again"
        )
        self._act_refresh.triggered.connect(self.refresh)
        self._act_open = self._add_action(
            bar,
            "mdi6.play-box-outline",
            "Open selected",
            "Open one document per selected stream",
        )
        self._act_open.triggered.connect(self._open_selected)
        self._act_save = self._add_action(
            bar,
            "mdi6.content-save-outline",
            "Save",
            "Save this workspace, replacing the configuration it came from",
        )
        self._act_save.triggered.connect(self.save_configuration)
        self._act_save_as = self._add_action(
            bar,
            "mdi6.content-save-plus-outline",
            "Save as…",
            "Save this workspace under a new name",
        )
        self._act_save_as.triggered.connect(self.save_configuration_as)
        bar.addSeparator()
        self._add_action(
            bar, "mdi6.help-circle-outline", "About", "Versions and Qt binding in use"
        ).triggered.connect(self._about)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        bar.addWidget(self._build_theme_toggle())

    def _add_action(self, bar: QToolBar, name: str, text: str, tooltip: str) -> QAction:
        """Add one toolbar action and register its icon for the theme flip.

        Parameters
        ----------
        bar : QToolBar
            The toolbar the action is added to.
        name : str
            QtAwesome icon name.
        text : str
            Action text.
        tooltip : str
            Action tooltip.

        Returns
        -------
        action : QAction
            The action which was added.
        """
        action = bar.addAction(icon(name), text)
        action.setToolTip(tooltip)
        # a 'QIcon' bakes its color at creation, thus a flip replays this table.
        self._toolbar_icons.append((action, name))
        return action

    def _build_theme_toggle(self) -> QWidget:
        """Build the Auto/Light/Dark control, synced to the current user setting.

        Notes
        -----
        The lookup needs no guard: the segments carry the theme vocabulary itself, thus
        the current setting is one of them by construction -- there is no fourth setting
        the resolver would accept and the toggle could not show.
        """
        toggle = AnimatedSegmentedControl(_THEME_SEGMENTS)
        toggle.set_index(_MODES.index(theme_controller.setting), emit=False)
        toggle.changed.connect(theme_controller.set_mode)
        # Every color of that widget is a palette role, thus it needs no retinting.
        return toggle

    def _build_status_bar(self) -> None:
        """Build the three permanent status-bar segments, left to right."""
        self._sb_state = QLabel()
        self._sb_identity = QLabel()
        self._sb_meta = QLabel()
        for label in (self._sb_state, self._sb_identity, self._sb_meta):
            label.setContentsMargins(6, 0, 6, 0)
            self.statusBar().addPermanentWidget(label)
        # 'showMessage' stays free for the transients: progress and the errors.

    # -- state -------------------------------------------------------------------------
    @property
    def documents(self) -> tuple[StreamDocument, ...]:
        """The open documents, in the order they were opened."""
        return tuple(self._documents)

    @property
    def active_document(self) -> StreamDocument | None:
        """The document the status bar describes, or ``None`` while none is open."""
        return self._active

    @property
    def closed(self) -> bool:
        """Whether the window has been closed, and is therefore spent.

        Notes
        -----
        A closed window is not reusable: :meth:`closeEvent` is the whole teardown, so it
        has stopped both workers and torn every document down. It is exposed because
        nothing else can tell -- ``QWidget`` emits ``destroyed`` on the C++ deletion
        only, and :class:`~mne_lsl.viewer.Viewer` keeps a reference to its window
        forever, so a closed one stays alive and answers every call.
        """
        return self._closed

    # -- discovery ---------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the configurations and start one discovery pass.

        Notes
        -----
        A no-op once the window is closed, and the guard is not decoration:
        ``Discovery.refresh`` starts its worker thread on demand, and after
        :meth:`closeEvent` nothing is left to stop it again.

        The configuration directory is re-listed here as well, which is the replacement
        for a filesystem watcher: one Refresh click picks up a file which appeared, was
        removed or was corrupted out of band. Measured at 7 ms for a hundred realistic
        files, thus it stays on the GUI thread.

        ponytail: GUI-thread listing, measured at two orders of magnitude under a frame
        for a hundred files. The upgrade is to move the listing onto the discovery
        worker -- it is Qt-free and LSL-free, so the move is small -- if a directory on
        a network mount, or a user with a thousand configurations, is ever measured as a
        stall.
        """
        if self._closed:
            return
        self.statusBar().clearMessage()
        self.reload_configurations()
        self._discovery.refresh()

    def _on_progress(self, tag: str) -> None:
        """Reflect a discovery state tag on the landing page and in the status bar."""
        self._landing.set_progress(tag)
        self.statusBar().showMessage(progress_text(tag), 4000)

    def _on_streams_found(self, descriptors: object) -> None:
        """Publish the descriptors of a finished pass on the landing page.

        Notes
        -----
        The Open action is *not* re-evaluated here: publishing the descriptors rebuilds
        the table, which the page reports through ``selection_changed`` -- the one path,
        connected in the constructor. A second call from here would run the same
        evaluation twice per discovery pass and, worse, keep passing if the page ever
        stopped reporting a rebuild at all.

        The three availability dictionaries are *replaced* rather than merged: they
        describe one pass, and an identity the network no longer reports must stop being
        loadable. The channel cache is the one thing which survives, because it is keyed
        on the outlet instance and therefore cannot go stale.

        Probe *failures* are dropped here, which is what makes Refresh a real recovery
        gesture: a timeout against a busy host says nothing about the stream, so a new
        pass must be allowed to ask again rather than inherit the verdict.
        """
        self._landing.set_streams(descriptors)
        self._probe_errors.clear()
        self._present = frozenset(
            descriptor.identity.as_tuple() for descriptor in descriptors
        )
        self._uids = {
            descriptor.identity.as_tuple(): descriptor.uid for descriptor in descriptors
        }
        self._descriptors = {
            descriptor.identity.as_tuple(): descriptor for descriptor in descriptors
        }
        self._publish_configurations()
        self._submit_probes()

    # -- the availability of the saved configurations ----------------------------------
    def reload_configurations(self) -> None:
        """Re-read the configuration directory and republish every card."""
        self._configs = list_configurations()
        self._publish_configurations()
        self._submit_probes()

    def _publish_configurations(self) -> None:
        """Evaluate every configuration and hand the rendered rows to the landing page.

        Notes
        -----
        The probe cache is projected onto the identities of the *current* pass before
        the evaluation, which is what lets the availability check stay pure and
        uid-free: the cache is keyed on the outlet instance, while the check compares
        channel sets.

        A Refresh never resets a card. The page-level indicator already carries the
        'checking for streams' progress, and resetting would flicker an available card
        to unavailable and back on every click.
        """
        probed: dict[tuple[str, str, str], list[str] | str] = {
            identity: self._probes[(identity, uid)]
            for identity, uid in self._uids.items()
            if (identity, uid) in self._probes
        }
        # A failure only speaks for an identity this pass has no channel list for; a
        # success always wins, so a stream which answered once does not revert to
        # 'could not be reached' because a later probe of it timed out.
        for identity, message in self._probe_errors.items():
            probed.setdefault(identity, message)
        loading = None if self._loading is None else self._loading.cfg.name
        rows = [
            ConfigurationState(cfg.name, STATE_LOADING, "Connecting…", 0)
            if cfg.name == loading
            else evaluate_state(cfg, self._present, probed)
            for cfg in self._configs
        ]
        # Every transition of '_loading' funnels through here, so the two gestures a
        # load must lock out are derived in one place rather than toggled by hand at
        # each of the three transitions -- a missed one leaves Refresh dead for the
        # session.
        self._landing.set_configurations(rows, loading=loading is not None)
        self._act_refresh.setEnabled(self._loading is None)
        self._update_open_action()

    def _submit_probes(self) -> None:
        """Probe the identity-matching regular streams whose channel set is not cached.

        Notes
        -----
        Built from the intersection of what the configurations require and what the last
        pass actually found, never from the configurations alone: a probe fired at an
        identity which is not on the network burns the full resolution timeout, once per
        Refresh, for a card which already reads 'no matching stream'.

        Zero identity-matching configurations therefore means zero probes, which is the
        common first-launch case, and an ``(identity, uid)`` already in the cache is
        skipped -- that is what keeps a Refresh over an unchanged network free, and what
        stops the viewer from opening and destroying an inlet against a live acquisition
        device on every click.
        """
        wanted: dict[tuple[str, str, str], StreamDescriptor] = {}
        for cfg in self._configs:
            for identity in cfg.streams:
                if channel_key(identity) not in cfg.channels:
                    continue  # an event source is matched on its identity only
                descriptor = self._descriptors.get(identity)
                if descriptor is None:
                    continue
                key = (identity, descriptor.uid)
                if key in self._probes or key in self._probing:
                    continue
                wanted[identity] = descriptor
        if wanted:
            # Assigned, not unioned: this is one record of what was last asked for, and
            # 'probe' supersedes, so anything outstanding from an earlier batch has just
            # been cancelled. Accumulating would make it a second, diverging copy of the
            # prober's own state.
            self._probing = {(i, d.uid) for i, d in wanted.items()}
            self._prober.probe(list(wanted.values()))

    def _on_probed(self, descriptor: object, names: object) -> None:
        """Cache the channel names of a probed stream and republish the cards.

        Notes
        -----
        The fan-out is automatic: one result is stored per stream, so every
        configuration naming that identity settles on it. Storing it per configuration
        instead would mean one probe of the same stream per configuration.
        """
        identity = descriptor.identity.as_tuple()
        self._probing.discard((identity, descriptor.uid))
        self._probe_errors.pop(identity, None)
        self._probes[(identity, descriptor.uid)] = list(names)
        self._publish_configurations()

    def _on_probe_failed(self, descriptor: object, message: str) -> None:
        """Cache the failure of a probe and republish the cards.

        Notes
        -----
        The message is stored, not an empty name list: the two unavailability reasons
        the design separates are 'the channels no longer match' and 'the stream could
        not be reached', and an empty list would report the first for the second.

        It is kept apart from the channel cache and keyed on the identity alone, because
        a failure says nothing about the channel set. Cached beside a success it would
        survive every Refresh for as long as the outlet instance lives -- a probe
        timeout on a loaded host would leave the card reading 'could not be reached'
        with no way back, and Refresh is the only recovery gesture the interface offers.
        """
        identity = descriptor.identity.as_tuple()
        self._probing.discard((identity, descriptor.uid))
        self._probe_errors[identity] = message
        self._publish_configurations()

    # -- opening documents -------------------------------------------------------------
    def open_streams(self, descriptors: Sequence[StreamDescriptor]) -> None:
        """Connect to ``descriptors`` in the background and open one document each.

        A descriptor whose identity is already open raises the existing document instead
        of connecting twice; a failed connection is reported without closing the
        documents which are already open. A connection which is still in flight is
        carried into the new batch rather than being dropped.

        Refused outright while a configuration is being opened. A load builds its
        documents into locals and publishes them only on success, so a document opened
        underneath it is registered but unnamed in the saved layout: the restore then
        closes it -- through ``viewToggled``, not ``closed``, so nothing tears it down
        -- leaving it out of the workspace but inside ``documents``, with its render
        clock running and, worse, inside the next Save. Gating the toolbar action alone
        is not enough, because the stream table's double click arrives here directly.

        Parameters
        ----------
        descriptors : sequence of StreamDescriptor
            Descriptors of the regular streams to open.
        """
        if self._closed or self._loading is not None:
            return
        wanted: dict[StreamIdentity, StreamDescriptor] = {}
        for descriptor in descriptors:
            identity = descriptor.identity
            existing = self._document_for(identity)
            if existing is not None:
                self._raise(existing)
                continue
            if descriptor.sfreq == 0:
                # an event source owns no document: it has no continuous signal to draw,
                # and 'StreamLSL' refuses a buffer in seconds for one.
                logger.warning(
                    "The stream %s is an event source and cannot be opened as a "
                    "document.",
                    identity.as_tuple(),
                )
                continue
            wanted[identity] = descriptor
        if not wanted:
            return
        # 'Connector.open' bumps its generation counter, thus it *supersedes* the batch
        # in flight instead of queueing behind it: the descriptors which have not come
        # back yet are merged into the new batch, or the streams the user asked for
        # first are dropped silently and their identities stay in flight for the life of
        # the process. Assigned, never accumulated, for the same reason.
        merged = self._connecting | wanted
        if merged == self._connecting:
            return  # a double click: the batch in flight already covers every identity
        self._connecting = merged
        self._batch_message = f"Connecting to {len(merged)} stream(s)…"
        self.statusBar().showMessage(self._batch_message)
        # Derived from the widest window the display can select, never from the current
        # one: a window wider than the buffer draws over part of the time axis for the
        # rest of the session, silently.
        self._connector.open(list(merged.values()), derive_bufsize(WINDOW_RANGE[1]))

    def adopt_stream(self, stream: BaseStream) -> StreamDocument:
        """Open a document for an already connected stream, without owning it.

        This is the :meth:`~mne_lsl.stream.BaseStream.plot` path: the stream is
        borrowed, thus closing the document never disconnects it, and the viewer never
        reconnects it on its own either -- a reconnection replaces the inlet and the
        buffer, which drops the filters, the callbacks and the acquisition delay its
        owner set. A borrowed document whose source goes away therefore offers Retry and
        waits for it. The window provides its dock manager to the document it builds.

        Parameters
        ----------
        stream : BaseStream
            A connected stream owned by the caller.

        Returns
        -------
        document : StreamDocument
            The document which was opened.

        Raises
        ------
        RuntimeError
            If the stream is not connected.
        TypeError
            If the stream is not an LSL stream, i.e. if it carries no identity.
        ValueError
            If the stream is irregularly sampled, i.e. if it declares ``sfreq == 0``.
        """
        identity = stream_identity(stream)
        existing = self._document_for(identity)
        if existing is not None:
            self._raise(existing)
            return existing
        doc = StreamDocument(self._manager, stream, identity, owns_stream=False)
        self._register(doc)
        return doc

    def _on_connected(self, descriptor: object, stream: object) -> None:
        """Open a document for a stream which just connected.

        Notes
        -----
        Both exits release the stream. The connector transfers its ownership with this
        signal, so every path which does not hand it to a document has to disconnect it
        or it leaks a live inlet and its acquisition thread for the life of the process.
        The construction is guarded for the same reason: it refuses a stream declaring
        no sampling rate -- reachable when a source was re-provisioned as an event
        stream between the discovery pass and this connection -- and it reads the
        stream, which raises if the source went away again in the meantime.
        """
        identity = descriptor.identity
        self._connecting.pop(identity, None)
        self._batch_step()
        if self._document_for(identity) is not None:
            logger.warning(
                "A document is already open for the stream %s; releasing the stream "
                "which just connected.",
                identity.as_tuple(),
            )
            release_stream(stream)
            return
        try:
            doc = StreamDocument(self._manager, stream, identity)
        except Exception as error:  # deliberately broad, see the note above
            logger.warning(
                "Could not open a document for the stream %s: %s",
                identity.as_tuple(),
                error,
            )
            release_stream(stream)
            self.statusBar().showMessage(f"Could not open {identity.name}: {error}")
            return
        self._register(doc)

    def _on_failed(self, descriptor: object, message: str) -> None:
        """Report a connection which failed, leaving the open documents untouched."""
        self._connecting.pop(descriptor.identity, None)
        self._batch_step()
        # No timeout and no dialog: one stable error surface, cleared by the next
        # refresh or open. The connector has already logged the exception.
        self.statusBar().showMessage(
            f"Could not open {descriptor.identity.name}: {message}"
        )

    def _batch_step(self) -> None:
        """Clear the batch's own message once every connection has come back.

        Notes
        -----
        The substitute for the ``finished`` signal the connector does not have: a batch
        is complete once nothing is left in flight.

        Only a message this batch itself wrote is cleared. Clearing unconditionally
        wipes the failure of one stream as soon as a sibling of the same batch succeeds,
        and the status bar is the only error surface there is.
        """
        if self._connecting:
            return
        if self.statusBar().currentMessage() == self._batch_message:
            self.statusBar().clearMessage()

    def _register(self, doc: StreamDocument) -> None:
        """Dock a document and publish it, i.e. the whole incremental open path.

        Parameters
        ----------
        doc : StreamDocument
            The document to register.

        Notes
        -----
        Split in two halves because a configuration load has to dock every document
        *before* it publishes any of them: nothing may reach ``self._documents`` until
        the whole workspace is known to have been built, so that a rollback has nothing
        to unwind.
        """
        self._dock(doc)
        self._publish(doc)

    def _dock(self, doc: StreamDocument, name: str | None = None) -> None:
        """Give a document its object name and add it to the dock manager.

        Parameters
        ----------
        doc : StreamDocument
            The document to dock.
        name : str | None
            Object name to assign, or ``None`` for the next one the counter grants.

        Notes
        -----
        The single ``addDockWidget`` call site, so that the object-name rule cannot come
        apart from it. Qt-ADS keys its dock-widget map, and thus its saved layout, on
        ``objectName()``, which the ``CDockWidget`` constructor sets to the tab title --
        and two streams may legitimately share a name. The counter is monotonic because
        a closed document keeps its map entry for the life of the process.

        Every assigned name is reserved in that counter, whether it came from the
        counter or from a saved layout: a duplicate object name overwrites the map entry
        and makes every subsequent ``saveState()`` name the same slot several times,
        which restores into as many phantom dock areas and is unrecoverable from the
        interface.

        The window owns the docking here and nothing else: the render clock is started
        by the document's own constructor, which is what owns the live/frozen state.
        """
        doc.setObjectName(name if name else f"stream-{self._next_index}")
        self._reserve_slot(doc.objectName())
        # 'None' *is* the two-argument form: the signature is 'addDockWidget(area,
        # dockwidget, DockAreaWidget=nullptr, Index=-1)', thus the first document needs
        # no branch of its own.
        self._manager.addDockWidget(
            ads.DockWidgetArea.CenterDockWidgetArea, doc, self._dock_area
        )
        self._dock_area = doc.dockAreaWidget()

    def _reserve_slot(self, name: str) -> None:
        """Keep the monotonic counter past ``name``, so it can never grant it again.

        Parameters
        ----------
        name : str
            Object name which was just assigned to a document.

        Notes
        -----
        A name of another shape reserves nothing and needs to: the counter only ever
        mints ``stream-<int>``, so it cannot collide with one.
        """
        index = _slot_index(name)
        if index is not None:
            self._next_index = max(self._next_index, index + 1)

    def _publish(self, doc: StreamDocument) -> None:
        """Add a docked document to the workspace and make it the active one.

        Parameters
        ----------
        doc : StreamDocument
            A document which is already docked.
        """
        self._documents.append(doc)
        doc.closed.connect(lambda document=doc: self._on_document_closed(document))
        doc.changed.connect(self._on_document_changed)
        self._stack.setCurrentWidget(self._dock_host)
        self._raise(doc)
        self._update_save_actions()

    def _raise(self, doc: StreamDocument) -> None:
        """Bring ``doc`` to the front and make it the status bar's subject.

        Notes
        -----
        Explicit, because ``focusedDockWidgetChanged`` does not fire for the first
        document added to a window: the signal follows a user tab switch, it is not the
        source of truth.
        """
        doc.setAsCurrentTab()
        self._set_active(doc)

    def _document_for(self, identity: StreamIdentity) -> StreamDocument | None:
        """Return the open document of ``identity``, or ``None``."""
        return next(
            (doc for doc in self._documents if doc.identity == identity),
            None,
        )

    def _open_selected(self) -> None:
        """Open the streams selected on the landing page."""
        self.open_streams(self._landing.selected_descriptors())

    def _update_open_action(self) -> None:
        """Enable Open only while a stream is selected and no load is running."""
        self._act_open.setEnabled(
            self._loading is None and bool(self._landing.selected_descriptors())
        )

    def _update_save_actions(self) -> None:
        """Enable both Save actions only while at least one document is open.

        Notes
        -----
        Not decoration: :func:`~mne_lsl.viewer.backend.save_configuration` refuses a
        configuration referencing no stream, because the reader refuses one on the way
        back in, so an enabled Save with nothing open would raise inside a Qt slot.
        """
        enabled = bool(self._documents)
        self._act_save.setEnabled(enabled)
        self._act_save_as.setEnabled(enabled)

    # -- saving a configuration --------------------------------------------------------
    def save_configuration(self) -> None:
        """Save this workspace, replacing the configuration it was loaded from.

        Notes
        -----
        Falls through to :meth:`save_configuration_as` when this workspace came from no
        configuration, or when the one it came from has been deleted or renamed
        meanwhile. Otherwise it writes straight away: the name is already the user's, so
        there is nothing to ask.
        """
        if not self._documents:
            return  # second line of defence, see '_update_save_actions'
        if self._source is None or self._source.casefold() not in self._saved_names():
            self.save_configuration_as()
            return
        self._write_configuration(self._source)

    def save_configuration_as(self) -> None:
        """Prompt for a name and save this workspace under it.

        Notes
        -----
        No character is rejected -- the slug function neutralizes them all -- and the
        user never sees or chooses a path: the storage location is deliberately hidden
        and the authoritative name lives inside the file.

        A name which is already taken raises one Replace confirmation. Two dialogs for
        one gesture is the single accepted exception to the one-modal rule, and it is
        the Replace verb the interface promises.
        """
        if not self._documents:
            return
        name, accepted = QInputDialog.getText(
            self, "Save configuration", "Name:", text=self._source or ""
        )
        if not accepted:
            return
        name = _clean_name(name)
        if not name:
            self.statusBar().showMessage("A configuration needs a name.", 4000)
            return
        taken = name.casefold() in self._saved_names()
        replacing_source = (
            self._source is not None and name.casefold() == self._source.casefold()
        )
        if taken and not replacing_source and not self._confirm_replace(name):
            return
        self._write_configuration(name)

    def _saved_names(self) -> set[str]:
        """Return the casefolded names of the configurations currently on disk.

        Notes
        -----
        Re-listed rather than read off ``self._configs``, which is only as fresh as the
        last Refresh: a collision check against a stale list would silently overwrite a
        configuration a second window had just written.
        """
        return {cfg.name.casefold() for cfg in list_configurations()}

    def _confirm_replace(self, name: str) -> bool:
        """Ask whether an existing configuration may be overwritten.

        Parameters
        ----------
        name : str
            Name of the configuration which already exists.

        Returns
        -------
        confirmed : bool
            Whether the user accepted.
        """
        return _message(
            self,
            "question",
            "Replace the configuration?",
            f"A configuration named '{name}' already exists. Replace it?",
        )

    def _write_configuration(self, name: str) -> None:
        """Capture this workspace under ``name`` and write it.

        Parameters
        ----------
        name : str
            Name to save under, already stripped and capped.

        Notes
        -----
        A document whose stream went away is saved like any other, and the status
        message says how many: the identity and every setting are still known, a
        configuration describes a *desired* workspace, and refusing the save would make
        a device which will be back tomorrow able to lose a workspace today. A frozen
        document saves like any other too, because the frozen state is not serialized at
        all.

        The confirmation is non-modal on purpose: a save which succeeded is not
        something to acknowledge. Only a save which *failed* is modal, because one
        gesture which ended in nothing having happened gets one dialog.
        """
        try:
            save_configuration(self._capture_configuration(name))
        except Exception as error:  # a trust boundary: the payload reaches json.dumps
            logger.warning("Could not save the configuration '%s': %s", name, error)
            _message(self, "critical", "Could not save the configuration", str(error))
            return
        self._source = name
        self.reload_configurations()
        message = f"Saved '{name}'."
        # the document's own state and not 'stream.connected': a stalled or refused
        # document *is* connected and would go unreported, while a mismatched one would
        # be described as merely disconnected.
        down = [doc for doc in self._documents if doc.state != LIVE]
        if down:
            plural = "" if len(down) == 1 else "s"
            message += f" {len(down)} stream{plural} currently interrupted."
        self.statusBar().showMessage(message, 6000)
        self._update_save_actions()

    def _capture_configuration(self, name: str) -> ViewerConfig:
        """Return the configuration describing this workspace.

        Parameters
        ----------
        name : str
            Name to save under.

        Returns
        -------
        cfg : ViewerConfig
            The envelope and the presentation payload.

        Notes
        -----
        The identities come from the documents, i.e. from what ``connect()`` back-filled
        from the inlet it opened, which is more authoritative than the discovery
        descriptor they were opened from. The channel sets are the acquisition names,
        which is both the availability contract and the key set of every channel-keyed
        block below.
        """
        return ViewerConfig(
            name=name,
            streams=[doc.identity.as_tuple() for doc in self._documents],
            channels={
                channel_key(doc.identity.as_tuple()): doc.model.acquisition_names()
                for doc in self._documents
            },
            presentation={
                "layout": self._capture_layout(),
                "window": self._capture_window(),
                "streams": [doc.capture_state() for doc in self._documents],
            },
        )

    def _capture_layout(self) -> str:
        """Return the Qt-ADS layout of the document area, as plain XML.

        Notes
        -----
        Plain and not compressed because ``configure_docking`` turns compression off and
        auto-formatting on, and the result round-trips through :func:`json.dumps`
        byte-for-byte -- so no base64 and no encoding field. The version travels inside
        the XML as its own ``UserVersion``, thus there is no version field either.
        """
        return bytes(self._manager.saveState(LAYOUT_VERSION)).decode("utf-8")

    def _capture_window(self) -> dict[str, Any]:
        """Return the saved rectangle and maximized state of this window.

        Notes
        -----
        The *normal* geometry and never ``geometry()``: while the window is maximized
        the latter is the full-screen rectangle, and storing that as the 'normal' size
        gives a restored window nothing sensible to un-maximize to. Never
        ``saveGeometry()`` either, whose internal heuristics are neither inspectable in
        the file nor auditable.
        """
        rect = self.normalGeometry()
        return {
            "x": rect.x(),
            "y": rect.y(),
            "width": rect.width(),
            "height": rect.height(),
            "maximized": self.isMaximized(),
        }

    # -- loading a configuration -------------------------------------------------------
    def open_configuration(self, name: str) -> None:
        """Connect every stream of a saved configuration and rebuild its workspace.

        Parameters
        ----------
        name : str
            Name of the configuration to open.

        Notes
        -----
        All or nothing, and it is the empty state which makes that nearly free: a
        configuration is only ever opened with no document open, so every document is
        built into a local list and published only once the whole workspace stands. A
        rollback is then 'discard the locals, delete the widgets, disconnect the
        streams' -- there is no previous workspace to restore and no partial interface
        to unwind.

        Event sources take part in the identity match and are **not** connected: they
        own no document in this milestone, so an inlet opened for one would be held by
        nobody and closed by nothing.

        The buffer size is derived from the widest window the display can select and not
        from the saved one, exactly as the incremental open path does. A window wider
        than the buffer is not an error -- ``get_data`` silently returns the shorter
        buffer -- and the display would then draw over part of its time axis for the
        rest of the session; the two call sites have to agree, or one configuration
        behaves differently depending on how it was opened.
        """
        if self._closed or self._loading is not None or self._documents:
            return
        cfg = next((entry for entry in self._configs if entry.name == name), None)
        if cfg is None or cfg.invalid_reason is not None:
            logger.warning("There is no loadable configuration named '%s'.", name)
            return
        descriptors: list[StreamDescriptor] = []
        for identity in cfg.streams:
            descriptor = self._descriptors.get(identity)
            if descriptor is None:
                # The race between the card's last evaluation and this click is one
                # event loop turn, thus this is defence -- and it is the difference
                # between one dialog and a 'KeyError' raised inside a Qt slot.
                self._report_load_failure(
                    f"{identity_text(identity)} is no longer on the network."
                )
                return
            if descriptor.sfreq != 0:
                descriptors.append(descriptor)
        if not descriptors:
            # Only reachable from a hand-edited file, since a saved workspace always
            # holds at least one document. It has to be refused here: an empty batch
            # makes the connector emit nothing at all, so the load would never finish
            # and the card would read 'Connecting…' with Refresh disabled for the rest
            # of the session.
            self._report_load_failure(
                f"'{cfg.name}' references no stream this viewer can open: an event "
                "source takes part in the match and owns no document."
            )
            return
        self._loading = _LoadAttempt(cfg, descriptors)
        # this card now reads 'Connecting…', every other one goes inert, and both
        # gestures a load must lock out are disabled -- all four derived from '_loading'
        # in one place, so none of them can be forgotten at one of the three
        # transitions.
        self._publish_configurations()
        self._loader.open(descriptors, derive_bufsize(WINDOW_RANGE[1]))

    def _on_load_connected(self, descriptor: object, stream: object) -> None:
        """Record a stream which connected for the load in flight.

        Notes
        -----
        A result which belongs to no current attempt is **released**, not merely
        ignored: the connector transfers stream ownership with its signal, so a
        connection landing after a rollback would otherwise leak a live inlet and its
        acquisition thread for the life of the process.
        """
        attempt = self._loading
        identity = descriptor.identity
        if attempt is None or identity not in attempt.pending:
            logger.warning(
                "Releasing the stream %s, which connected for a load nobody is waiting "
                "for any more.",
                identity.as_tuple(),
            )
            release_stream(stream)
            return
        attempt.pending.pop(identity)
        attempt.streams[identity] = stream
        self._finish_load()

    def _on_load_failed(self, descriptor: object, message: str) -> None:
        """Record a connection which failed for the load in flight."""
        attempt = self._loading
        identity = descriptor.identity
        if attempt is None or identity not in attempt.pending:
            return
        attempt.pending.pop(identity)
        attempt.errors.append(f"{identity_text(identity.as_tuple())}: {message}")
        self._finish_load()

    def _finish_load(self) -> None:
        """Continue the load once every connection has come back.

        Notes
        -----
        The substitute for the ``finished`` signal the connector does not have: a batch
        is complete once nothing is left in flight, and ``pending`` is the only record
        of that.

        The whole batch is waited out before a rollback, rather than rolling back on the
        first failure. Rolling back immediately would need ``Connector.stop()`` from
        inside a slot, which blocks the GUI thread in ``QThread.wait()`` for the length
        of one connection in flight. The outcome is identical -- nothing is published
        and every connected stream is released -- and the cost is that the user waits
        out the remaining connections before the dialog.
        """
        attempt = self._loading
        if attempt is None or attempt.pending:
            return
        if attempt.errors:
            self._rollback_load(
                attempt,
                f"'{attempt.cfg.name}' could not be opened: one of its streams did not "
                "connect.",
                "\n".join(attempt.errors),
            )
            return
        failure = self._validate_channels(attempt)
        if failure:
            self._rollback_load(attempt, failure, "")
            return
        self._apply_configuration(attempt)

    def _validate_channels(self, attempt: _LoadAttempt) -> str:
        """Return ``''`` when every saved channel set still matches, else the failure.

        Parameters
        ----------
        attempt : _LoadAttempt
            The load whose streams are all connected.

        Returns
        -------
        failure : str
            The one-line failure text, empty when every saved set still matches.

        Notes
        -----
        Read from ``stream.info['ch_names']`` and never from a fresh stream info: for a
        connected stream that list is already the interpreted, de-duplicated one, costs
        about a microsecond and emits no notice, while re-reading the description
        re-emits the duplicate-name warning -- which is an error under this repository's
        test configuration. The probe path is the only one which needs the description,
        because it has no measurement info to read.

        On the GUI thread, deliberately: with the processing block deferred the whole
        step is one set comparison per stream, and moving it to the worker would mean
        teaching the connector what a configuration is.

        The metadata read is guarded because a stream can disconnect *itself*: the
        acquisition thread catches any error and resets the stream, so a device
        unplugged between the connection returning and this running leaves the info
        unreadable. That is a validation failure, not a bug -- it earns the ordinary
        rollback and its one dialog, whereas an escaping exception would leave the load
        neither finished nor rolled back, with every later load in the session silently
        refused.
        """
        for identity, stream in attempt.streams.items():
            expected = attempt.cfg.channels.get(channel_key(identity.as_tuple()))
            if not expected:
                continue
            try:
                present = stream.info["ch_names"]
            except Exception as error:
                logger.warning("Could not read the channels of %s: %s", identity, error)
                return f"{identity.name} stopped responding while it was being checked."
            missing = missing_channels(expected, present)
            if missing:
                return channels_reason(identity.as_tuple(), missing)
        return ""

    def _apply_configuration(self, attempt: _LoadAttempt) -> None:
        """Build, dock, restore and publish the workspace of a connected configuration.

        Parameters
        ----------
        attempt : _LoadAttempt
            The load whose streams are all connected and validated.

        Notes
        -----
        Every document is appended to the local list *before* it is docked and before
        its state is applied, so a failure anywhere leaves the rollback with every
        widget it has to delete. ``apply_state`` raises for no input, but the document
        constructor refuses a stream declaring no sampling rate -- reachable when a
        saved regular stream has been re-provisioned as an event source.

        The saved edits are applied **through the channel model**, which is what
        ``apply_state`` does. Writing them onto the stream first would make the edited
        values the acquisition baseline, and the next save would then write empty
        deltas.
        """
        cfg = attempt.cfg
        docs: list[StreamDocument] = []
        try:
            # Inside the guard, not before it: the saved identities are user-editable,
            # so a non-hashable one raises here, and a raise outside this block escapes
            # into the exception policy, which logs and swallows -- leaving the load
            # neither finished nor rolled back.
            blocks = _presentation_blocks(cfg)
            # Before docking anything: every entry left in the map belongs to a document
            # nobody holds, since a load only ever starts from the empty state. Without
            # it, a second load of the same configuration re-registers the saved slot
            # names as duplicates, which overwrites the map entries and makes every
            # later 'saveState()' name one slot several times -- silently, and
            # permanently.
            self._purge_closed_documents()
            taken = set(self._manager.dockWidgetsMap())
            for identity in cfg.streams:
                stream = attempt.streams.get(StreamIdentity(*identity))
                if stream is None:
                    continue  # an event source, which this milestone does not connect
                block = blocks.get(identity, {})
                doc = StreamDocument(self._manager, stream, StreamIdentity(*identity))
                docs.append(doc)  # appended before anything below can raise
                self._dock(doc, _slot_for(block, taken))
                doc.apply_state(block)
        except Exception as error:  # a document which cannot be built is a load failure
            logger.exception("Could not build the workspace of '%s'.", cfg.name)
            self._rollback_load(
                attempt,
                f"'{cfg.name}' could not be opened: one of its documents could not be "
                "built.",
                str(error),
                docs,
            )
            return
        # Placement and geometry are cosmetic by the all-or-nothing rule, so a hostile
        # saved value must not cost the user a workspace whose streams are all connected
        # and validated. Both read user-editable numbers -- a coordinate outside the
        # platform integer range raises from Qt itself -- and a raise here would escape
        # into the exception policy with the documents built but never published.
        try:
            self._restore_layout(cfg.presentation.get("layout"), docs)
            _restore_geometry(self, cfg.presentation.get("window"))
        except Exception:
            logger.exception("Could not restore the saved placement of '%s'.", cfg.name)
        for doc in docs:
            self._publish(doc)
        self._source = cfg.name
        self._loading = None
        # the cards go live again; harmless, as the landing page is no longer shown.
        self._publish_configurations()
        self._update_save_actions()

    def _purge_closed_documents(self) -> None:
        """Delete every dock widget the manager still holds in a closed state.

        Notes
        -----
        A closed ``CDockWidget`` keeps its entry in the manager's map for the life of
        the process -- a *removed* or *deleted* one does not -- so this is both what
        frees the saved slot names for reuse and what stops one closed document's widget
        tree from living on for the session.

        ``deleteDockWidget()`` emits no ``closed``, thus the teardown is called
        explicitly first; it is idempotent, so a document which really did close is a
        no-op. Never a plain ``deleteLater()`` on a still-registered dock widget, which
        crashes the process during the deferred-deletion flush.
        """
        for widget in tuple(self._manager.dockWidgetsMap().values()):
            if not widget.isClosed():
                continue
            if isinstance(widget, StreamDocument):
                widget.teardown()
            widget.deleteDockWidget()
        # Mandatory, and measured: deleting the last dock widget of an area destroys the
        # area with it, and the next 'addDockWidget' handed that dangling pointer as its
        # insertion target crashes the process outright.
        self._dock_area = None

    def _restore_layout(self, layout: object, docs: Sequence[StreamDocument]) -> None:
        """Restore the saved dock layout, degrading to plain tabs on any problem.

        Parameters
        ----------
        layout : str
            The saved Qt-ADS XML.
        docs : sequence of StreamDocument
            The documents which were just docked, in saved order.

        Notes
        -----
        A layout which cannot be restored is **never** a load failure: all or nothing
        applies to streams and documents, not to widget placement, and one non-modal
        note is also the whole defence against the one cross-binding case which was
        never verified -- floating containers and auto-hide bars, neither of which this
        viewer produces.

        The re-add loop is mandatory rather than defensive. Measured: a registered
        widget the XML does not name is left closed and ``restoreState`` still returns
        ``True``; worse, an XML naming *no* registered widget also returns ``True``
        while closing every document, leaving an empty workspace behind a successful
        return code. This loop is the only thing between that and a blank window, and it
        degrades exactly to the specified plain-tabs fallback.
        """
        restored = False
        # The '<Container' test is not a sanity check, it prevents a process crash.
        # 'restoreState' dereferences the first container unconditionally once the root
        # element and the user version match, so a well-formed document carrying no
        # '<Container>' child segfaults inside C++ -- which the 'except' below cannot
        # catch. Measured: a 107-byte root-only document, and a real layout truncated
        # anywhere between the root start tag and the first container, both abort the
        # process, while '<Container/>' alone is safe. A blob which is *not* well-formed
        # is refused cleanly and returns False, which is why this is easy to miss.
        if isinstance(layout, str) and layout:
            if "<Container" in layout:
                try:
                    restored = self._manager.restoreState(
                        QByteArray(layout.encode("utf-8")), LAYOUT_VERSION
                    )
                except Exception as error:  # a hand-edited blob reaches a C++ parser
                    logger.warning("Could not restore the saved layout: %s", error)
                    restored = False
            else:
                logger.warning("The saved layout describes no dock container.")
            if not restored:
                self.statusBar().showMessage(
                    "The saved layout could not be restored; the streams are shown as "
                    "tabs.",
                    6000,
                )
        area = None
        for doc in docs:
            if doc.isClosed():
                self._manager.addDockWidget(
                    ads.DockWidgetArea.CenterDockWidgetArea, doc, area
                )
            area = doc.dockAreaWidget()
        self._dock_area = area

    def _rollback_load(
        self,
        attempt: _LoadAttempt,
        summary: str,
        detail: str,
        docs: Sequence[StreamDocument] = (),
    ) -> None:
        """Undo a failed load completely and report it once.

        Parameters
        ----------
        attempt : _LoadAttempt
            The load being abandoned.
        summary : str
            One line saying what could not be done.
        detail : str
            The underlying messages, empty when the summary is already the whole story.
        docs : sequence of StreamDocument
            The documents which were built, in the order they were built.

        Notes
        -----
        ``self._loading`` is cleared **first**, so that a connection still in flight
        from the same batch takes the 'not my attempt' branch and releases its stream
        instead of joining an attempt nobody is finishing.

        A document is disposed of with its own teardown -- which stops the render clock,
        drops the model-to-display edge and disconnects the stream it owns -- followed
        by ``deleteDockWidget()`` if it reached the manager, and by ``deleteLater()`` if
        it did not. Never ``deleteLater()`` on a docked one, which crashes the process.

        A fresh discovery pass is started at the end, so the card re-enters the
        progressive availability flow and may legitimately settle on unavailable.
        """
        self._loading = None
        registered = set(self._manager.dockWidgetsMap())
        for doc in reversed(tuple(docs)):
            doc.teardown()
            if doc.objectName() in registered:
                doc.deleteDockWidget()
            else:
                doc.deleteLater()
        for stream in reversed(tuple(attempt.streams.values())):
            release_stream(stream)
        self._dock_area = None
        # 'refresh' republishes, which is what re-enables Refresh and Open -- and why
        # '_loading' has to be cleared before it rather than after.
        self.refresh()
        self._report_load_failure(summary, detail)

    def _report_load_failure(self, summary: str, detail: str = "") -> None:
        """Show the one dialog a failed load gets.

        Parameters
        ----------
        summary : str
            One line saying what could not be done.
        detail : str
            The underlying messages, appended below the summary.

        Notes
        -----
        One click which ended in nothing having happened gets exactly one dialog, and
        nothing which happens in the background is ever modal. The detail is appended to
        the text rather than set as Qt's collapsible detailed text, because that needs a
        constructed dialog and an ``exec()`` -- a nested event loop no test can time out
        of, where the static helper is both monkeypatchable and free of one.
        """
        logger.warning("%s %s", summary, detail)
        _message(self, "critical", _LOAD_FAILURE_TITLE, summary, detail)

    # -- renaming and deleting a configuration -----------------------------------------
    def _rename_configuration(self, name: str) -> None:
        """Prompt for a new name and rename a saved configuration.

        Parameters
        ----------
        name : str
            Current name of the configuration.

        Notes
        -----
        A blank name or a collision is reported in one warning and the gesture ends; the
        prompt is not re-shown, which would be a loop the user cannot leave except by
        cancelling twice. This is the one manage verb which works while the streams are
        absent, i.e. in the state a configuration spends most of its life in.
        """
        new_name, accepted = QInputDialog.getText(
            self, "Rename configuration", "Name:", text=name
        )
        if not accepted:
            return
        # Normalized once, into a local, and reused: computing it twice is how the two
        # spellings drifted, and the remembered source has to be the name actually
        # written.
        cleaned = _clean_name(new_name)
        try:
            rename_configuration(name, cleaned)
        except Exception as error:  # a trust boundary: the name reaches the filesystem
            logger.warning("Could not rename the configuration '%s': %s", name, error)
            _message(self, "warning", "Could not rename the configuration", str(error))
            return
        if self._source is not None and self._source.casefold() == name.casefold():
            # the source follows the rename, or the next plain Save writes a third file
            # under the old name.
            self._source = cleaned
        self.reload_configurations()

    def _delete_configuration(self, name: str) -> None:
        """Confirm and delete a saved configuration.

        Parameters
        ----------
        name : str
            Name of the configuration to delete.

        Notes
        -----
        One confirmation, because there is no undo. Offered on an invalid card too: that
        is the only route to clearing a corrupt file from the interface.
        """
        if not _message(
            self,
            "question",
            "Delete the configuration?",
            f"Delete the configuration '{name}'? This cannot be undone.",
        ):
            return
        try:
            delete_configuration(name)
        except Exception as error:  # a trust boundary: this reaches the filesystem
            logger.warning("Could not delete the configuration '%s': %s", name, error)
            _message(self, "warning", "Could not delete the configuration", str(error))
            return
        if self._source is not None and self._source.casefold() == name.casefold():
            self._source = None
        self.reload_configurations()

    # -- the active document and the status bar ----------------------------------------
    def _on_focus_changed(self, _old: object, now: object) -> None:
        """Follow a user-driven focus change to another document."""
        if isinstance(now, StreamDocument):
            self._set_active(now)

    def _set_active(self, doc: StreamDocument | None) -> None:
        """Make ``doc`` the document the status bar describes."""
        self._active = doc
        self._update_status_bar()

    def _on_document_changed(self, _doc: StreamDocument) -> None:
        """Refresh the status bar for a change of any document.

        Notes
        -----
        The emitter is ignored: :meth:`_update_status_bar` renders
        :attr:`active_document` whoever emitted, thus a guard on the emitter would be an
        invariant no behaviour distinguishes -- measured at 4 µs for a full redraw, on a
        signal which fires once per user action and never per tick or per row.
        """
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Render the active document's fields into the three permanent segments."""
        if self._active is None:
            self._sb_state.setText("Disconnected")
            self._sb_identity.clear()
            self._sb_identity.setToolTip("")
            self._sb_meta.clear()
            return
        fields = self._active.status_fields()
        # read off the document itself: an identity is immutable, thus copying it into
        # the fields of every refresh would be one more place for it to disagree with.
        identity = self._active.identity
        self._sb_state.setText(fields["state"])
        self._sb_identity.setText(
            f"{identity.name} • {identity.stype} • {_elide(identity.source_id)}"
        )
        # the elided source ID stays recoverable, as half of an exact identity.
        self._sb_identity.setToolTip(
            f"{identity.name} • {identity.stype} • {identity.source_id}"
        )
        self._sb_meta.setText(
            " • ".join(
                (
                    fields["channels"],
                    fields["sfreq"],
                    fields["history"],
                    fields["latency"],
                )
            )
        )

    # -- theming -----------------------------------------------------------------------
    def _apply_ads_theme(self) -> None:
        """Push the docking style sheet and the docking title-bar glyphs.

        Notes
        -----
        ponytail: re-registering the glyphs colors the tabs and areas created *after* a
        theme flip, while an already-built title-bar button keeps its baked icon until
        it is rebuilt. The upgrade is to walk the live title bars and re-set the icons.
        """
        self._manager.setStyleSheet(_ADS_QSS)
        provider = self._manager.iconProvider()
        for slot, glyph in _ADS_ICONS.items():
            provider.registerCustomIcon(getattr(ads.eIcon, slot), icon(glyph))

    def _on_theme_changed(self, _mode: str) -> None:
        """Rebuild every baked icon of the window and of its documents."""
        for action, name in self._toolbar_icons:
            action.setIcon(icon(name))
        self._apply_ads_theme()
        self._landing.retint_icons()
        for doc in self._documents:
            doc.retint_icons()

    def showEvent(self, event: QShowEvent) -> None:
        """Follow the theme again, and catch up whatever a flip changed while hidden.

        Notes
        -----
        The counterpart of the unfollow in :meth:`closeEvent`, and the same pair the
        trace display and the Channels page carry: a window hidden across a theme flip
        would otherwise keep the previous mode's baked toolbar icons and dock chrome.
        """
        super().showEvent(event)
        if not self._following_theme:
            follow_theme(self, self._on_theme_changed, True)
            self._on_theme_changed(theme_controller.mode)  # catch up a missed flip

    def _about(self) -> None:
        """Show the three versions a bug report needs."""
        QMessageBox.about(
            self,
            "About the MNE-LSL viewer",
            f"MNE-LSL {__version__}\nQt binding: {qtpy.API_NAME}\nQt {qVersion()}",
        )

    # -- teardown ----------------------------------------------------------------------
    def _on_document_closed(self, doc: StreamDocument) -> None:
        """Drop a closed document and retarget the status bar.

        Notes
        -----
        The document's own ``teardown`` is connected to ``closed`` by its constructor,
        thus it has already run by the time this does.

        ponytail: the closed dock widget stays in the manager's map, so one document's
        widget tree -- its curve pool, its model and its page -- lives on for the
        process, per closed document. The upgrade is ``deleteDockWidget`` from here, the
        call :meth:`_purge_closed_documents` already makes at load time, which leaves an
        invalid Python wrapper behind and is therefore conditioned on the PySide6 job
        being green first.
        """
        if doc in self._documents:
            self._documents.remove(doc)
        if self._active is doc:
            # Asked of the dock area rather than guessed as the last document opened:
            # Qt-ADS has already promoted another tab by the time this runs -- measured,
            # shown and unshown -- and its choice is not the last one as soon as three
            # are open, so the bar would describe a document which is not the visible
            # tab. The membership check covers an area emptied by a split.
            area = doc.dockAreaWidget()
            current = None if area is None else area.currentDockWidget()
            if current not in self._documents:
                current = self._documents[-1] if self._documents else None
            self._active = current
        if not self._documents:
            self._stack.setCurrentWidget(self._landing)
        self._update_save_actions()
        self._update_status_bar()

    def close_all_documents(self) -> None:
        """Close every open document, tearing down its render clock and stream."""
        # a copy: the close handler mutates the list this iterates.
        for doc in tuple(self._documents):
            doc.closeDockWidget()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Close every document cleanly before the window closes.

        Notes
        -----
        All five waits are mandatory. Closing the window does not close the dock widgets
        by itself -- measured: no ``closed`` is emitted and every render clock keeps
        ticking -- and the ``aboutToQuit`` fallback each worker owner installs is only
        emitted when a real event loop exits, i.e. never on the path which merely shows
        the window. This is the only teardown there.

        The loader and the prober are the two which are easy to forget, because both
        start their thread lazily: a thread which is running when its ``QThread`` is
        destroyed makes Qt abort the process, and only sometimes, so a missing stop
        lands as flake. The reconnections are the third, for the opposite reason: they
        own no thread to stop, so there is nothing in this list to forget -- and without
        :func:`~mne_lsl.viewer.backend.wait_for_reconnects` the process blocks in the
        pool's destructor instead, after this method has returned, with every document
        already gone and the outcome delivered to nobody.

        The documents are closed *before* that wait, so that a reconnection which lands
        during it finds a torn-down document and releases the stream it just connected.

        A load in flight owns streams no document holds yet, and stopping the loader
        only cancels what has not been connected: without releasing them here, every
        stream the attempt had already connected keeps its inlet and its acquisition
        thread for the life of the process, since nothing else can reach them once the
        window is gone.
        """
        self._closed = True
        if self._loading is not None:
            attempt, self._loading = self._loading, None
            for stream in reversed(tuple(attempt.streams.values())):
                release_stream(stream)
        self.close_all_documents()
        self._discovery.stop()
        self._connector.stop()
        self._loader.stop()
        self._prober.stop()
        wait_for_reconnects()
        follow_theme(self, self._on_theme_changed, False)
        super().closeEvent(event)
