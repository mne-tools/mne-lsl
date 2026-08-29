"""One dockable stream document: a controller and a trace display."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QLabel,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..utils.logs import logger
from ._bootstrap import import_ads
from .backend import (
    RESUME_LIVE,
    RESUME_MISMATCH,
    StreamSignature,
    disconnect_text,
    release_stream,
    submit_reconnect,
)
from .controller import ChannelModel, ChannelsPage, unit_choices, unit_label
from .display import TraceDisplay
from .theme import _ICON_PX, icon, theme_controller, tokens
from .widgets import Banner

if TYPE_CHECKING:
    from typing import Any

    from ..stream import BaseStream
    from .backend import StreamIdentity

ads = import_ads()

# Initial widths of the controller panel and of the trace display, in pixels: the panel
# opens a little wider than its own minimum and the display takes the rest.
_PANEL_SIZES = (300, 900)

# The connection states of a document. Bare names rather than a 'STATE_*' family:
# 'backend/_config.py' already exports 'STATE_AVAILABLE' / 'STATE_CHECKING' / ... for a
# state of a saved *configuration* card, and a second 'STATE_*' family one import away
# is how a reader ends up reading the wrong one.
LIVE = "live"
INTERRUPTED = "interrupted"
MISMATCHED = "mismatched"
CLOSED = "closed"

# ponytail: one flat timeout, in seconds, for every stream. A source pushing one chunk a
# second and one pushing at 1 kHz are held to the same 5 s of silence, so a slow but
# healthy source is called stalled while a fast one is called stalled far too late. The
# upgrade is to derive it from the observed inter-chunk interval, which needs a running
# estimate the detector does not have today.
T_STALL = 5.0
# How long a resumed document must keep delivering before its retry ladder is reset, in
# seconds. Below this, a loss is a flap and the ladder keeps climbing.
T_STABLE = 5.0
# Delay before each reconnection attempt, in seconds, held at the last value. Paced so
# that a source which is down for a while is not re-resolved 30 times a second; the
# resolution inside one attempt already blocks for a second of its own.
_BACKOFF = (1.0, 2.0, 5.0, 10.0)


class StreamDocument(ads.CDockWidget):
    """One stream document, i.e. the dockable unit of one regularly sampled stream.

    The content is a toolbar over a horizontal splitter holding the controller and the
    trace display. Documents tab together and split cleanly when dragged to a side; the
    controller can be hidden per document.

    Parameters
    ----------
    manager : ads.CDockManager
        The dock manager owning the document area. Required: the two-argument
        ``CDockWidget(title, parent)`` form is deprecated in PySide6-QtAds 5.0.0 and its
        ``DeprecationWarning`` is an error under the pytest configuration of this
        repository.
    stream : BaseStream
        The connected stream rendered by this document.
    identity : StreamIdentity
        Exact identity of the stream, reported by the status bar and saved in a
        configuration.
    owns_stream : bool
        If ``True``, :meth:`teardown` disconnects the stream. A borrowed stream, i.e.
        one provided to :class:`~mne_lsl.viewer.Viewer` by
        :meth:`~mne_lsl.stream.BaseStream.plot`, is never disconnected on teardown, and
        is reconnected only when the operator asks for it through
        :meth:`StreamDocument.retry`: a reconnection replaces the object's inlet and
        buffer, which drops the filters, the callbacks and the acquisition delay its
        owner set on it.
    parent : QWidget | None
        Parent widget.

    Raises
    ------
    ValueError
        If the stream is irregularly sampled, i.e. if it declares ``sfreq == 0``.

    Attributes
    ----------
    changed : Signal
        Emitted with the document itself when its state moves, e.g. on a freeze or a
        change of the displayed channel count, so the shared status bar refreshes.
    """

    changed = Signal(object)

    def __init__(
        self,
        manager: ads.CDockManager,
        stream: BaseStream,
        identity: StreamIdentity,
        *,
        owns_stream: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the document."""
        # Refused here, the choke point both entry points meet at: 'open_streams' skips
        # an event source already, while 'adopt_stream' -- the 'BaseStream.plot()' path
        # -- would otherwise open a document over a stream with no continuous signal,
        # whose time axis means nothing and whose window 'get_data' reads as a count.
        sfreq = float(stream.info["sfreq"])
        if sfreq == 0:
            raise ValueError(
                f"The stream {identity.as_tuple()} is irregularly sampled and cannot "
                "be opened as a document: it carries no continuous signal to draw."
            )
        # 'CDockWidget''s C++ constructor calls 'setObjectName(title)', and Qt-ADS keys
        # its dock-widget map, thus its layout save/restore, by 'objectName()'. Naming a
        # document after the stream name would therefore make two same-name streams
        # collide, which is an explicitly supported case since an identity is the
        # '(name, stype, source_id)' triple. The unique object name is assigned by the
        # window, in the one method which also calls 'addDockWidget' -- where the map is
        # populated -- and the tab title carries the human stream name.
        super().__init__(manager, identity.name, parent)
        self._stream = stream
        self._identity = identity
        self._owns_stream = bool(owns_stream)
        self._frozen = False
        self._torn = False
        # Last width the controller panel had while it was shown. A laid-out
        # 'QSplitter' reports a hidden child as 0 pixels wide -- measured -- so this is
        # the only place the number a configuration has to save stays readable. The
        # splitter's own memory is what brings the panel back during the session; this
        # exists for serialization alone, see 'controller_width'.
        self._panel_width = _PANEL_SIZES[0]

        # a document docks and tabs inside the main window for this milestone; it does
        # not become a separate top-level window. The only call there is: closable,
        # movable and focusable are in 'DefaultDockWidgetFeatures' already, and setting
        # them explicitly yields a bit-identical mask -- measured 547 either way, from a
        # 551 default. 'test_focus_highlighting_is_live' and
        # 'test_teardown_via_closed_signal' are the net if a binding ever drops one.
        self.setFeature(ads.CDockWidget.DockWidgetFeature.DockWidgetFloatable, False)
        # The one teardown hook. Qt-ADS routes the document toolbar's close button, the
        # dock-area close button and the window's own close loop through
        # 'closeDockWidget()', which emits this exactly once, and it delivers no close
        # event to the content widget -- so without this nothing stops the render clock.
        self.closed.connect(self.teardown)

        self.model = ChannelModel(stream, parent=self)
        self.channels = ChannelsPage(self.model, parent=self)
        self.trace = TraceDisplay(stream, parent=self)
        # The initial push. It keeps the display's own default -- acquisition order,
        # every channel visible -- from being what is drawn: the model is the only owner
        # of the order and the visibility, so it states them once here and on every
        # change below.
        self.trace.set_channel_layout(self.model.visible_acq_indices())
        self.model.layout_changed.connect(self._push_layout)
        self.model.metadata_changed.connect(self.trace.refresh_metadata)

        self._panel = self._build_panel()
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._panel)
        self._splitter.addWidget(self.trace)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes(list(_PANEL_SIZES))

        self._content = QWidget()
        box = QVBoxLayout(self._content)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(self._build_toolbar())
        box.addWidget(self._splitter, 1)
        self.setWidget(self._content)

        # -- the connection state machine, driven by the render clock alone ------------
        # Before 'retint_icons', which forwards to the banner and therefore has to find
        # the attribute already set.
        self._state = LIVE
        self._notice = ""  # the bare reason the banner and the status bar show
        # Cached, because 'stream.info' and 'stream.dtype' are unreadable while a stream
        # is disconnected, which is exactly when a resume signature is needed. The
        # expected channel names are deliberately *not* cached: the model reports the
        # recorded wire names in acquisition order, and that list survives a rename.
        self._sfreq = sfreq
        self._dtype = str(stream.dtype)
        self._last_ts: float | None = None  # previous 'trace.last_timestamp'
        self._data_at: float | None = None  # monotonic() of the last change; None=unset
        self._next_attempt: float | None = None  # monotonic() deadline; None=no attempt
        self._backoff = 0  # index into '_BACKOFF'
        self._resumed_at: float | None = None  # monotonic() of the last resume
        self._attempt: object | None = None  # the in-flight emitter, or None
        self._awaiting_data = False  # reconnected, waiting for one non-empty window
        self._banner: Banner | None = None  # built on first need, never destroyed
        self.trace.polled.connect(self._on_polled)

        self.retint_icons()
        # Last, and here rather than in the window's registration: this object owns the
        # live/frozen state, so a document whose clock the caller has not started yet
        # reports 'Live' over a viewport which never advances. Through '_apply_clock'
        # rather than 'trace.start()', so that it really is the clock's only writer.
        self._apply_clock()

    # -- construction ------------------------------------------------------------------
    def _build_panel(self) -> QTabWidget:
        """Build the controller panel, i.e. the tab widget holding the Channels page."""
        panel = QTabWidget()
        panel.setTabPosition(QTabWidget.TabPosition.West)
        panel.setDocumentMode(True)
        # One tab: the Processing and Events pages are added by the phases which
        # implement them, and instantiating either stub today yields a widget which
        # raises.
        panel.addTab(self.channels, "Channels")
        return panel

    def _build_toolbar(self) -> QToolBar:
        """Build the document toolbar: the indicator, the toggles and the close."""
        bar = QToolBar()
        bar.setMovable(False)
        bar.setIconSize(QSize(_ICON_PX, _ICON_PX))

        self._indicator = QLabel()
        self._indicator.setContentsMargins(6, 0, 6, 0)
        self._indicator.setToolTip(
            "Whether the viewport advances. The acquisition keeps running while the "
            "viewport is frozen."
        )
        bar.addWidget(self._indicator)

        self._freeze_button = QToolButton()
        self._freeze_button.setCheckable(True)
        self._freeze_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._freeze_button.setToolTip(
            "Freeze the viewport on the current window, or resume it"
        )
        self._freeze_button.toggled.connect(self.set_frozen)
        bar.addWidget(self._freeze_button)

        self._controller_button = QToolButton()
        self._controller_button.setCheckable(True)
        self._controller_button.setChecked(True)
        self._controller_button.setToolTip("Show or hide the controller panel")
        self._controller_button.toggled.connect(self.set_controller_visible)
        bar.addWidget(self._controller_button)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        self._close_button = QToolButton()
        # A button added with 'addWidget' ignores the bar's own icon size and renders
        # oversized, thus the size is set on the button as well.
        self._close_button.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self._close_button.setToolTip("Close this stream document")
        self._close_button.clicked.connect(self.closeDockWidget)
        bar.addWidget(self._close_button)
        return bar

    # -- state -------------------------------------------------------------------------
    @property
    def identity(self) -> StreamIdentity:
        """Exact identity of the stream this document renders."""
        return self._identity

    @property
    def stream(self) -> BaseStream:
        """The stream this document renders, owned or borrowed."""
        return self._stream

    @property
    def owns_stream(self) -> bool:
        """Whether closing this document disconnects the stream."""
        return self._owns_stream

    @property
    def frozen(self) -> bool:
        """Whether the viewport is frozen while the acquisition keeps rolling."""
        return self._frozen

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or resume the viewport, i.e. stop or start the render clock.

        Parameters
        ----------
        frozen : bool
            Whether to freeze the viewport.

        Notes
        -----
        A no-op once the document was torn down. Reachable programmatically only, as the
        freeze button went with its toolbar, but resuming there restarts the 33 ms clock
        with nothing left to stop it -- and over a *borrowed* stream, which the teardown
        leaves connected, that clock draws into a closed widget.

        A freeze also suspends the disconnection detection, since the render clock is
        the only clock this document has. That is a deliberate gap: the viewport is
        already not advancing because the operator stopped it, and unfreezing notices a
        lost stream within one render period.
        """
        if self._torn:
            return
        self._frozen = bool(frozen)
        self._apply_clock()
        # mirrored under 'blockSignals', so that a programmatic call does not come back
        # through 'toggled' and emit 'changed' a second time.
        blocked = self._freeze_button.blockSignals(True)
        self._freeze_button.setChecked(self._frozen)
        self._freeze_button.blockSignals(blocked)
        self.changed.emit(self)  # the indicator was refreshed by '_apply_clock'

    def _refresh_freeze_ui(self) -> None:
        """Label the freeze button with the action it performs and paint the indicator.

        Notes
        -----
        One method and one branch on :attr:`StreamDocument.frozen` for both, as the two
        are never refreshed apart: a freeze and a theme flip each need the pair, and two
        copies of the condition are how a state ends up labelled 'Live' next to a frozen
        indicator.

        The indicator reports what its tooltip promises -- whether the viewport advances
        -- and an interrupted viewport does not, so the state overrides the label a
        freeze alone would produce. Only the label: the button keeps offering the
        operator's own toggle, which is a preference and takes effect on the resume.
        """
        palette = tokens(theme_controller.mode)
        if self._frozen:
            glyph, label = "mdi6.play", "Live"
            text, color = "■ Frozen", palette.warning
        else:
            glyph, label = "mdi6.pause", "Freeze"
            text, color = "● Live", palette.success
        if self._state in (INTERRUPTED, MISMATCHED):
            text, color = "■ Interrupted", palette.warning
        self._freeze_button.setIcon(icon(glyph))
        self._freeze_button.setText(label)
        # the glyph and the word are the cues which are not the color.
        self._indicator.setText(text)
        self._indicator.setStyleSheet(f"color: {color}; font-weight: 600;")

    @property
    def controller_visible(self) -> bool:
        """Whether the controller panel is shown."""
        # 'isHidden', not 'isVisible': a document which was never added to a shown
        # window has no visible child at all, while the toggle state is a property of
        # the panel alone.
        return not self._panel.isHidden()

    def set_controller_visible(self, visible: bool) -> None:
        """Show or hide the controller panel; it comes back at its previous width.

        Parameters
        ----------
        visible : bool
            Whether to show the controller panel.

        Notes
        -----
        No width is re-applied here, on purpose. Measured: a
        :class:`~qtpy.QtWidgets.QSplitter` restores a hidden child to the width it had,
        whatever the user dragged it to and even across a resize of the window while it
        was hidden. Re-applying the sizes around this toggle is therefore code no
        behaviour can distinguish.

        The width is nonetheless *recorded* on the way down, for
        :attr:`controller_width` alone: a laid-out splitter reports a hidden child as 0
        wide, so a configuration saved with the panel hidden would otherwise store a
        zero-width panel and restore one which stays invisible when toggled back on.
        """
        visible = bool(visible)
        if not visible:
            # 'or' because this may run twice: the second call reads the 0 the first one
            # produced, and the number to keep is the one from before the panel went
            # away.
            self._panel_width = self._splitter.sizes()[0] or self._panel_width
        self._panel.setVisible(visible)
        blocked = self._controller_button.blockSignals(True)
        self._controller_button.setChecked(visible)
        self._controller_button.blockSignals(blocked)

    @property
    def controller_width(self) -> int:
        """Width of the controller panel in pixels, as a configuration saves it.

        Notes
        -----
        The splitter's live value while the panel is shown, and the width it had before
        it was hidden otherwise: a laid-out :class:`~qtpy.QtWidgets.QSplitter` reports a
        hidden child as 0 wide, and saving that restores a panel the user cannot get
        back by toggling it.

        A zero falls back to the recorded width for the same reason, one the hidden test
        does not cover: a splitter lets its children collapse, so a panel the user
        dragged to nothing is 0 wide while still perfectly *shown*. Persisting that
        writes a configuration which opens with no Channels page, in every session it is
        loaded, recoverable only by finding the handle again -- whereas the collapse it
        came from lasts as long as the window and is undone by the toggle.
        """
        if not self.controller_visible:
            return self._panel_width
        return self._splitter.sizes()[0] or self._panel_width

    def set_controller_width(self, width: int) -> None:
        """Give the controller panel ``width`` pixels, the display taking the rest.

        Parameters
        ----------
        width : int
            Width of the controller panel in pixels, clamped to zero from below.

        Notes
        -----
        Applied to the splitter *and* recorded, so that a restore which hides the panel
        immediately afterwards still saves the width it was asked for. The total is read
        back from the splitter rather than from the widget, so the display keeps what
        the layout had given the pair.
        """
        width = max(0, int(width))
        self._panel_width = width or self._panel_width
        total = sum(self._splitter.sizes())
        self._splitter.setSizes([width, max(0, total - width)])

    def status_fields(self) -> dict[str, str]:
        """Return the status-bar fields describing the current state of the document.

        Returns
        -------
        fields : dict
            The connection and live/frozen state, the displayed and total channel
            counts, the sampling rate, the retained history and the processing latency.
            The identity is deliberately absent: it is immutable and the caller already
            holds it, in :attr:`StreamDocument.identity`.

        Notes
        -----
        Nothing is read off the stream while it is disconnected: both
        :attr:`~mne_lsl.stream.BaseStream.info` and
        :attr:`~mne_lsl.stream.BaseStream.n_buffer` raise there. The connection and the
        two fields it gates are read together, in one guarded block, *before* anything
        is reported. The acquisition thread resets the stream on its own account, so a
        check per field can answer differently per field, and a check made ahead of the
        read races the read which follows it -- which would raise out of the status
        bar's own slot. A read which fails is therefore a disconnection like any other,
        and cannot leave the state claiming ``Connected • Live`` next to five ``—``.

        The connection state and the live fields are decided independently, and the
        state never blanks the others. A stalled document is still *connected*, so
        blanking its channel count would report ``—`` for a stream the viewer can still
        read: one document therefore honestly reports ``Interrupted • No data`` next to
        a real channel count, while a document whose stream really went away reports
        ``—``, because there is nothing left to read.
        """
        fields = {
            "state": "Disconnected",
            "channels": "—",
            "sfreq": "—",
            "history": "—",
            "latency": "—",
        }
        try:
            connected = self._stream.connected
            if connected:
                sfreq = float(self._stream.info["sfreq"])
                n_buffer = self._stream.n_buffer
        except Exception as error:  # the acquisition thread reset it just now
            logger.debug("The stream %s became unreadable: %s", self._identity, error)
            connected = False
        if self._state in (INTERRUPTED, MISMATCHED):
            fields["state"] = f"Interrupted • {self._notice}"
        elif connected:
            fields["state"] = "Connected • " + ("Frozen" if self._frozen else "Live")
        if not connected:
            return fields
        # The displayed count is what the viewport shows, not what the layout holds: a
        # stream with fewer channels than the row count shows all of them.
        fields["channels"] = (
            f"{min(self.trace.n_visible, self.trace.n_rows)}/{self.trace.n_channels} ch"
        )
        fields["sfreq"] = f"{sfreq:g} Hz"
        # no zero guard: the constructor refuses a stream declaring 'sfreq == 0'.
        fields["history"] = f"{n_buffer / sfreq:g} s history"
        fields["latency"] = "No processing • 0 ms"
        return fields

    # -- the serializable state --------------------------------------------------------
    def _channels(self) -> list:
        """Return the channel records in presentation order."""
        return [self.model.channel(row) for row in range(self.model.rowCount())]

    def _rows_by_name(self) -> dict[str, int]:
        """Return the current display row of every channel, keyed by acquisition name.

        Notes
        -----
        Rebuilt at every step of :meth:`apply_state` rather than computed once: the
        acquisition name is stable across every edit -- a rename moves
        :attr:`~mne_lsl.viewer.controller.Channel.name`, never
        ``Channel.orig.name`` -- but the *row* moves as soon as the saved order is
        applied, and a map built before that reorder silently hides the wrong channels
        afterwards.
        """
        return {
            self.model.channel(row).orig.name: row
            for row in range(self.model.rowCount())
        }

    def capture_state(self) -> dict[str, Any]:
        """Return the serializable presentation state of this document.

        Returns
        -------
        state : dict
            The document's slot and identity, the channel edits as deltas against the
            acquisition metadata, the controller geometry and the display settings.

        Notes
        -----
        Every channel key is the **acquisition** name, never the edited one: that key
        set doubles as the availability contract of the configuration, so keying it by a
        renamed channel would make the configuration both unmatchable against a stream
        and unloadable against its own channel list.

        Deltas and never tables. A 256-channel document with no edit at all carries four
        empty containers and one name list, and ``channel_order`` is omitted entirely
        when it equals the acquisition order -- a full name list which says nothing is
        what turns an inspectable file into an unreadable one.

        Nothing live is captured and the stream is never read: the channel records
        already cache every field, and they cannot diverge because the model overwrites
        them wholesale after each of its own writes. That is also what lets a document
        whose stream went away still be saved, which is the point of a configuration
        describing a *desired* workspace. The frozen state, the buffer size, the render
        clock and the acquisition baseline itself are all deliberately absent: the first
        is live state and the others are re-supplied by the stream at connect.
        """
        channels = self._channels()
        order = [channel.orig.name for channel in channels]
        state: dict[str, Any] = {
            "slot": self.objectName(),
            "identity": list(self._identity.as_tuple()),
            "channel_order": order,
            "hidden": [c.orig.name for c in channels if not c.visible],
            "renames": {c.orig.name: c.name for c in channels if c.name != c.orig.name},
            "types": {
                c.orig.name: c.ch_type for c in channels if c.ch_type != c.orig.ch_type
            },
            "units": {
                c.orig.name: c.unit_mul
                for c in channels
                if c.unit_mul != c.orig.unit_mul
            },
            "bads": [c.orig.name for c in channels if c.bad],
            "controller": {
                "visible": self.controller_visible,
                "width": self.controller_width,
            },
            # already a fresh dict, see 'DisplayControls.state'
            "display": self.trace.controls.state,
        }
        if order == self.model.acquisition_names():
            del state["channel_order"]  # omit what equals the default
        return state

    def apply_state(self, state: Mapping[str, Any]) -> None:
        """Apply a saved presentation state; an unusable value is logged and skipped.

        Parameters
        ----------
        state : dict
            A mapping shaped like the one :meth:`capture_state` returns. Unknown keys
            are ignored, as is any value this document cannot use.

        Notes
        -----
        Best-effort per key, and it raises for no input at all. This reads a file the
        user can edit by hand, and the all-or-nothing guarantee of a configuration load
        is about streams and documents, not about individual settings: a restore which
        raised part-way would leave a document holding a mix of the saved and the
        default state, with a traceback in the log and a dialog naming no setting.

        Every write goes through the :class:`~mne_lsl.viewer.controller.ChannelModel`
        and never onto the stream. The model captures its acquisition baseline from
        whatever the stream declares when it is built, so applying the saved edits to
        the stream first would make the *edited* values that baseline -- which makes
        Reset a no-op for every restored edit and, far worse, makes the next
        :meth:`capture_state` produce **empty** deltas, i.e. destroy the configuration
        by the act of saving it. Driving the shipped mutators also reuses their checks
        instead of restating them.

        The order is fixed. Types precede units because the set of valid units follows
        the channel type, and a type change resets the multiplier. The controller width
        precedes its visibility so that hiding the panel cannot swallow the width. The
        display block is applied last, through the control bar, which validates every
        leaf itself and reaches the display along the same path a user click takes.
        """
        if not isinstance(state, Mapping):
            logger.warning(
                "Ignoring a saved document state which is not a mapping: %r.", state
            )
            return
        self._apply_types(state.get("types"))
        self._apply_units(state.get("units"))
        self._apply_renames(state.get("renames"))
        self._apply_bads(state.get("bads"))
        self._apply_order(state.get("channel_order"))
        self._apply_hidden(state.get("hidden"))
        self._apply_controller(state.get("controller"))
        display = state.get("display")
        if isinstance(display, Mapping):
            self.trace.controls.set_state(display)
        elif display is not None:
            self._skip("display", display)

    @staticmethod
    def _skip(key: str, value: object) -> None:
        """Log one unusable saved value and move on."""
        logger.warning("Ignoring the saved document setting %r: %r.", key, value)

    def _write(self, call, *args: object) -> None:
        """Run one restore step, logging and skipping a value the model refuses.

        Notes
        -----
        Deliberately broad. The mutators raise :class:`ValueError` for what they check
        themselves, but this is a trust boundary and the write reaches MNE, which has
        its own vocabulary of exceptions; the point of this method is that no one saved
        value can abort the restore of the ones around it.
        """
        try:
            call(*args)
        except Exception as error:  # see the note above
            logger.warning("Ignoring a saved document setting: %s", error)

    def _entries(self, payload: object, key: str) -> list[tuple[int, Any]]:
        """Return the ``(display row, saved value)`` pairs of a saved mapping.

        Parameters
        ----------
        payload : dict
            Saved mapping of acquisition name to value.
        key : str
            Name of the block, for the log messages.

        Returns
        -------
        entries : list of tuple
            One pair per channel the model still holds, in the saved iteration order.

        Notes
        -----
        The name is resolved here and the *value* is left untouched, so that each block
        checks the type it expects before grouping the rows by it: an unhashable value
        from a hand-edited file would otherwise raise from inside the grouping, which is
        the one thing this restore path may not do.
        """
        if payload is None:
            return []
        if not isinstance(payload, Mapping):
            self._skip(key, payload)
            return []
        rows = self._rows_by_name()
        entries: list[tuple[int, Any]] = []
        for name, value in payload.items():
            row = rows.get(name) if isinstance(name, str) else None
            if row is None:
                # A channel set may legally have grown, and a stream may have been
                # re-provisioned under the same identity: one log line, and the rest of
                # the block still applies.
                self._skip(f"{key}[{name!r}]", value)
                continue
            entries.append((row, value))
        return entries

    def _picked(self, payload: object, key: str) -> list[int] | None:
        """Return the current display rows of a saved list of acquisition names.

        Parameters
        ----------
        payload : list of str
            Saved acquisition names.
        key : str
            Name of the block, for the log messages.

        Returns
        -------
        rows : list of int | None
            The rows, in the saved order, or ``None`` when the block is absent or is not
            a list.
        """
        if payload is None:
            return None
        if not isinstance(payload, (list, tuple)):
            self._skip(key, payload)
            return None
        rows = self._rows_by_name()
        picked = []
        for name in payload:
            row = rows.get(name) if isinstance(name, str) else None
            if row is None:
                self._skip(f"{key} channel", name)
                continue
            picked.append(row)
        return picked

    def _apply_types(self, payload: object) -> None:
        """Restore the saved channel types, one bulk write per type."""
        grouped: dict[str, list[int]] = {}
        for row, value in self._entries(payload, "types"):
            if not isinstance(value, str):
                self._skip("types", value)
                continue
            grouped.setdefault(value, []).append(row)
        for value, rows in grouped.items():
            self._write(self.model.set_type, rows, value)

    def _apply_units(self, payload: object) -> None:
        """Restore the saved unit multipliers, one bulk write per human label.

        Notes
        -----
        The saved value is the integer multiplier, which is what a channel actually
        carries, while the model writes a human label. The translation reads the kind
        the channel holds *now*, i.e. after the saved types were applied, and a label
        the channel's kind does not offer is refused here rather than by the mutator, so
        that the log line names the block. Two channels sharing a label share a kind by
        construction, since the label spells the unit out.
        """
        labelled: dict[str, list[int]] = {}
        for row, mul in self._entries(payload, "units"):
            if isinstance(mul, bool) or not isinstance(mul, int):
                self._skip("units", mul)
                continue
            channel = self.model.channel(row)
            label = unit_label(channel.unit_kind, mul)
            if label not in unit_choices(channel.unit_kind):
                self._skip(f"units[{channel.orig.name!r}]", mul)
                continue
            labelled.setdefault(label, []).append(row)
        for label, rows in labelled.items():
            self._write(self.model.set_unit, rows, label)

    def _apply_renames(self, payload: object) -> None:
        """Restore the saved channel names in one write, or one per channel on failure.

        Notes
        -----
        Grouped, because renaming row by row cannot express a **permutation**: the
        underlying operation refuses a name still held by another channel, so a saved
        configuration which merely swaps two names loses the first write to a collision
        that the same mapping applied at once resolves cleanly. Row by row that loss is
        silent -- one skipped entry and one log line per collision.

        The per-row loop is kept as the fallback, so a single unusable value in a
        hand-edited file costs only its own channel instead of the whole block. That
        makes the grouped call an optimisation of the common case and never a new
        failure mode.
        """
        wanted: dict[int, str] = {}
        for row, value in self._entries(payload, "renames"):
            if not isinstance(value, str):
                self._skip("renames", value)
                continue
            wanted[row] = value
        if not wanted:
            return
        try:
            self.model.rename_many(wanted)
        except Exception as error:
            logger.warning(
                "Could not restore %i channel name(s) together (%s); applying them one "
                "at a time.",
                len(wanted),
                error,
            )
            for row, value in wanted.items():
                self._write(self.model.rename, row, value)

    def _apply_bads(self, payload: object) -> None:
        """Restore the saved bad list, marking the rest good.

        Notes
        -----
        The saved list is absolute and not a delta, thus a channel the device itself
        declares bad and the user marked good has to be marked good again -- otherwise
        the state a document reports after a restore differs from the one it restored
        from, which is exactly what the save/load/save round trip has to rule out. The
        second write is skipped when there is nothing to clear, i.e. in the common case.
        """
        rows = self._picked(payload, "bads")
        if rows is None:
            return
        self._write(self.model.set_bad, rows, True)
        saved = set(rows)
        stale = [
            row
            for row in range(self.model.rowCount())
            if row not in saved and self.model.channel(row).bad
        ]
        if stale:
            self._write(self.model.set_bad, stale, False)

    def _apply_order(self, payload: object) -> None:
        """Restore the saved presentation order.

        Notes
        -----
        The names are translated to acquisition indices, which is the model's ordering
        vocabulary. A saved order which no longer covers every channel is refused by the
        model as a whole rather than partially applied: the omitted channels would drop
        out of the row list, hence out of the display layout -- undrawable, unreachable.
        """
        rows = self._picked(payload, "channel_order")
        if rows is None:
            return
        self._write(
            self.model.set_order, [self.model.channel(row).acq_index for row in rows]
        )

    def _apply_hidden(self, payload: object) -> None:
        """Restore the saved hidden channels, in one write."""
        rows = self._picked(payload, "hidden")
        if rows:
            self._write(self.model.set_visible, rows, False)

    def _apply_controller(self, payload: object) -> None:
        """Restore the controller geometry, its width before its visibility."""
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            self._skip("controller", payload)
            return
        width = payload.get("width")
        if width is not None:
            if isinstance(width, bool) or not isinstance(width, int) or width < 0:
                self._skip("controller.width", width)
            else:
                self.set_controller_width(width)
        visible = payload.get("visible")
        if visible is not None:
            if isinstance(visible, bool):
                self.set_controller_visible(visible)
            else:
                self._skip("controller.visible", visible)

    def retint_icons(self) -> None:
        """Rebuild the toolbar icons and the indicator for the active theme."""
        # a 'QIcon' bakes its color at creation, thus a theme flip needs every icon of
        # the bar rebuilt rather than merely repainted.
        self._refresh_freeze_ui()  # the freeze glyph and the indicator color
        self._controller_button.setIcon(icon("mdi6.tune-variant"))
        self._close_button.setIcon(icon("mdi6.close"))
        if self._banner is not None:
            self._banner.retint_icons()  # the banner does not follow the theme itself

    # -- the connection state ----------------------------------------------------------
    @property
    def state(self) -> str:
        """Connection state of the document, one of the four module constants.

        :type: :class:`str`
        """
        return self._state

    @property
    def notice(self) -> str:
        """One-line reason of the current interruption, ``''`` while live.

        :type: :class:`str`
        """
        return self._notice

    def retry(self) -> None:
        """Look for the stream again, now, whatever the notice strip is showing.

        Notes
        -----
        A no-op while live, while closed and while an attempt is already in flight; the
        verb of both interrupted states otherwise. It is the *only* way a **borrowed**
        stream is ever reconnected: reconnecting one in place destroys the filters, the
        callbacks and the acquisition delay its owner set, so it happens on request and
        never on a timer -- which is why the notice strip of a borrowed stream offers
        Retry even while :data:`INTERRUPTED`.

        The whole check is re-run rather than whatever answered the identity being
        adopted, and the retry ladder starts over, because the operator asked for this
        attempt.

        Reopening the document as a new one is not offered: closing and reopening it
        already does that, at the cost of every channel edit, display setting and layout
        position -- which is what this verb exists to preserve.
        """
        if (
            self._torn
            or self._attempt is not None
            or self._state not in (INTERRUPTED, MISMATCHED)
        ):
            return
        self._backoff = 0
        self._enter(INTERRUPTED, self._notice)
        self._submit_attempt()

    def _on_polled(self) -> None:
        """Advance the state machine by one render tick.

        Notes
        -----
        The single clock of this document, ~30 times a second, on the GUI thread. It
        runs on :attr:`~mne_lsl.viewer.display.TraceDisplay.polled` rather than on a
        :class:`~qtpy.QtCore.QTimer` of its own: the retry deadline is a
        :func:`time.monotonic` comparison on a tick which already exists, which costs
        33 ms of granularity on a 1--10 s interval, and a per-document timer is one more
        object to stop on every teardown path.

        The three early returns restate, locally, what :meth:`_apply_clock` already
        arranges: this handler may only ever run for a document which is live or waiting
        for its stream to come back. A frozen document is the one which needs saying
        twice, because a freeze deliberately suspends the detection and nothing else --
        so a tick arriving from anywhere but the clock must not advance the machine.
        """
        if self._torn or self._state in (MISMATCHED, CLOSED):
            return
        if self._frozen:
            return  # the detection is suspended for the duration of a freeze
        if self._attempt is not None:
            return  # the clock is stopped for the duration of an attempt anyway
        now = time.monotonic()
        connected = self._stream.connected  # safe to read mid-teardown

        # The freshness clock, from the window the display already fetched. It arms on
        # the first non-empty window and never before: a stream which never delivered a
        # sample has no last acquisition to be late against, and an armed-from-
        # construction clock would declare every silent test outlet stalled.
        ts = self.trace.last_timestamp
        if ts is not None and ts > 0.0 and ts != self._last_ts:
            self._last_ts, self._data_at = ts, now

        if self._state == LIVE:
            if not connected:
                self._enter(
                    INTERRUPTED, disconnect_text(self._stream.disconnect_reason)
                )
            elif (
                self._data_at is not None
                and self.trace.n_rows
                and now - self._data_at > T_STALL
            ):
                # 'n_rows' because an all-hidden display fetches nothing at all, so its
                # freshness clock stops on its own and would otherwise expire.
                self._enter(INTERRUPTED, "No data")
            elif self._resumed_at is not None and not self._in_flap_window(now):
                self._backoff, self._resumed_at = 0, None  # the resume held
            return

        # INTERRUPTED. A tick never re-applies the settings and never declares the
        # document live on its own account: only a reconnection which reported a match
        # arms '_awaiting_data', and that is what keeps a *stall* -- which leaves the
        # stream connected and its stale timestamp above zero -- from resuming itself
        # every tick without any attempt ever being made.
        if self._awaiting_data:
            if not connected:
                self._retry_later()  # lost again before any sample arrived
            elif not self.trace.n_rows or (ts is not None and ts > 0.0):
                # A hidden layout fetches no window, so a resume cannot be confirmed
                # against one -- and there is no live viewport to flash either. Without
                # this term the timestamp is frozen at its pre-outage value, which
                # either never confirms or confirms off a stale number.
                self._go_live(now)
            elif now >= self._next_attempt:
                # The source answered and then said nothing: the hung-sender case
                # liblsl cannot see, which is why this confirmation has a deadline.
                self._retry_later("No data")
        elif self._next_attempt is not None and now >= self._next_attempt:
            self._submit_attempt()

    def _enter(self, state: str, notice: str) -> None:
        """Move to an interrupted state, show its notice and report the move.

        Parameters
        ----------
        state : str
            Either :data:`INTERRUPTED` or :data:`MISMATCHED`.
        notice : str
            The bare reason, e.g. ``'Stream lost'``. The status bar shows it as
            ``Interrupted • <reason>``.

        Notes
        -----
        The level and the retry affordance are derived from the state rather than passed
        in: they are fully determined by it, and a keyword a caller can get wrong is how
        a terminal state ends up without the one button which leaves it.

        The single writer of the *retry* deadline, which is what makes every failure
        path arm exactly one and re-text the notice exactly once -- see
        :meth:`_retry_later`, the one way in. The two other writes of that attribute are
        not retries: the ``None`` of :meth:`_submit_attempt`, which is an attempt
        starting, and the confirmation deadline of :meth:`_on_attempt`, already granted.

        A **borrowed** stream is never retried on a timer, only on
        :meth:`StreamDocument.retry`: reconnecting one in place destroys the filters,
        the callbacks and the acquisition delay its owner set, and a stall fires on a
        source which never went away at all. Its notice therefore offers Retry, which
        for an owned stream would only race the automatic attempt.

        A loss within :data:`T_STABLE` of a resume is re-worded and does **not** reset
        the retry ladder, which is the whole anti-flap rule; a counter is not kept, as
        the timestamp alone produces both required behaviours.
        """
        now = time.monotonic()
        if state == INTERRUPTED and self._in_flap_window(now):
            notice = "Connection unstable"
        self._state = state
        self._notice = notice
        self._awaiting_data = False
        if state == MISMATCHED:
            text, level, retry = notice, "error", True
            self._next_attempt = None  # terminal: nothing is retried on its own
        elif self._owns_stream:
            text = f"{notice} — reconnecting…"
            level, retry = "warning", False
            self._next_attempt = now + _BACKOFF[self._backoff]
        else:
            text, level, retry = notice, "warning", True  # see the note above
            self._next_attempt = None
        banner = self._notice_widget()
        banner.set_notice(text, level=level, retry=retry)
        banner.setVisible(True)
        self._apply_clock()
        self.changed.emit(self)  # the status bar is showing the previous state

    def _go_live(self, now: float) -> None:
        """Declare the document live again, one non-empty window after a resume."""
        self._state = LIVE
        self._notice = ""
        self._awaiting_data = False
        self._resumed_at = now  # the flap window, closed by 'T_STABLE' of good data
        if self._banner is not None:
            self._banner.setVisible(False)
        self._apply_clock()
        self.changed.emit(self)

    def _in_flap_window(self, now: float) -> bool:
        """Return whether a loss right now would still count as a flap.

        Parameters
        ----------
        now : float
            The :func:`time.monotonic` reading of the caller.

        Returns
        -------
        flapping : bool
            Whether the document resumed less than :data:`T_STABLE` seconds ago.

        Notes
        -----
        Read by both sides of the anti-flap rule, and the elapsed test is what makes the
        timestamp a *window* rather than a latch: without it the first flap of an outage
        re-words every later notice as ``'Connection unstable'`` for as long as the
        source stays away, so the real reason -- the one the operator has to act on --
        never reaches the banner again.
        """
        return self._resumed_at is not None and now - self._resumed_at <= T_STABLE

    def _retry_later(self, notice: str | None = None) -> None:
        """Climb one rung of the retry ladder and re-arm through :meth:`_enter`.

        Parameters
        ----------
        notice : str | None
            The bare reason to show, or ``None`` to keep the one already on the banner.

        Notes
        -----
        The one way a failed attempt is recorded, called from every failure path. That
        matters beyond tidiness: the ladder and the deadline are two writes which must
        agree, and going through :meth:`_enter` is what re-texts the notice strip and
        emits :attr:`StreamDocument.changed` -- so the shared status bar learns that a
        resume failed instead of keeping the text of the attempt before it.
        """
        self._backoff = min(self._backoff + 1, len(_BACKOFF) - 1)
        self._enter(INTERRUPTED, self._notice if notice is None else notice)

    def _submit_attempt(self) -> None:
        """Reconnect in the background, with the clock stopped for the whole attempt.

        Notes
        -----
        The display must stop reading the stream: the worker calls ``connect()`` before
        the match is evaluated, so a stream which came back with fewer channels is
        briefly connected while the display still holds the old layout -- and a fetch
        then raises ``IndexError`` from inside a Qt slot, 30 times a second, logging an
        invitation to open a bug report each time. :meth:`_apply_clock` both stops the
        clock and suspends the display for the whole attempt, because stopping the clock
        alone leaves every interaction free to fetch.
        """
        expected = StreamSignature(
            identity=self._identity,
            sfreq=self._sfreq,
            dtype=self._dtype,
            # the recorded wire names, in acquisition order, immune to every rename.
            ch_names=tuple(self.model.acquisition_names()),
        )
        self._next_attempt = None
        self._attempt = submit_reconnect(self._stream, expected, self._on_attempt)
        self._apply_clock()

    def _on_attempt(self, outcome: str, detail: str) -> None:
        """Map the outcome of one reconnection onto a state.

        Parameters
        ----------
        outcome : str
            One of :data:`~mne_lsl.viewer.backend.RESUME_LIVE`,
            :data:`~mne_lsl.viewer.backend.RESUME_MISMATCH` and
            :data:`~mne_lsl.viewer.backend.RESUME_RETRY`.
        detail : str
            The refusal reason, or the text of the exception which failed the attempt.

        Notes
        -----
        A match re-applies the settings here and hands the confirmation to a later tick.
        The display's layout is only valid because the match rule guaranteed the channel
        set, so the re-apply has to happen before the clock comes back -- while the move
        to :data:`LIVE` waits for one non-empty window, so that a source which returns
        and immediately vanishes again does not flash a live viewport.

        That confirmation carries a deadline of its own, armed here: a source which
        answered the identity and then pushed nothing is the hung-sender case
        ``recover=False`` leaves liblsl unable to report, and without the deadline the
        document waits for a window which never comes, repainting an all-NaN viewport at
        30 Hz for the rest of the session.

        The re-apply is guarded because it reads the stream: a source lost again in the
        ~1 ms between the worker's comparison and this slot makes
        :meth:`~mne_lsl.viewer.controller.ChannelModel.refresh` raise, and an escaping
        exception here would leave the clock stopped with no deadline armed, i.e. a
        document which is dead for the rest of the session.
        """
        self._attempt = None
        if self._torn:
            if outcome == RESUME_LIVE and self._owns_stream:
                # The document went away while this was in flight: the worker has just
                # connected a stream nobody will ever draw, and dropping it leaks a live
                # inlet plus its acquisition thread for the life of the process. A
                # borrowed stream is the caller's, and 'teardown' promised to leave it
                # alone.
                release_stream(self._stream)
            return
        now = time.monotonic()
        if outcome == RESUME_LIVE:
            try:
                self._reapply()
            except Exception as error:  # the source died again since the comparison
                logger.warning(
                    "Could not re-apply the settings after a resume: %s", error
                )
                self._retry_later()
                return
            self._awaiting_data = True
            self._last_ts = None  # so the first post-resume window reads as new
            self._next_attempt = now + T_STALL  # the confirmation deadline
            self._apply_clock()  # the clock comes back; the state is still interrupted
            return
        if outcome == RESUME_MISMATCH:
            self._enter(MISMATCHED, detail)
            return
        # RESUME_RETRY: the stream is not back yet. The same banner is re-texted rather
        # than a second one built, which is what keeps a long outage from stacking
        # notices, and the ladder climbs one rung.
        self._retry_later()

    def _reapply(self) -> None:
        """Re-read the stream and put the user's edits back on top of it.

        Notes
        -----
        The order is the invariant. :meth:`capture_state` runs first, while the model
        still holds the edits; :meth:`~mne_lsl.viewer.controller.ChannelModel.refresh`
        then makes the model the wire state again; :meth:`apply_state` makes the saved
        deltas a real difference against it once more. Reversing the last two makes the
        re-apply a no-op against the model's own cache.

        ``refresh()`` re-baselines ``Channel.orig`` **only** when the channel count
        changed, because only then does it rebuild. A changed count is a signature
        mismatch, a mismatch never reaches this method, and this method is the only
        caller of ``refresh()`` on the resume path -- so the re-baselining branch is
        unreachable by construction and not by luck. That matters because a re-baseline
        makes the next :meth:`capture_state` produce empty deltas, i.e. destroys the
        configuration by the act of saving it.
        """
        state = self.capture_state()
        self.model.refresh()
        self.apply_state(state)

    def _notice_widget(self) -> Banner:
        """Return the notice strip, building it on first need.

        Notes
        -----
        One banner per document, ever: it is inserted between the toolbar and the
        splitter and then only re-texted and shown or hidden. Built lazily so that a
        document which never loses its stream never pays for it, and never destroyed so
        that a long outage cannot stack notices.
        """
        if self._banner is None:
            self._banner = Banner(parent=self._content)
            self._banner.retry_clicked.connect(self.retry)
            self._banner.close_clicked.connect(self.closeDockWidget)
            self._content.layout().insertWidget(1, self._banner)
        return self._banner

    def _apply_clock(self) -> None:
        """Push the render clock and the stream access rule onto the display.

        Notes
        -----
        The only writer of :meth:`~mne_lsl.viewer.display.TraceDisplay.start`,
        :meth:`~mne_lsl.viewer.display.TraceDisplay.stop` and
        :meth:`~mne_lsl.viewer.display.TraceDisplay.suspend` in this class. Three
        independent reasons stop the clock and each has to survive the others being
        lifted; two call sites deciding on their own is how a reconnection silently
        undoes a freeze.

        The suspension is a narrower statement than the stopped clock and is therefore
        written separately: an attempt in flight means the stream may not be read *at
        all*, because the worker briefly holds it connected with a channel set the
        display has not been rebuilt against, while a freeze and a refused resume merely
        mean the viewport does not advance. Stopping the clock alone does not keep a
        scroll or a layout change off the stream, which is what
        :meth:`~mne_lsl.viewer.display.TraceDisplay.suspend` exists for.

        :data:`MISMATCHED` stops the clock as well. The stream is released on that path,
        so a running clock would merely spin on the disconnected early return of the
        fetch -- 30 ticks a second for the life of the document, and a viewport whose
        "frozen on the last frame" would then be an accident of that early return rather
        than a decision.

        The toolbar indicator is refreshed from here as well, since every caller of this
        method has just changed something the indicator reports and this is the one seam
        they all pass through: a transition which updated the clock and forgot the
        indicator left the toolbar claiming ``● Live`` over a viewport frozen on its
        last frame, for the rest of the outage.
        """
        if self._torn:
            return
        self.trace.suspend(self._attempt is not None)
        if self._frozen or self._attempt is not None or self._state == MISMATCHED:
            self.trace.stop()
        else:
            self.trace.start()
        self._refresh_freeze_ui()

    # -- the model -> display edge -----------------------------------------------------
    def _push_layout(self) -> None:
        """Push the model's visible channels onto the display, then report it."""
        # read fresh on every emission and never cached: the model is the owner of both
        # the order and the visibility.
        self.trace.set_channel_layout(self.model.visible_acq_indices())
        self.changed.emit(self)  # the status bar's displayed/total count moved

    # -- teardown and recovery ---------------------------------------------------------
    def teardown(self) -> None:
        """Stop the render clock and release the stream, if owned.

        Idempotent: it runs from the Qt-ADS ``closed`` signal, thus for a close from the
        document toolbar, from the dock-area close button and from the window closing,
        and may be called again on a stream which is already disconnected.

        Notes
        -----
        The children go first, so that no render tick can read a stream which is going
        away. A failing ``disconnect()`` is logged and swallowed: a window close tears
        down every document in a loop, and one half-connected stream which raises there
        would otherwise abandon the teardown of all the others.

        The model to display edge is dropped as well, and before the display is closed:
        the model outlives the document's widgets, and a layout push arriving afterwards
        reaches a closed display -- which for a *borrowed* stream, still connected, is a
        freshly fetched window drawn into a widget nobody can see.

        A reconnection in flight is deliberately not waited on: the outcome is emitted
        on a separate object, and :meth:`_on_attempt` returns immediately on a torn
        document after releasing the stream the worker may have just connected. That
        handler is nonetheless not what the release can be left to, because PyQt holds a
        bound-method receiver **weakly**: a document whose wrapper is collected -- which
        every configuration load does, through the purge of the closed dock widgets --
        loses the connection silently, and the stream the worker connected then leaks
        with its acquisition thread. The receiver is therefore re-pointed at a callable
        which does not need this object to survive, the inverse of the usual rule: here
        the emitter is kept alive by the runnable and the document must not be.

        The stream is disconnected without asking whether it is connected.
        :meth:`~mne_lsl.stream.StreamLSL.connect` can raise *after* it opened the inlet:
        the channel-info read and the time correction both follow it, which leaves a
        live, subscribed inlet on a stream reading as disconnected, i.e. exactly the
        state a failed attempt leaves behind. ``disconnect()`` is idempotent and it
        destroys the inlet unconditionally, so the guard bought nothing and cost that
        inlet.
        """
        if self._torn:
            return
        self._torn = True
        self._state = CLOSED
        if self._attempt is not None and self._owns_stream:
            self._attempt.finished.disconnect(self._on_attempt)
            self._attempt.finished.connect(
                lambda outcome, _detail, stream=self._stream: (
                    release_stream(stream) if outcome == RESUME_LIVE else None
                )
            )
        self.trace.polled.disconnect(self._on_polled)
        self.model.layout_changed.disconnect(self._push_layout)
        self.model.metadata_changed.disconnect(self.trace.refresh_metadata)
        self.trace.close()
        self.channels.close()
        if self._owns_stream:
            try:
                self._stream.disconnect()
            except Exception as error:  # deliberately broad, see the note above
                logger.warning(
                    "Could not disconnect the stream %s: %s",
                    self._identity.as_tuple(),
                    error,
                )
