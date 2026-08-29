"""Landing page: discovery progress, the saved configurations and the live streams."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from qtpy.QtCore import QItemSelectionModel, QSize, Qt, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QElidingLabel

from .backend import (
    STATE_AVAILABLE,
    STATE_CHECKING,
    STATE_INVALID,
    STATE_LOADING,
    STATE_UNAVAILABLE_CHANNELS,
    STATE_UNAVAILABLE_NO_MATCH,
)
from .theme import _ICON_PX, icon, theme_controller, tokens

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qtpy.QtGui import QKeyEvent, QMouseEvent

    from .backend import ConfigurationState, StreamDescriptor, StreamIdentity

# Human-readable discovery states, keyed by the tag 'Discovery.progress' emits. Declared
# here because the launcher label and the window's transient status message must read
# the same, and a second table is how the two silently diverge. Read through
# 'progress_text' below, never with '.get(tag, tag)' at the call site: that is the same
# divergence one level down.
PROGRESS_TEXT = {
    "checking": "Checking for streams…",
    "updated": "Updated just now",
    "failed": "Discovery failed — press Refresh to retry",
    "empty": "No streams found",
}
# Column headers of the available-stream table, in order.
_COLUMNS = ("Name", "Type", "Source ID", "Channels", "Rate (Hz)", "Host")
# Maximum width of the centred content column.
_COLUMN_MAX_W = 820
# Default width of the source-ID column, which is a uuid more often than not: sized so
# one fits, interactive so a user can widen it, and never 'ResizeToContents', which a
# long source ID turns into a horizontal scrollbar over every other column.
_SOURCE_ID_W = 220
# The card region scrolls past this height, so that a long list of saved configurations
# cannot push the available-stream table off the page.
#
# ponytail: 30 configurations is a scroll and not a search. The upgrade is a filter
# field above the region, which is also the point where a manage dialog would start to
# earn its keep -- one feature, later, rather than two now.
_CARDS_MAX_H = 260

# Presentation of one card per availability state: the group it sorts into, its glyph,
# the word which carries the state without relying on colour, and the semantic colour
# token which is the *additional* cue. One table rather than four, so the four cannot
# drift apart, keyed by the state constants the availability check emits.
#
# The three settled top-group states plus 'loading' sort above the ones which did not
# identity-match: 'unavailable-channels' *did* match, and burying it would hide the
# exact reason the whole channel probe exists to produce.
#
# ponytail: the 'checking' glyph is static text and not an animation. The upgrade is
# 'qtawesome.icon(..., animation=qtawesome.Spin(widget))', worth it only if the check
# gets slow enough to look stuck; today it lasts about half a second.
_CARD_STYLE: dict[str, tuple[int, str, str, str]] = {
    STATE_LOADING: (0, "mdi6.progress-clock", "Opening", "accent"),
    STATE_CHECKING: (0, "mdi6.progress-clock", "Checking", "text_secondary"),
    STATE_AVAILABLE: (0, "mdi6.check-circle-outline", "Available", "success"),
    STATE_UNAVAILABLE_CHANNELS: (0, "mdi6.lock-outline", "Unavailable", "warning"),
    STATE_UNAVAILABLE_NO_MATCH: (
        1,
        "mdi6.lock-outline",
        "Unavailable",
        "text_secondary",
    ),
    STATE_INVALID: (2, "mdi6.alert-outline", "Invalid", "error"),
}
# Presentation of a state this page does not know. Unreachable through the availability
# check, which only ever emits the six above, and present because both consumers are Qt
# slots where a 'KeyError' becomes an unhandled exception -- the same reason
# 'progress_text' resolves an unknown tag instead of indexing.
_CARD_UNKNOWN = (1, "mdi6.help-circle-outline", "Unknown", "text_secondary")


def progress_text(tag: str) -> str:
    """Return the human-readable text of a discovery state ``tag``.

    Parameters
    ----------
    tag : str
        State tag, as ``Discovery.progress`` emits it.

    Returns
    -------
    text : str
        The text of the tag, or the tag itself when it is unknown: both consumers are Qt
        slots, where a :class:`KeyError` becomes an unhandled exception, and a tag with
        no text of its own is still more informative than nothing.
    """
    return PROGRESS_TEXT.get(tag, tag)


def _card_key(card: ConfigurationCard) -> tuple[int, str]:
    """Return the presentation key of ``card``: its group, then its casefolded name.

    Parameters
    ----------
    card : ConfigurationCard
        The card to place.

    Returns
    -------
    key : tuple
        The sort key. Casefolded, so that two names differing only by case cannot swap
        places between two passes of an unchanged directory.
    """
    return (_CARD_STYLE.get(card.state, _CARD_UNKNOWN)[0], card.name.casefold())


class ConfigurationCard(QFrame):
    """One saved configuration, as a persistent widget updated in place.

    Parameters
    ----------
    name : str
        Name of the configuration, which is also the title and the payload of every
        signal below.
    parent : QWidget | None
        Parent widget.

    Attributes
    ----------
    open_requested : Signal
        Emitted with the configuration name when an *available* card is activated, by a
        click or from the keyboard.
    rename_requested : Signal
        Emitted with the configuration name when the rename button is pressed.
    delete_requested : Signal
        Emitted with the configuration name when the delete button is pressed.

    Notes
    -----
    Passive throughout: it renders the state it is handed, reads no file, and never
    evaluates its own availability. The name, and not an index, is the payload of every
    signal -- the cards are reordered on every pass, so an index would open, rename or
    delete a different configuration than the one which was pressed.
    """

    open_requested = Signal(str)
    rename_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        """Initialize the card."""
        super().__init__(parent)
        self._name = name
        self._state = STATE_CHECKING
        self._glyph_name = _CARD_STYLE[STATE_CHECKING][1]
        self._activatable = False
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._glyph = QLabel()
        self._title = QLabel(name)
        self._title.setTextFormat(Qt.TextFormat.PlainText)
        font = QFont(self._title.font())
        font.setBold(True)
        self._title.setFont(font)
        self._word = QLabel()
        self._word.setTextFormat(Qt.TextFormat.PlainText)
        self._reason = QElidingLabel()
        # A configuration name reaches this line through the unavailability reasons, and
        # a stream name -- remote input -- reaches it too: a 'QLabel' is 'AutoText' by
        # default, where one named '<!--' erases the rest of the line. The tooltip is
        # escaped instead, as a tooltip always auto-detects rich text.
        self._reason.setTextFormat(Qt.TextFormat.PlainText)
        self._rename = self._build_button(
            "mdi6.rename-box", "Rename", "Rename this configuration"
        )
        self._rename.clicked.connect(lambda: self.rename_requested.emit(self._name))
        self._delete = self._build_button(
            "mdi6.trash-can-outline", "Delete", "Delete this configuration"
        )
        self._delete.clicked.connect(lambda: self.delete_requested.emit(self._name))

        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(self._glyph)
        head.addWidget(self._title)
        head.addWidget(self._word)
        head.addStretch(1)
        head.addWidget(self._rename)
        head.addWidget(self._delete)
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 6, 8, 6)
        column.setSpacing(2)
        column.addLayout(head)
        column.addWidget(self._reason)
        self.retint_icons()

    def _build_button(self, glyph: str, text: str, tooltip: str) -> QToolButton:
        """Build one icon-only card action, named for a screen reader.

        Parameters
        ----------
        glyph : str
            QtAwesome icon name.
        text : str
            Accessible name of the button, since it carries no visible label.
        tooltip : str
            Tooltip of the button.

        Returns
        -------
        button : QToolButton
            The button, whose icon is baked by :meth:`retint_icons`.
        """
        button = QToolButton()
        button.setAutoRaise(True)
        button.setIconSize(QSize(_ICON_PX, _ICON_PX))
        button.setAccessibleName(text)
        button.setToolTip(tooltip)
        # the glyph is replayed on a theme flip, thus it is kept next to the button.
        button.setProperty("glyph", glyph)
        return button

    @property
    def name(self) -> str:
        """Name of the configuration this card renders."""
        return self._name

    @property
    def state(self) -> str:
        """Availability state this card was last given."""
        return self._state

    @property
    def activatable(self) -> bool:
        """Whether activating this card opens its configuration."""
        return self._activatable

    def set_state(
        self, state: str, reason: str, n_streams: int, *, enabled: bool
    ) -> None:
        """Render one availability state.

        Parameters
        ----------
        state : str
            One of the availability state constants.
        reason : str
            One-line explanation, empty for an available configuration.
        n_streams : int
            Number of streams the configuration requires, shown when it is available.
        enabled : bool
            Whether the card can be activated. Computed by the page, in one place, so
            that a load disables every card with no second entry point.

        Notes
        -----
        Activation is gated by :attr:`activatable` rather than by ``setEnabled`` on the
        frame, because Qt disables the whole subtree of a disabled widget: a dead card
        would take its own Delete button down with it, and deleting the file is the only
        way to clear a corrupt configuration from the interface. The state is carried by
        the glyph, the word and the reason, with the semantic colour as an extra cue and
        the pointer shape as the affordance, so nothing here depends on colour alone.

        Rename is disabled for an invalid configuration, whose own name may be the thing
        which could not be read, and while one is opening. Delete is never disabled.
        """
        self._state = state
        self._activatable = bool(enabled)
        _, self._glyph_name, word, _ = _CARD_STYLE.get(state, _CARD_UNKNOWN)
        self._word.setText(word)
        if state == STATE_AVAILABLE and not reason:
            reason = f"{n_streams} stream{'' if n_streams == 1 else 's'}"
        self._reason.setText(reason)
        # elided in place, thus the tooltip is the only way back to the full text.
        self._reason.setToolTip(escape(reason))
        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus if enabled else Qt.FocusPolicy.NoFocus
        )
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self._rename.setEnabled(state not in (STATE_INVALID, STATE_LOADING))
        self.setAccessibleName(f"{self._name}, {word}")
        self.setAccessibleDescription(reason)
        self.retint_icons()

    def retint_icons(self) -> None:
        """Rebuild the glyphs and the state colour for the active theme.

        Notes
        -----
        A :class:`~qtpy.QtGui.QIcon` bakes its colour at creation, thus a theme flip
        needs every glyph of this card rebuilt rather than merely repainted.
        """
        palette = tokens(theme_controller.mode)
        token = _CARD_STYLE.get(self._state, _CARD_UNKNOWN)[3]
        colour = getattr(palette, token)
        self._glyph.setPixmap(icon(self._glyph_name, color=colour).pixmap(_ICON_PX))
        self._word.setStyleSheet(f"color: {colour};")
        self._reason.setStyleSheet(f"color: {palette.text_secondary};")
        for button in (self._rename, self._delete):
            button.setIcon(icon(button.property("glyph")))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Open the configuration when an activatable card is released inside itself.

        Notes
        -----
        The release has to land inside the widget, as a button's does: a press which
        travelled outside before being let go is a cancelled gesture, and a load is not
        something to start by accident.
        """
        super().mouseReleaseEvent(event)
        if self._activatable and self.rect().contains(event.position().toPoint()):
            self.open_requested.emit(self._name)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Open the configuration on Return, Enter or Space, as a button would."""
        keys = (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)
        if self._activatable and event.key() in keys:
            self.open_requested.emit(self._name)
            return
        super().keyPressEvent(event)


class EmptyStatePage(QWidget):
    """Compact launcher shown while no stream document is open.

    A centred, max-width column: a title, the saved configurations and the available
    regular streams, with a multi-selection. The page owns no button: it reports its
    selection and a double click on it, and the window owns the Open action. The page is
    passive throughout -- the window resolves the identities, performs the connections,
    reads the configuration directory and evaluates every card state it is handed here.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget.

    Attributes
    ----------
    selection_changed : Signal
        Emitted when the stream multi-selection changes, so the window can enable or
        disable its 'Open selected' action.
    open_requested : Signal
        Emitted with the ``list`` of selected
        :class:`~mne_lsl.viewer.backend.StreamDescriptor` to open.
    open_configuration_requested : Signal
        Emitted with the name of the configuration whose card was activated.
    rename_configuration_requested : Signal
        Emitted with the name of the configuration whose rename button was pressed.
    delete_configuration_requested : Signal
        Emitted with the name of the configuration whose delete button was pressed.

    Notes
    -----
    The three configuration signals carry a ``_requested`` suffix rather than the names
    of the verbs they ask for. The names of the persistence functions are on this
    module's forbidden list -- the source-level check which pins the page's passivity --
    and a signal named after one of them would collide with that check for no reason,
    since a syntax-tree scan cannot tell a signal definition from a call.
    """

    selection_changed = Signal()
    open_requested = Signal(object)
    open_configuration_requested = Signal(str)
    rename_configuration_requested = Signal(str)
    delete_configuration_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the page."""
        super().__init__(parent)
        # The regular descriptors, in table-row order: this list *is* the row to
        # descriptor map. Nothing is stashed in an item data role, as what a 'QVariant'
        # round trip does to a frozen dataclass differs between the two Qt bindings.
        self._descriptors: list[StreamDescriptor] = []
        # One persistent card per configuration name, updated in place. Never torn down
        # and rebuilt per pass: a rebuild would drop keyboard focus in the middle of a
        # check and make the move-to-the-top transition impossible.
        self._cards: dict[str, ConfigurationCard] = {}

        title = QLabel("Open live streams")
        font = QFont(title.font())
        # font-relative rather than a pixel size, so the title follows the system font.
        font.setPointSizeF(font.pointSizeF() * 1.6)
        font.setBold(True)
        title.setFont(font)
        subtitle = QLabel(
            "Click a stream to select it, click again to deselect. Each selected "
            "stream opens its own document."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid);")

        self._progress = QLabel()
        self._progress.setStyleSheet("color: palette(mid); font-style: italic;")

        self._table = self._build_table()
        self._events_label = QElidingLabel()
        # A stream name is remote input and this is a 'QLabel', i.e. 'AutoText': one
        # named '<!--' erases every name after it from the line, '<b>x</b>' shows as 'x'
        # and an '<img>' embeds a local file. The table is immune, as an item renders
        # its text verbatim, and the tooltip is escaped below -- a tooltip always
        # auto-detects rich text and has no format of its own.
        self._events_label.setTextFormat(Qt.TextFormat.PlainText)
        self._events_label.setStyleSheet("color: palette(mid);")

        content = QWidget()
        content.setMaximumWidth(_COLUMN_MAX_W)
        column = QVBoxLayout(content)
        column.setSpacing(8)
        column.addWidget(title)
        column.addWidget(subtitle)
        column.addWidget(self._progress)
        # the saved workspaces come first: reopening one is the gesture a returning user
        # is here for, while picking streams by hand is how a new one is built.
        column.addWidget(self._build_cards_region())
        column.addWidget(self._table, 1)
        column.addWidget(self._events_label)
        # keeps the empty page top-aligned: with the table hidden, this spacer is what
        # absorbs the height instead of three labels spreading down 800 px.
        column.addStretch()

        outer = QHBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        # Stretch 0 on the spacers and 1 on the column: a stretch>0 spacer takes the
        # extra width before the column does, which is what pinned the whole page to a
        # ~300 px sizeHint and left '_COLUMN_MAX_W' unreachable. The spacers keep their
        # 'Expanding' policy, so they still centre the column once it clamps.
        outer.addStretch()
        outer.addWidget(content, 1)
        outer.addStretch()

        self.set_progress("checking")  # never an empty label before the first pass
        self.set_streams([])  # the fallback text of the event line, from the same path
        self.set_configurations([])  # hides the whole region, as on a first launch

    def _build_cards_region(self) -> QGroupBox:
        """Build the scrollable region holding the saved-configuration cards."""
        self._cards_host = QWidget()
        self._cards_box = QVBoxLayout(self._cards_host)
        self._cards_box.setContentsMargins(0, 0, 0, 0)
        self._cards_box.setSpacing(6)
        self._cards_box.addStretch(1)
        self._cards_scroll = QScrollArea()
        self._cards_scroll.setWidgetResizable(True)
        self._cards_scroll.setWidget(self._cards_host)
        self._cards_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._cards_scroll.setMaximumHeight(_CARDS_MAX_H)
        self._cards_group = QGroupBox("Saved configurations")
        box = QVBoxLayout(self._cards_group)
        box.setSpacing(6)
        box.addWidget(self._cards_scroll)
        return self._cards_group

    def set_configurations(
        self, states: Sequence[ConfigurationState], *, loading: bool = False
    ) -> None:
        """Render one card per saved configuration, updating the existing ones in place.

        Parameters
        ----------
        states : sequence of ConfigurationState
            The rendered availability of every saved configuration, as the window
            evaluated it. An empty sequence hides the whole region.
        loading : bool
            Whether a configuration is currently being opened, in which case no card is
            activatable. Passed in rather than inferred from ``states``: a card carrying
            the loading state is the only evidence a rendered list holds, so deleting
            the configuration being opened would erase it and re-activate every sibling
            while the load is still running -- and activating one is then a silent
            no-op.

        Notes
        -----
        Cards are persistent widgets: a name which has no card gets one, a name which
        vanished loses its card, and the rest are updated in place. Rebuilding the
        region on every pass would drop keyboard focus mid-check and make the
        move-to-the-top transition impossible.

        Activatability is computed here, in one place: a card opens only while it is
        available and **no** configuration is loading. That is how 'every other card is
        disabled during a load' happens with no second entry point and no window-side
        gating of this page.

        The order is ``(group, casefolded name)`` and nothing else. A stable order is
        what keeps a card from jumping under the pointer between two passes, and this
        page never compares a state against a literal outside its presentation table.
        """
        wanted = {state.name: state for state in states}
        for name in tuple(self._cards):
            if name not in wanted:
                card = self._cards.pop(name)
                self._cards_box.removeWidget(card)
                # hidden as well: 'removeWidget' takes a widget out of the layout and
                # leaves it shown at its last geometry, so the card of a configuration
                # which vanished keeps painting over its neighbours until the deferred
                # deletion is delivered.
                card.hide()
                card.deleteLater()
        for name, state in wanted.items():
            card = self._cards.get(name)
            if card is None:
                card = ConfigurationCard(name, self._cards_host)
                card.open_requested.connect(self.open_configuration_requested)
                card.rename_requested.connect(self.rename_configuration_requested)
                card.delete_requested.connect(self.delete_configuration_requested)
                self._cards[name] = card
                self._cards_box.insertWidget(self._cards_box.count() - 1, card)
            card.set_state(
                state.state,
                state.reason,
                state.n_streams,
                enabled=state.state == STATE_AVAILABLE and not loading,
            )
        for index, card in enumerate(sorted(self._cards.values(), key=_card_key)):
            self._cards_box.insertWidget(index, card)
        self._cards_group.setVisible(bool(wanted))

    def configuration_names(self) -> tuple[str, ...]:
        """Return the card names in the order the region shows them.

        Returns
        -------
        names : tuple of str
            The configuration names, grouped by availability and alphabetical inside a
            group. A test seam and nothing else: the window keeps its own list.
        """
        return tuple(card.name for card in sorted(self._cards.values(), key=_card_key))

    def retint_icons(self) -> None:
        """Rebuild the baked glyphs of every card for the active theme."""
        for card in self._cards.values():
            card.retint_icons()

    def _build_table(self) -> QTableWidget:
        """Build the available-stream table."""
        table = QTableWidget(0, len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # A plain left click toggles one row, with no modifier: the page exists to build
        # a multi-selection, and 'ExtendedSelection' makes the second stream cost a
        # Cmd-click nobody discovers. The cost is that Shift+click no longer extends a
        # range -- it toggles the clicked row like any other click. Note that this also
        # makes every *programmatic* view-level call a toggle: 'selectRow' and
        # 'setCurrentIndex' no longer clear first, which is why the page selects through
        # the selection model with explicit flags instead.
        #
        # ponytail: a checkbox column is the upgrade if the toggle proves undiscoverable
        # in use; it adds a check state the selection highlight can disagree with.
        table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = table.horizontalHeader()
        # Stretch on Name rather than 'setStretchLastSection', which hands the slack to
        # Host and leaves Name at a fixed default -- the permanent horizontal scrollbar
        # this replaces.
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 3, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(2, _SOURCE_ID_W)
        table.itemSelectionChanged.connect(self.selection_changed)
        table.itemDoubleClicked.connect(self._on_double_click)
        return table

    def set_streams(self, descriptors: Sequence[StreamDescriptor]) -> None:
        """Populate the table with the regular streams and note the event sources.

        Parameters
        ----------
        descriptors : sequence of StreamDescriptor
            Every descriptor found by the last discovery pass. Descriptors with a null
            sampling rate are listed as event sources rather than as openable streams.

        Notes
        -----
        The selection is re-anchored on the identities it held. It lives in table *row*
        indices, thus a pass which merely inserted or removed one stream would leave the
        highlight on a different stream than the user picked -- and the window would
        then connect that one, with nothing on screen signalling the switch. An identity
        which the pass no longer reports simply loses its selection.

        The table's own signals are blocked for the whole rebuild, which is what makes
        the single emission at the end of this method the *only* one: Qt reports the
        rows it drops out of a live selection, and re-anchoring is one ``select()`` per
        matching row, so the window would otherwise re-evaluate its Open action -- and
        re-materialize every selected index to do it -- once per selected stream, on
        every discovery pass.
        """
        regular = [descriptor for descriptor in descriptors if descriptor.sfreq != 0]
        events = [descriptor for descriptor in descriptors if descriptor.sfreq == 0]
        selected = {descriptor.identity for descriptor in self.selected_descriptors()}
        blocked = self._table.blockSignals(True)
        # Hidden for the fill, and shown again below: the four 'ResizeToContents'
        # columns recompute on every 'setItem', and each walks *every* row, so a
        # visible table rebuilds in time quadratic in the row count -- 51 ms at 30
        # streams and 531 ms at 100, against 0.7 and 2.1 ms hidden. Blocking the signals
        # does not cover it: a header resize is not a signal.
        self._table.setVisible(False)
        try:
            self._descriptors = regular
            self._table.setRowCount(len(regular))
            for row, descriptor in enumerate(regular):
                identity = descriptor.identity
                values = (
                    identity.name,
                    identity.stype,
                    identity.source_id,
                    str(descriptor.n_channels),
                    f"{descriptor.sfreq:g}",
                    descriptor.hostname,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    # a tooltip always auto-detects rich text whatever the item was
                    # told, and a stream name is remote input -- hence escaped.
                    item.setToolTip(escape(value))
                    if column == 0:
                        font = QFont(item.font())
                        font.setBold(True)
                        item.setFont(font)
                    self._table.setItem(row, column, item)
            self._select_identities(selected)
        finally:
            self._table.blockSignals(blocked)
        # A 0-row grid states nothing the progress label right above does not already
        # say -- "Checking for streams…", "No streams found", or the failure.
        self._table.setVisible(bool(regular))
        if events:
            names = ", ".join(descriptor.identity.name for descriptor in events)
            # 'Discovery' calls a pass which found only event sources 'updated', since
            # it counts them as streams, so the progress label reads "Updated just now"
            # over a table that just vanished. This line is the only one which knows the
            # split, thus it is the one which says so.
            found = "Event sources" if regular else "Only event sources found"
            text = f"{found} (not openable as a document): {names}"
        else:
            text = ""  # hidden below, thus only cleared: no stale name behind the line
        self._events_label.setText(text)
        # Elided in place, thus the tooltip is the only way back to the full text -- and
        # escaped, as a tooltip renders markup whatever the label was told.
        self._events_label.setToolTip(escape(text))
        # shown only when there is a name to give
        self._events_label.setVisible(bool(events))
        # The table was rebuilt under the selection, thus the window has to re-evaluate
        # its Open action even though no user interaction took place.
        self.selection_changed.emit()

    def _select_identities(self, identities: set[StreamIdentity]) -> None:
        """Select the rows whose descriptor carries one of ``identities``.

        Parameters
        ----------
        identities : set of StreamIdentity
            Identities to select; one absent from the table selects nothing.
        """
        model = self._table.selectionModel()
        flags = (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        )
        self._table.clearSelection()
        for row, descriptor in enumerate(self._descriptors):
            if descriptor.identity in identities:
                model.select(self._table.model().index(row, 0), flags)

    def set_progress(self, tag: str) -> None:
        """Reflect a discovery state tag, e.g. ``'checking'``, in the progress label.

        Parameters
        ----------
        tag : str
            The state tag, resolved by :func:`progress_text`.
        """
        self._progress.setText(progress_text(tag))

    def selected_descriptors(self) -> list[StreamDescriptor]:
        """Return the descriptors of the currently selected regular streams.

        Returns
        -------
        descriptors : list of StreamDescriptor
            The selected descriptors, in table-row order and without duplicates -- a
            selected row reports one index per column.
        """
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        # Cross-binding insurance, and not a reachable path under PyQt6: measured, Qt
        # clears the selection of the rows it removes *before* delivering
        # 'itemSelectionChanged', so no index past the descriptors can arrive here. Kept
        # because PySide6 is not exercised locally and the failure there would be an
        # 'IndexError' raised inside a Qt slot rather than a missing selection.
        return [self._descriptors[row] for row in rows if row < len(self._descriptors)]

    def _on_double_click(self, *_args) -> None:
        """Request the selected streams to be opened, if any."""
        descriptors = self.selected_descriptors()
        if descriptors:
            self.open_requested.emit(descriptors)
