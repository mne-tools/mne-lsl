from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import QEvent, QItemSelectionModel, QPointF, Qt
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QAbstractItemView

import mne_lsl.viewer._launcher
from mne_lsl.viewer._launcher import (
    PROGRESS_TEXT,
    ConfigurationCard,
    EmptyStatePage,
)
from mne_lsl.viewer.backend import (
    STATE_AVAILABLE,
    STATE_CHECKING,
    STATE_INVALID,
    STATE_LOADING,
    STATE_UNAVAILABLE_CHANNELS,
    STATE_UNAVAILABLE_NO_MATCH,
    ConfigurationState,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType

    from qtpy.QtWidgets import QApplication

    from mne_lsl.viewer.backend import StreamDescriptor
    from mne_lsl.viewer.theme import ThemeController

# Nothing the page may name: it lists what it is given, reports what it is told and
# resolves nothing itself. This is the property the whole design rests on -- the window
# owns discovery, the connections and the configurations. It is why the page's three
# configuration signals carry a '_requested' suffix: a signal named after one of the
# persistence verbs below would trip this check, which cannot tell a signal definition
# from a call.
_FORBIDDEN = frozenset(
    {
        "Connector",
        "Discovery",
        "Prober",
        "ViewerConfig",
        "config_dir",
        "connect_stream",
        "create_stream",
        "delete_configuration",
        "evaluate_state",
        "list_configurations",
        "probe_channels",
        "rename_configuration",
        "resolve_descriptors",
        "save_configuration",
    }
)


@pytest.fixture
def page(flush_deletes: Callable[..., None]) -> Generator[EmptyStatePage]:
    """Yield a landing page, closed and deleted afterwards."""
    built = EmptyStatePage()
    built.resize(900, 700)
    yield built
    built.close()
    flush_deletes(built)


def _select(page: EmptyStatePage, *rows: int) -> None:
    """Select ``rows`` of the stream table, in the order given.

    Through the selection model with explicit flags, which is what the page itself does:
    it bypasses the selection *mode*, whereas 'selectRow' obeys it and so toggles under
    'MultiSelection' -- selecting the same row twice would deselect it.
    """
    table = page._table
    table.clearSelection()
    flags = (
        QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows
    )
    for row in rows:
        table.selectionModel().select(table.model().index(row, 0), flags)


def _shown(page: EmptyStatePage, width: int, height: int) -> None:
    """Show ``page`` offscreen at one size, with its layout applied.

    A layout which was never activated reports the geometry a widget had before it was
    ever shown, thus every geometry assertion below would read a stale number -- and
    'visualRect' an empty rectangle. Offscreen and with no event loop: the platform
    comes from this package's conftest and the fixture closes the page.
    """
    page.resize(width, height)
    page.show()
    page.layout().activate()


def _click_row(page: EmptyStatePage, row: int) -> None:
    """Left-click the first cell of ``row``, with no modifier, as a user would.

    Through 'QTest' rather than the selection model, which ignores the selection *mode*:
    the mode is precisely what decides whether a plain click toggles the row or replaces
    the whole selection with it.
    """
    table = page._table
    QTest.mouseClick(
        table.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        table.visualRect(table.model().index(row, 0)).center(),
    )


def test_empty_page(page: EmptyStatePage) -> None:
    """Test that a fresh page shows no stream and is never blank while it waits."""
    assert page._table.rowCount() == 0
    assert page.selected_descriptors() == []
    assert page._progress.text() == PROGRESS_TEXT["checking"]
    assert page._events_label.text() == ""


def test_set_streams_lists_regular_only(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that only the regularly sampled streams are openable rows.

    An event source cannot own a document -- 'StreamLSL' refuses to connect to one with
    a buffer in seconds -- thus listing it as a row would offer an action which always
    fails.
    """
    regular = [descriptor(name=f"regular-{index}") for index in range(3)]
    events = [descriptor(name=f"marker-{index}", sfreq=0.0) for index in range(2)]
    page.set_streams([*events, *regular])
    assert page._table.rowCount() == 3
    assert [page._table.item(row, 0).text() for row in range(3)] == [
        "regular-0",
        "regular-1",
        "regular-2",
    ]
    for name in ("marker-0", "marker-1"):
        assert name in page._events_label.text()
    assert page._events_label.toolTip() == page._events_label.text()
    page.set_streams(regular)
    assert page._events_label.text() == ""


def test_set_streams_columns(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that each cell holds its own field, so a column swap cannot pass.

    Nothing else in the suite would catch the type and the source ID trading places.
    """
    entry = descriptor(
        name="Polar",
        stype="ecg",
        source_id="unit-7",
        n_channels=3,
        sfreq=130.5,
        hostname="host-9",
    )
    page.set_streams([entry])
    assert [page._table.item(0, column).text() for column in range(6)] == [
        "Polar",
        "ecg",
        "unit-7",
        "3",
        "130.5",
        "host-9",
    ]
    assert page._table.item(0, 0).font().bold()


def test_set_streams_rebuild(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a shorter pass leaves no stale row and no stale descriptor.

    A descriptor kept past the pass which removed it is a stream the window would try to
    open although discovery no longer sees it.
    """
    first = [descriptor(name=f"stream-{index}") for index in range(3)]
    page.set_streams(first)
    _select(page, 2)
    assert page.selected_descriptors() == [first[2]]
    page.set_streams(first[:1])
    assert page._table.rowCount() == 1
    assert page.selected_descriptors() == []


def test_set_streams_keeps_the_selected_identity(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a discovery pass re-anchors the selection on the identity it held.

    The selection lives in table *row* indices, thus a pass which merely inserted one
    stream shifts every later row under it: the user picks C, a colleague's B appears,
    and the window connects B with the highlight never moving. Asserted for an insertion
    and for a same-length rebuild, as the row count alone cannot detect either.
    """
    a, b, c = (descriptor(name=name) for name in ("A", "B", "C"))
    page.set_streams([a, c])
    _select(page, 1)
    assert page.selected_descriptors() == [c]
    page.set_streams([a, b, c])  # a colleague's stream appeared meanwhile
    assert page.selected_descriptors() == [c]
    assert sorted({index.row() for index in page._table.selectedIndexes()}) == [2]
    page.set_streams([c, a, b])  # same length, every row holds another stream
    assert page.selected_descriptors() == [c]


def test_event_source_names_are_never_markup(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a stream name reaches the event line as text and never as markup.

    A name is remote input and the line is a 'QLabel', i.e. 'AutoText' by default: a
    source named '<!--' erases every name after it from what is rendered, and '<b>x</b>'
    shows as 'x', so the name on screen is not the one published. A tooltip has no
    format of its own and always auto-detects rich text, hence the escaping -- which the
    table item needs too, as its own tooltip is the only way back to an elided name.
    """
    page.set_streams(
        [
            descriptor(name="<!--", sfreq=0.0),
            descriptor(name="<b>x</b>", sfreq=0.0),
            descriptor(name="<b>x</b>"),
        ]
    )
    assert page._events_label.textFormat() == Qt.TextFormat.PlainText
    assert "<!--" in page._events_label.text()
    assert "<b>x</b>" in page._events_label.text()
    assert "&lt;!--" in page._events_label.toolTip()
    assert "&lt;b&gt;x&lt;/b&gt;" in page._events_label.toolTip()
    assert "<b>" not in page._events_label.toolTip()
    assert page._table.item(0, 0).text() == "<b>x</b>"
    assert page._table.item(0, 0).toolTip() == "&lt;b&gt;x&lt;/b&gt;"


def test_selected_descriptors_row_order(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that the selection comes back in row order, once per row, as passed in.

    'selectedIndexes()' reports one index per cell, thus six entries per selected row
    and in click order; and indexing the unsplit list would return an event source.

    The selection *mode* is asserted here because the helper below drives the selection
    model, which ignores it: a table left on 'SingleSelection' still reports two rows
    selected to every test of this module while a user could only ever pick one.
    """
    events = [descriptor(name="marker", sfreq=0.0)]
    regular = [descriptor(name=f"stream-{index}") for index in range(3)]
    page.set_streams([*events, *regular])
    assert page._table.selectionMode() == QAbstractItemView.SelectionMode.MultiSelection
    _select(page, 2, 0)
    selected = page.selected_descriptors()
    assert selected == [regular[0], regular[2]]
    assert selected[0] is regular[0]
    assert selected[1] is regular[2]


def test_selection_changed_emitted(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a selection, and a rebuilt table, both report themselves.

    Without the first the 'Open selected' action stays disabled forever; without the
    second the window never re-evaluates it over the table a discovery pass replaced.

    The rebuild is asserted first with **nothing selected**, and at an unchanged row
    count: Qt emits 'itemSelectionChanged' by itself whenever it drops rows out of a
    selection -- measured twice for a rebuild under a live selection, never for one
    under an empty selection -- so that shape alone would pass with the page's own
    emission deleted.

    Then under a live selection, which is where those extra emissions come from and
    where the re-anchoring adds one more per matching row: the count is still exactly
    one, since the rebuild blocks the table's own signals. Without that the window
    re-evaluates its Open action -- re-materializing every selected index to do it --
    once per selected stream on every discovery pass.
    """
    page.set_streams([descriptor(name="stream-0"), descriptor(name="stream-1")])
    seen: list[int] = []
    page.selection_changed.connect(lambda: seen.append(1))
    _select(page, 1)
    assert len(seen) == 1
    page._table.clearSelection()
    seen.clear()
    page.set_streams([descriptor(name="other-0"), descriptor(name="other-1")])
    assert len(seen) == 1

    _select(page, 0, 1)
    seen.clear()
    page.set_streams([descriptor(name="other-0"), descriptor(name="other-1")])
    assert len(seen) == 1


def test_open_requested_on_double_click(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a double click opens the selection, and nothing when there is none.

    An empty emission would make the window start a batch of zero connections and show a
    'Connecting to 0 stream(s)' message.
    """
    regular = [descriptor(name=f"stream-{index}") for index in range(2)]
    page.set_streams(regular)
    seen: list[object] = []
    page.open_requested.connect(seen.append)
    page._table.clearSelection()
    page._table.itemDoubleClicked.emit(page._table.item(0, 0))
    assert seen == []
    _select(page, 1)
    page._table.itemDoubleClicked.emit(page._table.item(1, 0))
    assert seen == [[regular[1]]]


def test_set_progress_tags(page: EmptyStatePage) -> None:
    """Test that every discovery tag maps to its own text, and an unknown one to itself.

    A 'KeyError' here would surface from inside a Qt slot, and two tags collapsing to
    one string would make a failed pass indistinguishable from a finished one.
    """
    texts = set()
    for tag in ("checking", "updated", "failed", "empty"):
        page.set_progress(tag)
        assert page._progress.text() == PROGRESS_TEXT[tag]
        assert page._progress.text()
        texts.add(page._progress.text())
    assert len(texts) == 4
    page.set_progress("some-new-tag")
    assert page._progress.text() == "some-new-tag"


def test_launcher_is_passive(
    module_scan: Callable[[ModuleType], tuple[set[str], set[str]]],
) -> None:
    """Test that the landing page resolves, connects and reads nothing itself.

    A source-level check, as importing any viewer module necessarily imports the LSL
    layer. The identifiers come from the syntax tree, thus a docstring naming the
    window's responsibilities is documentation and not a dependency.
    """
    imports, identifiers = module_scan(mne_lsl.viewer._launcher)
    segments = {segment for path in imports for segment in path.split(".")}
    assert (segments | identifiers).isdisjoint(_FORBIDDEN)


def test_content_column_fills_the_window(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that the centred column grows to its maximum instead of to its size hint.

    A widget added with stretch 0 between two stretch-1 spacers never gets a pixel of
    the extra width: the spacers take it first, so the column stays at its ~300 px size
    hint whatever maximum it was given, and the table inside it is horizontally scrolled
    forever. The column is asserted to stop at its maximum rather than to fill the
    window, since a full-bleed table leaves the title stranded at 1400 px.
    """
    page.set_streams([descriptor(name="stream-0")])
    _shown(page, 1400, 850)
    assert page._table.width() > 700, page._table.width()
    assert page._table.parentWidget().width() == mne_lsl.viewer._launcher._COLUMN_MAX_W


def test_every_column_is_visible(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that all six columns fit, with the name absorbing the slack.

    A source ID is a uuid more often than not, so sizing that column to its content
    pushes the last columns out of the view -- and 'setStretchLastSection' hands the
    slack to Host, leaving the one column a user reads the row by at its fixed default.
    The scroll *range* is what carries "no horizontal scrollbar": the visibility flag of
    a scroll bar lags an event-loop pass behind a resize, which this suite never runs.
    """
    source_id = "0f9a0f2e-6c4f-4a5e-9a1a-2b3c4d5e6f70"
    page.set_streams([descriptor(name="stream-0", source_id=source_id)])
    _shown(page, 1400, 850)
    header = page._table.horizontalHeader()
    assert page._table.horizontalScrollBar().maximum() == 0
    assert header.sectionSize(0) > header.sectionSize(5), header.sectionSize(0)
    # the ID keeps its fixed default: sizing it to a uuid is what eats the row
    assert header.sectionSize(2) == mne_lsl.viewer._launcher._SOURCE_ID_W


def test_table_fills_the_page_and_hides_when_empty(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that the table takes the free height, and vanishes with nothing to show.

    A height-capped table leaves the bottom half of the page empty on any real window,
    while a 0-row grid states nothing the progress label right above it does not already
    say -- and the trailing spacer is what keeps that label at the top of the page
    rather than pushed halfway down 850 px of nothing. The empty page is asserted as it
    is first shown, since that is the state every launch starts in and the only one
    where the spacer has the whole page to absorb.
    """
    _shown(page, 1400, 850)
    assert not page._table.isVisibleTo(page)
    assert page._progress.y() < 200, page._progress.y()
    page.set_streams([descriptor(name="stream-0"), descriptor(name="stream-1")])
    page.layout().activate()
    assert page._table.height() > 400, page._table.height()
    assert page._table.isVisibleTo(page)
    page.set_streams([])
    page.layout().activate()
    assert not page._table.isVisibleTo(page)


def test_event_line_hidden_without_event_sources(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that the event line shows up only when there is an event source to name.

    'No event sources' is the normal case and states nothing actionable, so it is a
    permanent line of noise under the table. The text is cleared rather than kept, thus
    the hidden line can never carry the names of a previous pass. A pass which found
    *only* event sources says so, since discovery calls that one 'updated' and the
    progress label above would otherwise read "Updated just now" over an empty page.
    """
    page.set_streams([descriptor(name="stream-0")])
    assert not page._events_label.isVisibleTo(page)
    assert page._events_label.text() == ""
    page.set_streams([descriptor(name="stream-0"), descriptor(name="mrk", sfreq=0.0)])
    assert page._events_label.isVisibleTo(page)
    assert page._events_label.text().startswith("Event sources")
    assert "mrk" in page._events_label.text()
    page.set_streams([descriptor(name="mrk", sfreq=0.0)])
    assert page._events_label.isVisibleTo(page)
    assert page._events_label.text().startswith("Only event sources found")
    page.set_streams([descriptor(name="stream-0")])
    assert not page._events_label.isVisibleTo(page)


def test_plain_click_toggles_one_row(
    page: EmptyStatePage, descriptor: Callable[..., StreamDescriptor]
) -> None:
    """Test that a plain click adds a stream to the selection, and a second removes it.

    The page exists to build a multi-selection, and under 'ExtendedSelection' the second
    stream costs a modifier nobody discovers: the click below would replace the
    selection instead of extending it. The selection-mode equality assertion alone
    passes for any value someone types, hence the gesture.
    """
    streams = [descriptor(name=f"stream-{index}") for index in range(3)]
    page.set_streams(streams)
    _shown(page, 900, 700)
    _click_row(page, 0)
    assert page.selected_descriptors() == [streams[0]]
    _click_row(page, 1)
    assert page.selected_descriptors() == [streams[0], streams[1]]
    _click_row(page, 1)  # the same row again: toggles off, with no modifier
    assert page.selected_descriptors() == [streams[0]]
    _click_row(page, 0)
    assert page.selected_descriptors() == []


# -- the saved-configuration cards -----------------------------------------------------
def _state(name: str, state: str, reason: str = "", n_streams: int = 0):
    """Return one rendered card row, as the window's availability check emits it."""
    return ConfigurationState(
        name=name, state=state, reason=reason, n_streams=n_streams
    )


def _click(card: ConfigurationCard) -> None:
    """Release the left button at the centre of ``card``.

    The handler is driven rather than 'QTest.mouseClick': the activation gate is a
    property of the card and not of Qt's own event filtering, and a synthetic release
    reaching a non-activatable card is exactly what the gate exists to refuse.
    """
    centre = QPointF(card.rect().center())
    card.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            centre,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def test_configuration_region_hidden_when_empty(page: EmptyStatePage) -> None:
    """Test that a page with no saved configuration shows no configuration region.

    Rendering the group box regardless would put an empty framed panel on the page of
    every first launch, which is the state the launcher spends most of its life in.
    """
    assert not page._cards_group.isVisibleTo(page)
    page.set_configurations([_state("one", STATE_AVAILABLE, n_streams=2)])
    assert page._cards_group.isVisibleTo(page)
    page.set_configurations([])
    assert not page._cards_group.isVisibleTo(page)


def test_cards_are_persistent_widgets(page: EmptyStatePage) -> None:
    """Test that the same card object survives three passes which change its state.

    Object identity and not the count: a page which tears the region down and rebuilds
    it drops keyboard focus mid-check and makes the move-to-the-top transition
    impossible, while keeping the count exactly right.
    """
    page.set_configurations([_state("one", STATE_CHECKING)])
    card = page._cards["one"]
    for state in (STATE_UNAVAILABLE_NO_MATCH, STATE_AVAILABLE, STATE_LOADING):
        page.set_configurations([_state("one", state, n_streams=2)])
        assert page._cards["one"] is card
        assert card.state == state


def test_card_removed_when_the_configuration_vanishes(page: EmptyStatePage) -> None:
    """Test that a name absent from a new pass loses its card.

    A deleted configuration otherwise keeps a card until the window is reopened, and
    activating it starts a load of a file which no longer exists.
    """
    page.set_configurations(
        [_state("one", STATE_AVAILABLE), _state("two", STATE_INVALID)]
    )
    assert page.configuration_names() == ("one", "two")
    card = page._cards["one"]
    page.set_configurations([_state("two", STATE_INVALID)])
    assert page.configuration_names() == ("two",)
    assert "one" not in page._cards
    # asserted before the deferred deletion is flushed, which is the whole window the
    # card is visible in: 'removeWidget' takes a widget out of the layout and leaves it
    # shown, painting at its last geometry over the cards re-sorted underneath it.
    assert card.isHidden()


def test_card_group_order(page: EmptyStatePage) -> None:
    """Test that the cards group by availability and sort by name inside a group.

    'unavailable-channels' belongs to the **top** group: it did identity-match, so
    burying it next to the streams which are simply absent hides the exact reason the
    eager probe exists to produce. The invalid card sorts last, and ties break on the
    casefolded name so two names differing only by case cannot swap between two passes.
    """
    page.set_configurations(
        [
            _state("zz-invalid", STATE_INVALID, "unreadable"),
            _state("mm-no-match", STATE_UNAVAILABLE_NO_MATCH, "absent"),
            _state("B-available", STATE_AVAILABLE, n_streams=1),
            _state("a-channels", STATE_UNAVAILABLE_CHANNELS, "channels"),
            _state("c-checking", STATE_CHECKING, "checking"),
        ]
    )
    assert page.configuration_names() == (
        "a-channels",
        "B-available",
        "c-checking",
        "mm-no-match",
        "zz-invalid",
    )


def test_only_available_is_clickable(page: EmptyStatePage) -> None:
    """Test that a click opens the configuration of an available card and of no other.

    Activating a checking card starts a load against a channel set nobody has read yet,
    which is the worst of the six state errors: the load connects every stream and only
    then finds out the configuration does not match.
    """
    states = (
        STATE_CHECKING,
        STATE_AVAILABLE,
        STATE_UNAVAILABLE_CHANNELS,
        STATE_UNAVAILABLE_NO_MATCH,
        STATE_INVALID,
        STATE_LOADING,
    )
    opened: list[str] = []
    page.open_configuration_requested.connect(opened.append)
    for state in states:
        page.set_configurations([_state(state, state, n_streams=1)])
        _click(page._cards[state])
    assert opened == [STATE_AVAILABLE]


def test_loading_disables_every_card(page: EmptyStatePage) -> None:
    """Test that no card is activatable while a configuration is being opened.

    Two concurrent loads mean two load attempts, and the second silently strands the
    streams the first one connected.

    Driven by the explicit flag, not by a card carrying the loading state: the caller
    has to say so, because the row of the configuration being opened can disappear --
    delete it mid-load and an inferred flag would re-activate every sibling while the
    load is still running. This fails if the flag stops reaching the cards.
    """
    opened: list[str] = []
    page.open_configuration_requested.connect(opened.append)
    page.set_configurations([_state("ready", STATE_AVAILABLE, n_streams=1)])
    assert page._cards["ready"].activatable
    page.set_configurations(
        [
            _state("ready", STATE_AVAILABLE, n_streams=1),
            _state("busy", STATE_LOADING, "Connecting…"),
        ],
        loading=True,
    )
    assert not page._cards["ready"].activatable
    _click(page._cards["ready"])
    assert opened == []
    # and the row being opened can vanish without reviving its siblings
    page.set_configurations(
        [_state("ready", STATE_AVAILABLE, n_streams=1)], loading=True
    )
    assert not page._cards["ready"].activatable
    _click(page._cards["ready"])
    assert opened == []


def test_card_open_by_keyboard(page: EmptyStatePage) -> None:
    """Test that Return, Enter and Space open a focusable available card.

    Dropping the key handler makes the whole feature mouse-only, and a non-activatable
    card must not be a tab stop either or the focus ring lands on something inert.
    """
    opened: list[str] = []
    page.open_configuration_requested.connect(opened.append)
    page.set_configurations([_state("one", STATE_AVAILABLE, n_streams=1)])
    card = page._cards["one"]
    assert card.focusPolicy() == Qt.FocusPolicy.StrongFocus
    for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
        card.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        )
    assert opened == ["one", "one", "one"]
    page.set_configurations([_state("one", STATE_CHECKING)])
    assert card.focusPolicy() == Qt.FocusPolicy.NoFocus
    card.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
        )
    )
    assert opened == ["one", "one", "one"]


def test_card_reason_is_never_markup(page: EmptyStatePage) -> None:
    """Test that a reason built from a stream name renders verbatim, tooltip escaped.

    A stream name arrives over the network and lands in an unavailability reason: a
    'QLabel' is 'AutoText' by default, so one named '<!--' erases every character after
    it from the line and the reason shown is not the reason computed. A tooltip has no
    format of its own and always auto-detects rich text, hence the escaping.
    """
    reason = "No matching stream: <!-- (eeg/<b>x</b>)."
    page.set_configurations([_state("one", STATE_UNAVAILABLE_NO_MATCH, reason)])
    card = page._cards["one"]
    assert card._reason.textFormat() == Qt.TextFormat.PlainText
    assert card._title.textFormat() == Qt.TextFormat.PlainText
    assert card._reason.text() == reason
    assert "&lt;!--" in card._reason.toolTip()
    assert "<b>" not in card._reason.toolTip()


def test_delete_enabled_and_rename_disabled_on_invalid(page: EmptyStatePage) -> None:
    """Test that Delete works in every state and Rename does not on the broken ones.

    Deleting the file is the only way to clear a corrupt configuration from the
    interface, so gating Delete on activatability makes it unremovable. Renaming one
    whose own name could not be read would write a file titled by the card's fallback,
    and renaming one which is opening changes the name under the load in flight.
    """
    states = (
        STATE_CHECKING,
        STATE_AVAILABLE,
        STATE_UNAVAILABLE_CHANNELS,
        STATE_UNAVAILABLE_NO_MATCH,
        STATE_INVALID,
        STATE_LOADING,
    )
    page.set_configurations([_state(state, state, n_streams=1) for state in states])
    for state in states:
        card = page._cards[state]
        assert card._delete.isEnabled(), state
        assert card._rename.isEnabled() is (
            state not in (STATE_INVALID, STATE_LOADING)
        ), state


def test_card_signals_carry_the_name(page: EmptyStatePage) -> None:
    """Test that all three card signals carry the configuration name.

    Emitting an index instead would open, rename or delete a different configuration as
    soon as a pass reorders the cards, which every pass may do.
    """
    seen: list[tuple[str, str]] = []
    page.open_configuration_requested.connect(lambda name: seen.append(("open", name)))
    page.rename_configuration_requested.connect(
        lambda name: seen.append(("rename", name))
    )
    page.delete_configuration_requested.connect(
        lambda name: seen.append(("delete", name))
    )
    page.set_configurations([_state("mine", STATE_AVAILABLE, n_streams=1)])
    card = page._cards["mine"]
    _click(card)
    card._rename.click()
    card._delete.click()
    assert seen == [("open", "mine"), ("rename", "mine"), ("delete", "mine")]


def test_set_configurations_updates_in_place_not_by_index(page: EmptyStatePage) -> None:
    """Test that a card keeps its own name when a pass reorders the region.

    A positional card-to-name map opens the configuration which now occupies the row the
    pressed card used to be in, while the label under the pointer says otherwise.
    """
    opened: list[str] = []
    page.open_configuration_requested.connect(opened.append)
    page.set_configurations(
        [_state("aaa", STATE_AVAILABLE, n_streams=1), _state("bbb", STATE_CHECKING)]
    )
    assert page.configuration_names() == ("aaa", "bbb")
    page.set_configurations(
        [
            _state("aaa", STATE_UNAVAILABLE_NO_MATCH, "absent"),
            _state("bbb", STATE_AVAILABLE, n_streams=1),
        ]
    )
    assert page.configuration_names() == ("bbb", "aaa")
    card = page._cards["bbb"]
    assert card._title.text() == "bbb"
    _click(card)
    assert opened == ["bbb"]


def test_retint_icons_reaches_every_card(
    app: QApplication, controller: ThemeController, page: EmptyStatePage
) -> None:
    """Test that a theme flip rebuilds the glyph of every card.

    A 'QIcon' bakes its colour at creation, thus a flip which does not replay the table
    leaves the cards holding the previous mode's glyphs while everything else moved.
    """
    controller.install(app, "light")
    page.set_configurations(
        [
            _state("one", STATE_AVAILABLE, n_streams=1),
            _state("two", STATE_INVALID, "bad"),
        ]
    )
    page.retint_icons()
    before = {
        name: card._glyph.pixmap().toImage() for name, card in page._cards.items()
    }
    controller.set_mode("dark")
    page.retint_icons()
    for name, card in page._cards.items():
        assert not before[name].isNull(), name
        assert card._glyph.pixmap().toImage() != before[name], name


def test_cards_region_scrolls(page: EmptyStatePage) -> None:
    """Test that a long list of configurations scrolls instead of growing the page.

    Without the cap, 20 saved configurations push the available-stream table off the
    page entirely, i.e. the region for building a *new* workspace becomes unreachable.
    """
    page.set_configurations(
        [_state(f"cfg-{index:02d}", STATE_CHECKING) for index in range(20)]
    )
    assert page._cards_scroll.maximumHeight() == mne_lsl.viewer._launcher._CARDS_MAX_H
    assert page._cards_host.sizeHint().height() > page._cards_scroll.maximumHeight()
