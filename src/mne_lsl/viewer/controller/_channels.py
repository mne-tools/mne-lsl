"""Channels page: a single-column list, a compact toolbar and a contextual inspector.

Accepted design: Proposal A's lean single-column list plus a contextual inspector which
appears only when at least one channel is selected. A plain click selects, while the
leading eye glyph is the dedicated visibility toggle; ordering is command-driven through
the animated segmented control and writes the shared model order.

Search and the three filters are a **browse aid only**: they hide rows with
``setRowHidden`` and never touch the model, so they never reach the traces. Show and
Hide are the trace contract. The two are deliberately separate -- a filter which drove
the display would, with ``Show: Hidden`` selected, draw exactly the channels the user
chose not to draw.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QRect, QSignalBlocker, QSize, Qt
from qtpy.QtGui import QFont, QFontMetrics, QIcon, QPainter, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QMessageBox,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..theme import follow_theme, icon, theme_controller, tokens, type_color
from ..widgets import AnimatedSegmentedControl
from ._model import (
    CH_TYPES,
    BadRole,
    NameRole,
    TypeRole,
    UnitRole,
    VisibleRole,
    unit_choices,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from qtpy.QtCore import QAbstractItemModel, QModelIndex, QObject, QPoint
    from qtpy.QtGui import QCloseEvent, QPixmap, QShowEvent
    from qtpy.QtWidgets import QStyleOptionViewItem

    from ._model import Channel, ChannelModel

# Shown in a field or a combo box when the selection holds more than one distinct value.
# A bulk edit must never look as if the whole selection already shared the first row's
# value, or committing the combo would silently rewrite the rest.
MULTIPLE = "⟨multiple⟩"

# Rows of a selection listed in the inspector's acquisition-original line, before it is
# summarized with a trailing count.
_MAX_ORIGINALS = 4

# Narrowest the page may be docked at. The row layout packs its metadata cluster
# right-to-left from the right edge, thus a much narrower page pushes the cluster under
# the eye glyph; the delegate degrades gracefully below this, but there is no reason to
# let a dock get there.
_MIN_WIDTH = 240


def _common(values: Iterable):
    """Return the single value shared by ``values``, or ``None`` if they differ.

    Parameters
    ----------
    values : iterable
        The values of one field over a selection.

    Returns
    -------
    value : object | None
        The shared value, or ``None`` for a mixed selection.
    """
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


class ChannelDelegate(QStyledItemDelegate):
    """Paint one compact channel row and route the eye-glyph clicks.

    The row renders as aligned columns ``name │ ● type │ unit``: the metadata cluster,
    i.e. the bad marker, the type-color dot, the type text and the unit, is packed
    right-to-left with measured font metrics and pinned to the right edge, while the
    name fills the left span and elides. A click inside the leading eye rectangle
    toggles the visibility; a click anywhere else is left to the view, which performs
    its normal multi-selection.

    Parameters
    ----------
    parent : QObject | None
        Parent object.

    Notes
    -----
    Every state carries a cue which is not a color, as required for a display which may
    be read on a projector or by a color-blind user: the eye glyph for the visibility,
    an italic name for a hidden channel, a struck-through name plus a cross for a bad
    one, a left accent bar for the selection, and the channel type as text next to its
    color dot.
    """

    _ROW_H = 26
    # Side of the eye and of the bad glyph, in logical pixels.
    _EYE = 20
    _BAD = 14

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the delegate and build its palette-derived icons."""
        super().__init__(parent)
        self.refresh_palette()

    def refresh_palette(self) -> None:
        """Rebuild the palette-derived eye and bad icons after a theme change.

        Notes
        -----
        Idempotent, and mandatory on a theme flip: QtAwesome bakes the color into the
        icon engine when the icon is built and never re-colors an existing one, so a
        flip would otherwise leave the glyphs in the previous mode's color forever.

        The eye is keyed on ``(visible, selected)`` because a selected row is painted on
        the highlight brush, where the ordinary text color does not read.

        The pixmap cache is dropped rather than rebuilt here: QtAwesome's icon engine is
        Python, so ``QIcon.paint`` re-renders the glyph on every call -- 9.3 µs against
        0.4 µs for a ready pixmap, twice per row. It is refilled by the first paint,
        which is also the first place the device pixel ratio is known.
        """
        palette = QApplication.palette()
        text = palette.color(QPalette.ColorRole.Text).name()
        selected = palette.color(QPalette.ColorRole.HighlightedText).name()
        self._eye = {
            (True, False): icon("mdi6.eye-outline", color=text),
            (False, False): icon("mdi6.eye-off-outline", color=text),
            (True, True): icon("mdi6.eye-outline", color=selected),
            (False, True): icon("mdi6.eye-off-outline", color=selected),
        }
        self._bad = icon("mdi6.close", color=tokens(theme_controller.mode).error)
        self._pixmaps: dict[object, QPixmap] = {}
        self._dpr = 0.0

    def _rasterize(self, dpr: float) -> None:
        """Rasterize every glyph for a device pixel ratio of ``dpr``."""
        self._dpr = dpr
        size = QSize(self._EYE, self._EYE)
        self._pixmaps = {
            key: glyph.pixmap(size, dpr) for key, glyph in self._eye.items()
        }
        self._pixmaps["bad"] = self._bad.pixmap(QSize(self._BAD, self._BAD), dpr)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """Return the fixed row size; the width comes from the view."""
        return QSize(120, self._ROW_H)

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        """Paint the row of ``index``.

        Notes
        -----
        Every role is read once into a local: the row's five values are each needed
        twice or more, and ``data`` goes through the model's role dispatch each time.
        """
        painter.save()
        rect = option.rect
        palette = option.palette
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        visible = bool(index.data(VisibleRole))
        bad = bool(index.data(BadRole))
        ch_type = str(index.data(TypeRole))
        layout = self._row_layout(rect, index, option.font)
        dpr = painter.device().devicePixelRatioF()
        if dpr != self._dpr:
            self._rasterize(dpr)

        if selected:
            painter.fillRect(rect, palette.color(QPalette.ColorRole.Highlight))
            painter.fillRect(
                QRect(rect.left(), rect.top(), 3, rect.height()),
                palette.color(QPalette.ColorRole.HighlightedText),
            )
            foreground = palette.color(QPalette.ColorRole.HighlightedText)
        else:
            foreground = palette.color(QPalette.ColorRole.Text)

        painter.drawPixmap(layout["eye"], self._pixmaps[(visible, selected)])

        name_font = QFont(option.font)
        # bold only while selected: the accepted design's selection cue is the accent
        # bar plus the weight plus the background, and a row which is always bold makes
        # the weight distinguish nothing.
        name_font.setBold(selected)
        name_font.setItalic(not visible)
        name_font.setStrikeOut(bad)
        painter.setFont(name_font)
        painter.setPen(foreground)
        # elided rather than clipped: a 200-character name would otherwise paint
        # straight over the metadata cluster.
        name = QFontMetrics(name_font).elidedText(
            str(index.data(NameRole)),
            Qt.TextElideMode.ElideRight,
            layout["name"].width(),
        )
        painter.drawText(
            layout["name"],
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            name,
        )

        painter.setFont(option.font)
        painter.setPen(foreground)
        painter.drawText(
            layout["type"],
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            ch_type,
        )
        painter.drawText(
            layout["unit"],
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            str(index.data(UnitRole)),
        )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(type_color(ch_type))
        painter.drawEllipse(layout["dot"])

        if bad:
            painter.drawPixmap(layout["bad"], self._pixmaps["bad"])
        painter.restore()

    def editorEvent(
        self,
        event: QEvent,
        model: QAbstractItemModel,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> bool:
        """Toggle the visibility on a click inside the eye rectangle.

        Parameters
        ----------
        event : QEvent
            The event delivered to the row.
        model : QAbstractItemModel
            The model behind the view.
        option : QStyleOptionViewItem
            Style option carrying the row rectangle.
        index : QModelIndex
            The row's model index.

        Returns
        -------
        handled : bool
            ``True`` when the event was consumed, i.e. inside the eye rectangle.

        Notes
        -----
        Both the press and the release are swallowed, so a click on the eye does not
        also move the selection, and the toggle itself happens on the release alone. A
        click anywhere else returns ``False`` and the view performs its ordinary
        multi-selection.

        An invalid index is declined rather than consumed. ``QListView`` bails out
        before it reaches a delegate with one, so this is unreachable today, but
        consuming a click and then calling ``setData`` on an invalid index would eat the
        click and do nothing at all.
        """
        if not index.isValid():
            return False
        if (
            event.type()
            not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease)
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return False
        if not self._eye_rect(option.rect).contains(event.position().toPoint()):
            return False  # a row click selects; leave it to the view
        if event.type() == QEvent.Type.MouseButtonRelease:
            model.setData(index, not bool(index.data(VisibleRole)), VisibleRole)
        return True

    def _eye_rect(self, rect: QRect) -> QRect:
        """Return the clickable eye-toggle sub-rectangle of a row."""
        return QRect(rect.left() + 6, rect.top() + (rect.height() - 20) // 2, 20, 20)

    def _row_layout(
        self, rect: QRect, index: QModelIndex, font: QFont
    ) -> dict[str, QRect | None]:
        """Compute the sub-rectangles of one row.

        Parameters
        ----------
        rect : QRect
            The full row rectangle.
        index : QModelIndex
            The row's model index.
        font : QFont
            The metadata font, used to measure the type and unit text.

        Returns
        -------
        rects : dict
            The ``eye``, ``name``, ``dot``, ``type`` and ``unit`` rectangles, plus
            ``bad``, which is ``None`` unless the channel is marked bad.

        Notes
        -----
        The cluster is packed right-to-left from the right edge with measured text
        widths, and the name then fills whatever is left, so the two abut and there is
        no reserved dead column between the name and the type -- which is what the
        fixed-width boxes it replaces suffered from.

        The cluster origin is floored at the name's own origin. Packing right-to-left on
        a row narrower than the cluster would otherwise place it at a negative x, where
        it is simply not drawn: the type, the unit and the bad marker would all vanish
        rather than degrade, and just above that width they would sit under the eye
        glyph. Shifted back they overlap the name, which still elides.
        """
        height = rect.height()
        middle = rect.top() + height // 2
        metrics = QFontMetrics(font)
        unit = str(index.data(UnitRole))
        ch_type = str(index.data(TypeRole))
        bad = bool(index.data(BadRole))

        right = rect.right() - 8
        width = metrics.horizontalAdvance(unit)
        unit_rect = QRect(right - width, rect.top(), width, height)
        right = unit_rect.left() - 10
        width = metrics.horizontalAdvance(ch_type)
        type_rect = QRect(right - width, rect.top(), width, height)
        right = type_rect.left() - 6
        dot = QRect(right - 8, middle - 4, 8, 8)
        right = dot.left() - 8
        bad_rect = None
        if bad:
            bad_rect = QRect(right - 14, middle - 7, 14, 14)
            right = bad_rect.left() - 6
        left = rect.left() + 30
        shift = max(left - right, 0)
        if shift:
            for item in (unit_rect, type_rect, dot):
                item.translate(shift, 0)
            if bad_rect is not None:
                bad_rect.translate(shift, 0)
            right += shift
        name_rect = QRect(left, rect.top(), max(right - left, 10), height)
        return {
            "eye": self._eye_rect(rect),
            "name": name_rect,
            "dot": dot,
            "type": type_rect,
            "unit": unit_rect,
            "bad": bad_rect,
        }


class ChannelsPage(QWidget):
    """Channels page of the controller.

    The page does not own its model: the stream document builds one
    :class:`~mne_lsl.viewer.controller.ChannelModel` and shares it with the trace
    display, so the two presentations never hold duplicate channel state.

    Parameters
    ----------
    model : ChannelModel
        The channel model of the stream document.
    parent : QWidget | None
        Parent widget.

    Notes
    -----
    The model is borrowed and never reparented: the document owns it, and it outlives
    this page whenever the panel is hidden or closed.
    """

    def __init__(self, model: ChannelModel, parent: QWidget | None = None) -> None:
        """Initialize the page over an existing channel model."""
        super().__init__(parent)
        self._model = model
        self._following_theme = False
        self._shown = 0  # rows the filter last left visible

        self._view = QListView()
        self._view.setModel(model)
        self._view.setItemDelegate(ChannelDelegate(self._view))
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._view.setUniformItemSizes(True)
        self._view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._context_menu)
        self._selection = self._view.selectionModel()
        self._selection.selectionChanged.connect(self._on_selection_changed)

        model.dataChanged.connect(self._on_data_changed)
        # a reorder needs no repair -- Qt keys the hidden flag on the *item*, so the
        # filter already follows its channel -- but re-running it keeps the page from
        # silently depending on an implementation detail of 'QListView'.
        model.layoutChanged.connect(self._apply_filter)
        model.modelReset.connect(self._on_model_reset)

        self._status = QLabel()
        self._status.setToolTip(
            "Rows shown by the search and the filters, out of the total channel count. "
            "The window's status bar counts the channels the traces draw, which is a "
            "different quantity: filtering the list never hides a trace."
        )
        toolbar = self._build_toolbar()
        self._inspector = self._build_inspector()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)
        outer.addWidget(toolbar)
        outer.addWidget(self._view, 1)
        outer.addWidget(self._inspector)
        outer.addWidget(self._status)
        self.setMinimumWidth(_MIN_WIDTH)

        self._apply_filter()
        self._reflect_selection()  # hides the inspector while nothing is selected

        # Connected by the page itself rather than by the shell: the delegate icons bake
        # their color, thus forgetting the connection leaves them stale after a flip.
        follow_theme(self, self.retheme, True)

    # -- public surface ----------------------------------------------------------------
    @property
    def model(self) -> ChannelModel:
        """Channel model the page presents; owned by the stream document."""
        return self._model

    @property
    def view(self) -> QListView:
        """List view showing one row per channel."""
        return self._view

    def retheme(self, *_args) -> None:
        """Rebuild the palette-derived delegate icons after a theme flip and repaint.

        Notes
        -----
        Takes the resolved mode the theme controller emits and ignores it: every glyph
        is rebuilt from the live palette and the live tokens, with nothing to key on.
        """
        self._view.itemDelegate().refresh_palette()
        self._retint_icons()
        self._view.viewport().update()

    def showEvent(self, ev: QShowEvent) -> None:
        """Follow the theme again, and rebuild whatever a flip changed while hidden.

        Notes
        -----
        The counterpart of :meth:`ChannelsPage.closeEvent`, which drops the connection.
        Without it, a panel closed and reopened keeps the previous mode's baked glyphs
        for the rest of the process -- and the eye glyph is the only visibility cue that
        is not a color, so it ends up drawn in the other mode's text color.
        """
        super().showEvent(ev)
        if not self._following_theme:
            follow_theme(self, self.retheme, True)
            self.retheme()  # the mode may have flipped while the page was closed

    def closeEvent(self, ev: QCloseEvent) -> None:
        """Drop the theme connection, as the controller is a process singleton."""
        follow_theme(self, self.retheme, False)
        super().closeEvent(ev)

    # -- toolbar -----------------------------------------------------------------------
    def _build_toolbar(self) -> QWidget:
        """Build the compact two-row toolbar above the list."""
        bar = QWidget()
        box = QVBoxLayout(bar)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search name…")
        self._search.setClearButtonEnabled(True)
        self._search.setAccessibleName("Search channel names")
        # A decoration, not a control: named so a screen reader does not announce an
        # unlabelled button, and disabled because a click has nothing to do. The glyph
        # itself is set by '_retint_icons', the single site which knows the icon names.
        self._search_action = self._search.addAction(
            QIcon(), QLineEdit.ActionPosition.LeadingPosition
        )
        self._search_action.setText("Search")
        self._search_action.setToolTip("Search channel names")
        self._search_action.setEnabled(False)
        self._search.textChanged.connect(self._apply_filter)
        row = QHBoxLayout()
        row.addWidget(self._search, 1)
        row.addWidget(self._build_filter_button())
        box.addLayout(row)

        self._order = AnimatedSegmentedControl(
            [
                ("Acq", "Acquisition order", "acquisition"),
                ("Type", "Channel type", "type"),
                ("Abc", "Alphabetical", "alphabetical"),
            ]
        )
        self._order.changed.connect(self._model.order_by)
        row = QHBoxLayout()
        row.addWidget(QLabel("Order"))
        row.addWidget(self._order, 1)
        box.addLayout(row)
        return bar

    def _build_filter_button(self) -> QToolButton:
        """Build the ``Filter`` popover holding the three filter combo boxes."""
        self._type_filter = QComboBox()
        self._type_filter.addItem("All types", None)
        for ch_type in CH_TYPES:
            self._type_filter.addItem(ch_type, ch_type)
        self._vis_filter = QComboBox()
        self._vis_filter.addItem("All", None)
        self._vis_filter.addItem("Visible", "visible")
        self._vis_filter.addItem("Hidden", "hidden")
        self._bad_filter = QComboBox()
        self._bad_filter.addItem("All", None)
        self._bad_filter.addItem("Good", "good")
        self._bad_filter.addItem("Bad", "bad")

        popover = QWidget()
        form = QFormLayout(popover)
        form.setContentsMargins(8, 8, 8, 8)
        for label, combo in (
            ("Type", self._type_filter),
            ("Show", self._vis_filter),
            ("State", self._bad_filter),
        ):
            combo.setAccessibleName(f"Filter by {label.lower()}")
            combo.currentIndexChanged.connect(self._apply_filter)
            form.addRow(label, combo)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear_filters)
        form.addRow(clear)

        action = QWidgetAction(self)
        action.setDefaultWidget(popover)
        menu = QMenu(self)
        menu.addAction(action)
        self._filter_btn = QToolButton()
        self._filter_btn.setText("Filter")
        self._filter_btn.setAccessibleName("Filter channels")
        self._filter_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._filter_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._filter_btn.setMenu(menu)
        return self._filter_btn

    # -- inspector ---------------------------------------------------------------------
    def _build_inspector(self) -> QFrame:
        """Build the contextual inspector, shown only while a channel is selected."""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(frame)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setVerticalSpacing(4)

        self._insp_header = QLabel()
        grid.addWidget(self._insp_header, 0, 0, 1, 3)

        self._name_edit = QLineEdit()
        self._name_edit.setAccessibleName("Channel name")
        self._name_edit.textChanged.connect(self._on_name_edited)
        self._rename_btn = QPushButton("Rename")
        self._rename_btn.clicked.connect(self._rename_selected)
        grid.addWidget(QLabel("Name"), 1, 0)
        grid.addWidget(self._name_edit, 1, 1)
        grid.addWidget(self._rename_btn, 1, 2)

        self._type_combo = QComboBox()
        self._type_combo.setAccessibleName("Channel type")
        # 'textActivated' and not 'currentTextChanged': only a user pick may write, or
        # reflecting the selection into the combo would write it straight back.
        self._type_combo.textActivated.connect(self._apply_type)
        grid.addWidget(QLabel("Type"), 2, 0)
        grid.addWidget(self._type_combo, 2, 1, 1, 2)

        self._unit_combo = QComboBox()
        self._unit_combo.setAccessibleName("Channel unit")
        self._unit_combo.textActivated.connect(self._apply_unit)
        grid.addWidget(QLabel("Unit"), 3, 0)
        grid.addWidget(self._unit_combo, 3, 1, 1, 2)

        self._visible_btn = QToolButton()
        self._visible_btn.setText("Visible")
        self._visible_btn.setCheckable(True)
        self._visible_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._visible_btn.toggled.connect(self._apply_visible)
        self._bad_btn = QToolButton()
        self._bad_btn.setText("Bad")
        self._bad_btn.setCheckable(True)
        self._bad_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._bad_btn.toggled.connect(self._apply_bad)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Restore the acquisition metadata of the selection")
        self._reset_btn.clicked.connect(self._apply_reset)
        self._retint_icons()
        state = QHBoxLayout()
        state.addWidget(self._visible_btn)
        state.addWidget(self._bad_btn)
        state.addWidget(self._reset_btn)
        state.addStretch(1)
        grid.addLayout(state, 4, 0, 1, 3)

        self._orig_label = QLabel()
        self._orig_label.setWordWrap(True)
        self._orig_label.setStyleSheet("color: palette(mid);")
        grid.addWidget(self._orig_label, 5, 0, 1, 3)
        return frame

    def _retint_icons(self) -> None:
        """Rebuild every QtAwesome icon of the page for the current theme.

        Notes
        -----
        The single site which names the page's glyphs. It runs at construction as well
        as on every flip, so a builder which also set an icon would hold a dead copy:
        the glyph would change there and silently revert on the first theme change.
        """
        self._search_action.setIcon(icon("mdi6.magnify"))
        self._filter_btn.setIcon(icon("mdi6.filter-variant"))
        self._visible_btn.setIcon(icon("mdi6.eye-outline"))
        self._bad_btn.setIcon(
            icon("mdi6.close", color=tokens(theme_controller.mode).error)
        )
        self._reset_btn.setIcon(icon("mdi6.backup-restore"))

    # -- selection ---------------------------------------------------------------------
    def _selected_rows(self) -> list[int]:
        """Return the selected display rows, ascending and deduplicated.

        Notes
        -----
        Read off the selection *ranges* rather than off ``selectedIndexes``, which
        materializes one ``QModelIndex`` per row and filters each through ``flags()``:
        0.29 ms against 0.004 ms for a full 256-row selection, and this runs three or
        four times per action and once per keystroke in the name field. Equivalent
        because the model marks every valid index selectable.
        """
        rows: set[int] = set()
        for span in self._selection.selection():
            rows.update(range(span.top(), span.bottom() + 1))
        return sorted(rows)

    def _reflect_selection(self) -> None:
        """Mirror the selection into the inspector, and hide it when empty.

        Notes
        -----
        The widget signals are blocked while reflecting, or writing the selection's
        current state into the widgets would re-trigger the edit slots and write it
        straight back to the stream -- the classic feedback loop.

        The Unit combo is disabled when the selection's kind offers no unit, i.e. for a
        channel which has none and for a selection spanning several kinds, since every
        label would then be refused by the write path.
        """
        rows = self._selected_rows()
        self._inspector.setVisible(bool(rows))
        if not rows:
            return
        channels = [self._model.channel(row) for row in rows]
        single = len(rows) == 1
        blockers = [
            QSignalBlocker(widget)
            for widget in (
                self._name_edit,
                self._type_combo,
                self._unit_combo,
                self._visible_btn,
                self._bad_btn,
            )
        ]
        self._insp_header.setText(
            f"{len(rows)} channel{'' if single else 's'} selected"
        )
        self._name_edit.setEnabled(single)
        self._name_edit.setText(channels[0].name if single else "")
        self._name_edit.setPlaceholderText("" if single else MULTIPLE)
        self._fill_combo(
            self._type_combo, CH_TYPES, _common(c.ch_type for c in channels)
        )
        choices = unit_choices(_common(c.unit_kind for c in channels))
        self._unit_combo.setEnabled(bool(choices))
        self._unit_combo.setToolTip(
            "" if choices else "Change the Type to give this channel a physical unit."
        )
        self._fill_combo(self._unit_combo, choices, _common(c.unit for c in channels))
        self._visible_btn.setChecked(all(c.visible for c in channels))
        self._bad_btn.setChecked(all(c.bad for c in channels))
        self._orig_label.setText(self._originals_text(channels))
        del blockers  # unblock
        self._on_name_edited(self._name_edit.text())

    @staticmethod
    def _fill_combo(combo: QComboBox, choices: Sequence[str], value) -> None:
        """Fill ``combo`` with ``choices`` and select ``value``, or the mixed marker."""
        combo.clear()
        combo.addItems(list(choices))
        if value is None:
            combo.insertItem(0, MULTIPLE)
            combo.setCurrentIndex(0)
            return
        if combo.findText(value) < 0:
            combo.addItem(value)  # the current value is off the offered ladder
        combo.setCurrentText(value)

    @staticmethod
    def _originals_text(channels: Sequence[Channel]) -> str:
        """Return the acquisition-original one-liner of ``channels``."""
        text = " · ".join(channel.original for channel in channels[:_MAX_ORIGINALS])
        if len(channels) > _MAX_ORIGINALS:
            text += f" · +{len(channels) - _MAX_ORIGINALS} more"
        return f"orig: {text}"

    # -- inspector edits, applied to the whole selection -------------------------------
    def _write(self, write: Callable[[], object]) -> object:
        """Run a model write, report a refusal and re-sync the inspector.

        Parameters
        ----------
        write : callable
            The model call, already bound to its rows and value.

        Returns
        -------
        result : object
            What ``write`` returned, or ``None`` if it was refused.

        Notes
        -----
        Qt commits a widget's visual state *before* it emits: ``QAbstractButton`` sets
        ``checked`` before ``toggled`` and ``QComboBox`` sets ``currentIndex`` before
        ``textActivated``. A write which raised, or which the model declined because the
        stream is down, emits no ``dataChanged``, so nothing would put the widget back:
        the inspector would keep displaying a value the stream never received, which is
        worse than the exception. Reflecting the selection afterwards is what makes the
        widgets show what the model actually holds.
        """
        try:
            return write()
        except ValueError as error:
            QMessageBox.warning(self, "Channel metadata", str(error))
            return None
        finally:
            self._reflect_selection()

    def _apply_type(self, text: str) -> None:
        """Set the channel type of the selection from the inspector combo."""
        if text in CH_TYPES:  # skips the mixed marker
            self._write(partial(self._model.set_type, self._selected_rows(), text))

    def _apply_unit(self, text: str) -> None:
        """Set the unit of the selection from the inspector combo.

        Notes
        -----
        Checked against the offered ladder rather than merely against the mixed marker:
        the combo also carries the selection's *current* label, which may sit off the
        ladder when the stream declared an unusual multiplier, and picking that entry
        again would otherwise be refused by the model from inside a Qt slot.
        """
        rows = self._selected_rows()
        kind = _common(self._model.channel(row).unit_kind for row in rows)
        if text in unit_choices(kind):
            self._write(partial(self._model.set_unit, rows, text))

    def _apply_visible(self, checked: bool) -> None:
        """Set the visibility of the selection from the inspector toggle."""
        self._model.set_visible(self._selected_rows(), checked)

    def _apply_bad(self, checked: bool) -> None:
        """Set the bad state of the selection from the inspector toggle."""
        self._write(partial(self._model.set_bad, self._selected_rows(), checked))

    def _apply_reset(self) -> None:
        """Restore the acquisition metadata of the selection."""
        self._reset_rows(self._selected_rows())

    def _reset_rows(self, rows: Sequence[int]) -> None:
        """Restore the acquisition metadata of ``rows``, reporting what it could not.

        Notes
        -----
        Reset is the escape hatch out of a confusing metadata state, so the one thing it
        may decline -- restoring a name another channel now holds, e.g. after a swap --
        is reported instead of leaving the row named ``'b'`` under an inspector still
        showing ``orig: a``.
        """
        skipped = self._write(partial(self._model.reset_metadata, rows))
        if skipped:
            QMessageBox.warning(
                self,
                "Reset channels",
                f"The acquisition name of {', '.join(skipped)} is held by another "
                "channel and was not restored; every other field was.",
            )

    def _on_name_edited(self, text: str) -> None:
        """Enable Rename only for a single selection with a free, non-blank name.

        Notes
        -----
        Disabling is cheaper than an error dialog and removes the failure path entirely:
        the two names the write path would accept and should not -- a blank one and one
        already in use -- are simply unreachable from this button.
        """
        rows = self._selected_rows()
        name = text.strip()
        taken = {self._model.channel(row).name for row in range(self._model.rowCount())}
        if len(rows) == 1:
            taken.discard(self._model.channel(rows[0]).name)
        self._rename_btn.setEnabled(len(rows) == 1 and bool(name) and name not in taken)

    def _rename_selected(self) -> None:
        """Rename the single selected channel from the inspector name field."""
        rows = self._selected_rows()
        if len(rows) == 1 and self._rename_btn.isEnabled():
            self._model.rename(rows[0], self._name_edit.text())

    def _rename_dialog(self, row: int) -> None:
        """Prompt for and apply a new name for the channel at ``row``.

        Notes
        -----
        The context-menu path cannot be guarded by a button state, thus it catches the
        model's refusal and reports it -- one modal for one explicit user gesture.
        """
        name, accepted = QInputDialog.getText(
            self, "Rename channel", "New name:", text=self._model.channel(row).name
        )
        if not accepted:
            return
        try:
            self._model.rename(row, name)
        except ValueError as error:
            QMessageBox.warning(self, "Rename channel", str(error))

    # -- filtering and status ----------------------------------------------------------
    def _clear_filters(self) -> None:
        """Reset the search box and the three filter combo boxes."""
        self._search.clear()
        for combo in (self._type_filter, self._vis_filter, self._bad_filter):
            combo.setCurrentIndex(0)

    def _apply_filter(self, *_args) -> None:
        """Hide the rows which match neither the search box nor the filters.

        Notes
        -----
        ``setRowHidden`` only: this is a browse aid and writes no model state, so it
        cannot reach the traces. Qt keys the hidden flag on the *item* rather than on
        the row number, thus the filter survives a reorder on its own and re-running it
        is a recompute rather than a repair.

        Only a row whose state actually changes is written, because ``setRowHidden``
        schedules a full relayout of the view unconditionally: at 256 rows that turned
        every eye click and every bulk edit into a whole-viewport repaint on top of Qt's
        own, 2.01 ms against 0.22 ms.
        """
        text = self._search.text().strip().casefold()
        by_type = self._type_filter.currentData()
        by_visible = self._vis_filter.currentData()
        by_bad = self._bad_filter.currentData()
        shown = 0
        for row in range(self._model.rowCount()):
            channel = self._model.channel(row)
            keep = text in channel.name.casefold()
            if by_type is not None and channel.ch_type != by_type:
                keep = False
            if by_visible == "visible" and not channel.visible:
                keep = False
            if by_visible == "hidden" and channel.visible:
                keep = False
            if by_bad == "good" and channel.bad:
                keep = False
            if by_bad == "bad" and not channel.bad:
                keep = False
            if self._view.isRowHidden(row) is keep:
                self._view.setRowHidden(row, not keep)
            shown += keep
        self._shown = shown
        self._update_status()

    def _update_status(self) -> None:
        """Refresh the persistent ``N selected · shown/total`` line.

        Notes
        -----
        The shown count is the one :meth:`ChannelsPage._apply_filter` just computed, and
        not a second pass over every row.
        """
        selected = len(self._selected_rows())
        self._status.setText(
            f"{selected} selected · {self._shown}/{self._model.rowCount()}"
        )

    # -- model and theme handlers ------------------------------------------------------
    def _on_data_changed(self, *_args) -> None:
        """Re-filter and refresh the inspector after a model edit."""
        self._apply_filter()
        self._reflect_selection()

    def _on_model_reset(self) -> None:
        """Drop the selection, resync the Order control and re-filter after a rebuild.

        Notes
        -----
        The selection is cleared first because a structural change invalidated every row
        identity: keeping it would leave the inspector editing whichever channels now
        happen to sit at those rows. Qt's own selection model also drops it on a reset,
        thus the call is explicit about a correctness property rather than load-bearing.

        The Order control is put back on its first segment because a rebuild restores
        the acquisition order, and the control's own re-click is a no-op on the segment
        it already shows -- so a panel left reading ``Abc`` over an acquisition-ordered
        model would have no way back to that ordering at all.
        """
        self._selection.clearSelection()
        self._order.set_index(0, emit=False)
        self._apply_filter()
        self._reflect_selection()

    def _on_selection_changed(self, *_args) -> None:
        """Refresh the inspector and the status line."""
        self._reflect_selection()
        self._update_status()

    # -- context menu, a fast path beside the inspector --------------------------------
    def _context_menu(self, pos: QPoint) -> None:
        """Open the selection context menu at ``pos``.

        Notes
        -----
        The menu is deleted after it closes rather than left to its parent: the page
        outlives every right-click, so an undeleted menu accumulates for the life of the
        process -- 500 right-clicks left 501 live menus, each carrying its actions and
        the bound callables of its two submenus.
        """
        rows = self._context_rows(pos)
        if not rows:
            return
        menu = self._menu_for(rows)
        menu.exec(self._view.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def _context_rows(self, pos: QPoint) -> list[int]:
        """Return the rows a context menu at ``pos`` applies to.

        Notes
        -----
        The selection, or the row under the cursor when there is none: a right-click
        without a prior selection would otherwise open a menu which does nothing.
        """
        rows = self._selected_rows()
        if rows:
            return rows
        index = self._view.indexAt(pos)
        return [index.row()] if index.isValid() else []

    def _menu_for(self, rows: Sequence[int]) -> QMenu:
        """Build the context menu of ``rows``."""
        model = self._model
        menu = QMenu(self)
        menu.addAction("Show", partial(model.set_visible, rows, True))
        menu.addAction("Hide", partial(model.set_visible, rows, False))
        menu.addSeparator()
        # every metadata entry goes through '_write', as the model refuses some of them
        # and an exception raised straight out of a menu action is merely logged.
        type_menu = menu.addMenu("Set type")
        for ch_type in CH_TYPES:
            type_menu.addAction(
                ch_type, partial(self._write, partial(model.set_type, rows, ch_type))
            )
        # keyed on the kind the rows share, so the menu can only offer a writable unit.
        choices = unit_choices(_common(model.channel(row).unit_kind for row in rows))
        unit_menu = menu.addMenu("Set unit")
        unit_menu.setEnabled(bool(choices))
        for label in choices:
            unit_menu.addAction(
                label, partial(self._write, partial(model.set_unit, rows, label))
            )
        for text, value in (("Mark bad", True), ("Mark good", False)):
            menu.addAction(
                text, partial(self._write, partial(model.set_bad, rows, value))
            )
        rename = menu.addAction("Rename…", partial(self._rename_dialog, rows[0]))
        rename.setEnabled(len(rows) == 1)
        menu.addSeparator()
        # through the page, not straight to the model: a reset which had to decline a
        # rename reports it, and the menu path has no inspector state to fall back on.
        menu.addAction("Reset", partial(self._reset_rows, rows))
        return menu
