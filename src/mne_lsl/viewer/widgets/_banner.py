"""One-line notice strip shown above a document's content."""

from __future__ import annotations

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from ...utils._checks import check_value
from ..theme import _ICON_PX, icon, theme_controller, tokens

# The two levels a notice may carry, each with the glyph which is its cue that is not
# the color: a warning is a transient state the viewer is working on, an error is one it
# stopped working on and is waiting for the operator. The keys double as the token names
# the color is read from, see 'retint_icons', which is why they are not free-form tags.
_LEVELS = {"warning": "mdi6.alert-outline", "error": "mdi6.alert-circle-outline"}


class Banner(QWidget):
    """One-line notice strip with an optional retry and a close action.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget.

    Attributes
    ----------
    retry_clicked : Signal
        Emitted when the Retry button is clicked.
    close_clicked : Signal
        Emitted when the Close document button is clicked.

    Notes
    -----
    The widget knows nothing about streams, states or reconnection: it takes a string, a
    level and whether a retry is offered. That is what makes it testable without a
    stream connection, and what keeps a document's state machine out of a leaf widget.

    It has no ``follow_theme`` handler of its own. A document already owns a
    :meth:`~mne_lsl.viewer._document.StreamDocument.retint_icons` which the window calls
    for every document on a theme flip, and it forwards to :meth:`retint_icons` here: a
    banner cannot outlive its document, so a third copy of the show/close bookkeeping a
    self-following widget needs would buy nothing.
    """

    retry_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the banner."""
        super().__init__(parent)
        self._level = "warning"

        box = QHBoxLayout(self)
        box.setContentsMargins(8, 3, 6, 3)
        box.setSpacing(6)

        self._glyph = QLabel()
        box.addWidget(self._glyph)

        self._label = QLabel()
        # Mandatory: the text carries a stream-derived message, e.g. the text of an
        # exception raised while reconnecting, and a rich-text label would interpret any
        # markup a stream name or an error message happens to contain.
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setWordWrap(False)
        box.addWidget(self._label)
        box.addStretch(1)

        self._retry_button = QToolButton()
        self._retry_button.setText("Retry")
        self._retry_button.setToolTip(
            "Look for the stream again and check that it is the same stream"
        )
        self._retry_button.clicked.connect(self.retry_clicked)
        # hidden by default, so the constructed state matches 'set_notice''s default.
        self._retry_button.setVisible(False)
        box.addWidget(self._retry_button)

        self._close_button = QToolButton()
        self._close_button.setIconSize(QSize(_ICON_PX, _ICON_PX))
        self._close_button.setText("Close document")
        self._close_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        # Labelled with what it does, not with 'Close': on a notice strip that reads as
        # 'dismiss this message', while it closes the document and discards its edits.
        self._close_button.setToolTip(
            "Close this stream document. Its channel edits and display settings are "
            "discarded."
        )
        self._close_button.clicked.connect(self.close_clicked)
        box.addWidget(self._close_button)

        self.retint_icons()

    def set_notice(
        self, text: str, *, level: str = "warning", retry: bool = False
    ) -> None:
        """Show ``text`` at ``level``, offering a retry or not.

        Parameters
        ----------
        text : str
            The notice, shown verbatim as plain text.
        level : str
            ``'warning'`` for a state the viewer is still working on, ``'error'`` for
            one it is waiting on the operator for.
        retry : bool
            Whether the Retry button is offered. Close is always offered.

        Raises
        ------
        ValueError
            If ``level`` is neither ``'warning'`` nor ``'error'``.
        """
        check_value(level, tuple(_LEVELS), "level")
        self._level = level
        self._label.setText(str(text))
        self._retry_button.setVisible(bool(retry))
        self.retint_icons()

    def retint_icons(self) -> None:
        """Rebuild the glyph and the text color for the active theme.

        Notes
        -----
        A ``QIcon`` bakes its color at creation, thus a theme flip needs the glyph
        rebuilt rather than merely repainted.

        The color is looked up by the level's own name, which *is* a
        :class:`~mne_lsl.viewer.theme.Tokens` field: a ternary beside the level-to-glyph
        table would be a second place a third level has to be added to, and the one
        which fails silently, by falling through to the warning color.
        """
        palette = tokens(theme_controller.mode)
        color = getattr(palette, self._level)
        self._glyph.setPixmap(
            icon(_LEVELS[self._level], color=color).pixmap(_ICON_PX, _ICON_PX)
        )
        self._label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._close_button.setIcon(icon("mdi6.close"))
