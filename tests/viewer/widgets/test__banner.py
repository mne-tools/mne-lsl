from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import Qt

from mne_lsl.viewer.widgets import Banner

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pytestqt.qtbot import QtBot
    from qtpy.QtWidgets import QApplication

    from mne_lsl.viewer.theme import ThemeController


@pytest.fixture
def banner(flush_deletes: Callable[..., None]) -> Generator[Banner, None, None]:
    """Yield a banner, closed and deleted afterwards.

    Through 'flush_deletes' and not 'deleteLater' plus 'processEvents': the latter does
    not deliver a 'DeferredDelete' event outside a running event loop, so the C++
    destruction -- the path which surfaces a use-after-delete -- never runs at all.
    """
    widget = Banner()
    yield widget
    widget.close()
    flush_deletes(widget)


def test_set_notice_shows_the_text_verbatim(banner: Banner) -> None:
    """Test that the notice is shown as plain text, markup included.

    The text carries a stream-derived message, e.g. the text of an exception raised
    while reconnecting. Kills dropping 'setTextFormat(PlainText)', which would interpret
    any markup a stream name or an error message happens to contain.
    """
    banner.set_notice("<b>Stream lost</b> — reconnecting…")
    assert banner._label.text() == "<b>Stream lost</b> — reconnecting…"
    assert banner._label.textFormat() == Qt.TextFormat.PlainText


def test_retry_is_offered_only_when_asked(banner: Banner) -> None:
    """Test that Retry follows the flag and that Close is always offered.

    Kills inverting the flag, which would offer a retry for a state the viewer already
    retries on its own and withhold it from the one terminal state.
    """
    # 'isHidden', not 'isVisible': a widget which was never shown has no visible child
    # at all, and no test in this suite shows a top-level window.
    banner.set_notice("Stream lost", retry=False)
    assert banner._retry_button.isHidden()
    assert not banner._close_button.isHidden()
    banner.set_notice("the channels changed", level="error", retry=True)
    assert not banner._retry_button.isHidden()
    assert not banner._close_button.isHidden()


def test_buttons_emit(banner: Banner, qtbot: QtBot) -> None:
    """Test that both buttons reach their signal.

    Kills disconnecting either one, which leaves a visible button doing nothing.
    """
    banner.set_notice("the channels changed", level="error", retry=True)
    with qtbot.waitSignal(banner.retry_clicked, timeout=1000):
        banner._retry_button.click()
    with qtbot.waitSignal(banner.close_clicked, timeout=1000):
        banner._close_button.click()


def test_levels_differ_and_an_unknown_one_raises(banner: Banner) -> None:
    """Test that the two levels are distinguishable and that a third is refused.

    The glyph and the wording are the cues which are not the colour, so both are
    asserted. Kills collapsing the levels into one.
    """
    banner.set_notice("Stream lost", level="warning")
    warning = (banner._label.styleSheet(), banner._glyph.pixmap().cacheKey())
    banner.set_notice("Stream lost", level="error")
    error = (banner._label.styleSheet(), banner._glyph.pixmap().cacheKey())
    assert warning[0] != error[0]
    assert warning[1] != error[1]
    with pytest.raises(ValueError, match="Invalid value for the 'level' parameter"):
        banner.set_notice("Stream lost", level="critical")
    # the refused call left the previous level in place, not a half-applied one
    assert banner._level == "error"


def test_theme_flip_retints(
    app: QApplication, controller: ThemeController, banner: Banner
) -> None:
    """Test that a retint after a theme flip changes the colour and the glyph.

    A 'QIcon' bakes its colour at creation, so a flip needs the glyph rebuilt rather
    than repainted. Kills dropping the retint, which leaves the banner in the previous
    mode's colours for the rest of the process.
    """
    controller.install(app, "light")
    banner.set_notice("Stream lost")
    before = (banner._label.styleSheet(), banner._glyph.pixmap().cacheKey())
    controller.set_mode("dark")
    banner.retint_icons()  # the document forwards the flip; the banner does not listen
    after = (banner._label.styleSheet(), banner._glyph.pixmap().cacheKey())
    assert before[0] != after[0]
    assert before[1] != after[1]
