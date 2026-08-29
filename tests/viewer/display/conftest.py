from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
import pytest
from qtpy.QtWidgets import QToolButton

from mne_lsl.viewer.display import TraceDisplay

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qtpy.QtWidgets import QApplication, QWidget

    from mne_lsl.stream import StreamLSL


@pytest.fixture
def push(
    default_stream: tuple[StreamLSL, Callable[..., None]],
) -> Callable[..., None]:
    """Return the callable pushing a synthesized block onto the default stream."""
    return default_stream[1]


@pytest.fixture
def make_display(
    flush_deletes: Callable[..., None],
) -> Generator[Callable[[StreamLSL], TraceDisplay]]:
    """Yield a factory building trace displays, closed at teardown.

    Same shape as the 'lsl_stream' factory, and for the same reason: the tests needing a
    non-default stream built their display in the body, and a hand-rolled 'try/finally'
    per test is what this replaces.
    """
    created: list[TraceDisplay] = []

    def _make(stream: StreamLSL, width: int = 1000, height: int = 600) -> TraceDisplay:
        """Build one display over ``stream`` and register it for teardown."""
        widget = TraceDisplay(stream)
        widget.resize(width, height)
        created.append(widget)
        return widget

    yield _make
    for widget in reversed(created):
        widget.stop()
        widget.close()
    flush_deletes(*reversed(created))
    created.clear()


@pytest.fixture
def display(
    make_display: Callable[[StreamLSL], TraceDisplay], stream: StreamLSL
) -> TraceDisplay:
    """Return a trace display over the default stream, closed afterwards."""
    return make_display(stream)


@pytest.fixture
def pg_background() -> Generator[Callable[[str], None]]:
    """Yield a setter for pyqtgraph's background configuration, restored afterwards.

    The 'controller' fixture deliberately leaves the pyqtgraph configuration alone, as
    no test may assert a default application look; a test which pushes a color no theme
    ever sets therefore has to put the previous one back itself.
    """
    previous = pg.getConfigOption("background")
    yield lambda color: pg.setConfigOption("background", color)
    pg.setConfigOption("background", previous)


@pytest.fixture
def tool_button() -> Callable[[QWidget, str], QToolButton]:
    """Return a helper looking a tool button up by its tooltip.

    The bar identifies its steppers by tooltip alone -- there is no object name and no
    accessor -- thus both the control-bar tests and the display's icon tests need this
    lookup.
    """

    def _button(widget: QWidget, tip: str) -> QToolButton:
        """Return the tool button of ``widget`` whose tooltip is ``tip``."""
        for button in widget.findChildren(QToolButton):
            if button.toolTip() == tip:
                return button
        pytest.fail(f"No tool button with the tooltip {tip!r}.")

    return _button


@pytest.fixture
def shown_display(app: QApplication, display: TraceDisplay) -> Generator[TraceDisplay]:
    """Yield a displayed trace display, for the assertions which need real painting.

    'AxisItem.drawPicture' only runs when the item is actually painted, thus a test
    covering it has to show the widget and let the scene render.
    """
    display.show()
    app.processEvents()
    yield display
    display.hide()
