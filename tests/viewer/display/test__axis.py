from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QPainter

from mne_lsl.viewer.display import ChannelAxis, TraceViewBox
from mne_lsl.viewer.display._axis import _MAX_LABEL

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtCore import QRect
    from qtpy.QtWidgets import QApplication

    from mne_lsl.viewer.display import TraceDisplay
    from mne_lsl.viewer.theme import ThemeController


class _Wheel:
    """Minimal stand-in for the graphics wheel event the axis and view box receive."""

    def __init__(self, delta: int, modifiers: Qt.KeyboardModifier) -> None:
        self._delta = delta
        self._modifiers = modifiers
        self.accepted = False

    def delta(self) -> int:
        """Return the wheel delta, in eighths of a degree."""
        return self._delta

    def modifiers(self) -> Qt.KeyboardModifier:
        """Return the keyboard modifiers of the event."""
        return self._modifiers

    def accept(self) -> None:
        """Record that the event was accepted."""
        self.accepted = True


class _Drag:
    """Minimal stand-in for a graphics drag event; only 'ignore' is exercised."""

    def __init__(self) -> None:
        self.ignored = False

    def ignore(self) -> None:
        """Record that the event was ignored."""
        self.ignored = True


def _distance(a: QColor, b: QColor) -> int:
    """Return the squared RGB distance between two colors."""
    return (
        (a.red() - b.red()) ** 2
        + (a.green() - b.green()) ** 2
        + (a.blue() - b.blue()) ** 2
    )


def _paint_axis(
    axis: ChannelAxis, monkeypatch: pytest.MonkeyPatch, background: QColor
) -> tuple[QImage, list[tuple[QRect, int, str]]]:
    """Paint ``axis`` onto an image and return it with the label specs it drew.

    The specs are captured on the way through 'drawPicture' -- pyqtgraph builds them
    inside 'paint' and hands them over, thus there is no other way to learn where a
    given label landed -- and their rectangles are translated into image coordinates.
    """
    specs: list[tuple[QRect, int, str]] = []
    original = axis.drawPicture
    bounds = axis.boundingRect()

    def _record(p, axis_spec, tick_specs, text_specs):
        specs.extend(
            (rect.toAlignedRect().translated(-bounds.topLeft().toPoint()), flags, text)
            for rect, flags, text in text_specs
        )
        original(p, axis_spec, tick_specs, text_specs)

    monkeypatch.setattr(axis, "drawPicture", _record)
    axis.picture = None  # the cached picture would replay without calling 'drawPicture'
    image = QImage(
        ceil(bounds.width()) + 1, ceil(bounds.height()) + 1, QImage.Format.Format_RGB32
    )
    image.fill(background)
    painter = QPainter(image)
    try:
        painter.translate(-bounds.topLeft())
        axis.paint(painter, None, None)
    finally:
        painter.end()
    axis.picture = None  # drop the picture the recording proxy built
    return image, specs


def _ink(image: QImage, rect: QRect, background: QColor) -> QColor:
    """Return the pixel of ``rect`` furthest from ``background``, i.e. the glyph color.

    Text is antialiased, thus a glyph is a blend of its pen and the background; the
    pixel furthest from the background is the least blended one, and therefore the
    closest thing on screen to the pen the label was painted with.
    """
    best, best_distance = background, -1
    for x in range(max(0, rect.left()), min(image.width(), rect.right() + 1)):
        for y in range(max(0, rect.top()), min(image.height(), rect.bottom() + 1)):
            color = QColor(image.pixel(x, y))
            distance = _distance(color, background)
            if distance > best_distance:
                best, best_distance = color, distance
    return best


def test_axis_and_viewbox_are_wired(display: TraceDisplay) -> None:
    """Test that the display owns the custom axis and view box."""
    assert isinstance(display._axis, ChannelAxis)
    assert isinstance(display._vb, TraceViewBox)
    assert display._axis._display is display
    assert display._vb._display is display


def test_tick_values_one_per_row(display: TraceDisplay) -> None:
    """Test that there is one integer tick per visible row, clamped both ways."""
    display.controls.set_rows(4)
    display.scroll_to(2.0)
    lo, hi = display._vb.viewRange()[1]
    ticks = display._axis.tickValues(lo, hi, 400.0)
    assert len(ticks) == 1
    spacing, values = ticks[0]
    assert spacing == 1.0
    # the reserved event headroom sits above the top content row and grows no label.
    assert values[0] == 2.0
    assert values == [2.0, 3.0, 4.0, 5.0]
    # the bottom is clamped to the last row of the layout, not to the channel count.
    display.set_channel_layout([0, 1, 2])
    _, values = display._axis.tickValues(-1.0, 20.0, 400.0)[0]
    assert values == [0.0, 1.0, 2.0]


def test_tick_values_empty_when_no_rows(display: TraceDisplay) -> None:
    """Test that an empty layout produces no tick at all."""
    display.set_channel_layout([])
    assert display._axis.tickValues(0.0, 10.0, 400.0) == []


def test_tick_strings_names_and_bad_prefix(display: TraceDisplay) -> None:
    """Test that the labels are the current names, a bad channel prefixed with 'X '."""
    names = list(display._stream.info.ch_names)
    assert display._axis.tickStrings([0.0, 1.0], 1.0, 1.0) == names[:2]
    display._stream.info["bads"] = [names[1]]
    display.refresh_metadata()
    assert display._axis.tickStrings([0.0, 1.0], 1.0, 1.0) == [
        names[0],
        f"X {names[1]}",
    ]


def test_tick_strings_out_of_range(display: TraceDisplay) -> None:
    """Test that a value beyond the layout is an empty label, not an 'IndexError'."""
    assert display._axis.tickStrings([-1.0, 1e6], 1.0, 1.0) == ["", ""]


def test_tick_label_colors_match_traces(display: TraceDisplay) -> None:
    """Test that the axis and the traces never disagree on a channel's color.

    The cached label color is compared against the *pen of the curve*, not against
    'color_for' again: the axis and the display both ask 'color_for', thus comparing the
    cache to that same call asserts nothing at all -- the pen is what the user sees next
    to the label and the only independent witness of the color.
    """
    display._render()
    labels = display._axis.tickStrings([0.0, 1.0, 2.0], 1.0, 1.0)
    assert len(set(labels)) == 3
    for row, label in enumerate(labels):
        pen = display._assigned[row].opts["pen"]
        assert display._axis._color_by_text[label].name() == pen.color().name(), row
    # reordering must move the colors with the identities, not with the rows.
    display.set_channel_layout(list(reversed(range(display.n_channels))))
    labels = display._axis.tickStrings([0.0, 1.0], 1.0, 1.0)
    for row, label in enumerate(labels):
        pen = display._assigned[row].opts["pen"]
        assert display._axis._color_by_text[label].name() == pen.color().name(), row


def test_tick_strings_elide_a_long_name(display: TraceDisplay) -> None:
    """Test that a long channel name is elided and does not squeeze out the traces.

    'AxisItem' grows to fit its widest label, and the 30-60 character names ordinary in
    clinical LSL and EDF streams then take most of the plot.
    """
    reference = "L" * _MAX_LABEL  # the longest name painted in full
    display._stream.rename_channels({display.channel_name(0): reference})
    display.refresh_metadata()
    assert display._axis.tickStrings([0.0], 1.0, 1.0) == [reference]
    display._axis.invalidate()
    width = display._axis.width()

    display._stream.rename_channels({reference: "L" * 200})
    display.refresh_metadata()
    label = display._axis.tickStrings([0.0], 1.0, 1.0)[0]
    assert len(label) == _MAX_LABEL
    assert label == "L" * (_MAX_LABEL - 1) + "…"
    # the axis is auto-sized from its widest label, thus an elided one bounds the width.
    display._axis.invalidate()
    assert display._axis.width() <= width


def test_labels_toggle(display: TraceDisplay) -> None:
    """Test that the Labels switch turns the axis values off and invalidates it."""
    assert display._axis.style["showValues"]
    display.controls.set_state({"labels": False})
    assert not display._axis.style["showValues"]
    assert display._axis.picture is None
    display.controls.set_state({"labels": True})
    assert display._axis.style["showValues"]


@pytest.mark.parametrize("receiver", ["_axis", "_vb"])
@pytest.mark.parametrize(
    ("delta", "modifier", "top", "scaled"),
    [
        (-120, Qt.KeyboardModifier.NoModifier, 3.0, False),
        (120, Qt.KeyboardModifier.ControlModifier, 0.0, True),
    ],
)
def test_wheel_is_routed_to_the_display(
    display: TraceDisplay,
    receiver: str,
    delta: int,
    modifier: Qt.KeyboardModifier,
    top: float,
    scaled: bool,
) -> None:
    """Test that both receivers hand a notch to the display's single router.

    The axis and the view box carry the same forwarding, thus the two behaviours -- a
    plain notch scrolls, a Ctrl notch scales -- are asserted once per receiver rather
    than once per receiver and per behaviour in a test of its own.
    """
    display.controls.set_rows(4)
    ranges, scale = display._vb.viewRange(), display.controls.state["scale"]
    event = _Wheel(delta, modifier)
    getattr(display, receiver).wheelEvent(event)
    assert event.accepted
    assert display.top_offset == pytest.approx(top)
    assert (display.controls.state["scale"] > scale) is scaled
    # the plot never zooms in time, whichever the modifier.
    assert display._vb.viewRange()[0] == pytest.approx(ranges[0])
    if scaled:
        assert display._vb.viewRange()[1] == pytest.approx(ranges[1])


def test_viewbox_mouse_disabled(display: TraceDisplay) -> None:
    """Test that the view box never pans, zooms, auto-ranges nor shows a menu."""
    state = display._vb.state
    assert state["mouseEnabled"] == [False, False]
    assert state["autoRange"] == [False, False]
    assert state["yInverted"] is True  # row 0 at the top
    assert display._vb.menuEnabled() is False


def test_viewbox_drag_swallowed(display: TraceDisplay) -> None:
    """Test that a drag is ignored and leaves both ranges untouched."""
    ranges = display._vb.viewRange()
    event = _Drag()
    display._vb.mouseDragEvent(event)
    assert event.ignored
    assert display._vb.viewRange() == ranges


def test_draw_picture_colors_each_label_individually(
    app: QApplication,
    controller: ThemeController,
    shown_display: TraceDisplay,
    push: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the paint really uses one pen per label, not one pen for all of them.

    The whole reason 'drawPicture' is overridden at all: 'AxisItem' paints every label
    with a single pen. Asserting that the override ran, or that the grab holds more than
    one color, proves nothing -- the traces alone satisfy both. The two topmost labels
    are therefore read back from the painted pixels, and each has to be nearer its own
    trace color than its neighbour's, which one shared pen cannot do for both at once.
    """
    controller.install(app, "dark")  # bright trace colors on a black canvas
    push(50)
    shown_display._render()
    app.processEvents()
    background = QColor(Qt.GlobalColor.black)
    image, specs = _paint_axis(shown_display._axis, monkeypatch, background)
    assert len(specs) >= 2, "the axis painted fewer than two labels"
    assert shown_display._axis._color_by_text  # 'tickStrings' really ran
    inks = [_ink(image, rect, background) for rect, _, _ in specs[:2]]
    expected = [shown_display._axis._color_by_text[text] for _, _, text in specs[:2]]
    assert expected[0].name() != expected[1].name()
    for index, ink in enumerate(inks):
        own = _distance(ink, expected[index])
        other = _distance(ink, expected[1 - index])
        assert own < other, (index, ink.name(), expected[index].name())


def test_draw_picture_paints(
    app: QApplication, shown_display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that the hand-rolled axis paint really runs, and draws something.

    'AxisItem.drawPicture' only runs when the item is actually painted, thus the widget
    has to be shown and the scene rendered before a pixmap says anything -- and the
    paint is spied on rather than inferred, as an axis which never painted would still
    leave a non-uniform pixmap behind, out of the traces alone.
    """
    painted: list[int] = []
    original = shown_display._axis.drawPicture

    def _record(p, axis_spec, tick_specs, text_specs):
        painted.append(len(text_specs))
        original(p, axis_spec, tick_specs, text_specs)

    shown_display._axis.drawPicture = _record
    shown_display._axis.picture = None
    push(50)
    shown_display._render()
    app.processEvents()
    pixmap = shown_display._plot.grab()
    assert not pixmap.isNull()
    image = pixmap.toImage()
    colors = {
        image.pixel(x, y)
        for x in range(0, image.width(), 7)
        for y in range(0, image.height(), 7)
    }
    assert len(colors) > 1, colors
    assert painted, "the axis was never painted"
    assert painted[-1] > 0, "the axis painted no label"
    assert shown_display._axis._color_by_text  # 'tickStrings' really ran
