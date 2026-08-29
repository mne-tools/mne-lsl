from __future__ import annotations

import pkgutil
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QKeyEvent

import mne_lsl.viewer.display
from mne_lsl.viewer.controller import CH_TYPES
from mne_lsl.viewer.display import TraceDisplay, _axis, _controls, _trace
from mne_lsl.viewer.display._trace import (
    _EVENT_LABEL_MARGIN,
    _EVENT_POOL,
    _OVERSCAN,
    _RANGE_NATIVE,
    _RANGE_SI,
    _RENDER_MS,
    _ROW_FILL,
    _SB_RES,
)
from mne_lsl.viewer.theme import plot_colors, tokens, trace_color, type_color
from mne_lsl.viewer.theme._colors import _TYPE_COLORS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from pytestqt.qtbot import QtBot
    from qtpy.QtWidgets import QApplication, QToolButton, QWidget

    from mne_lsl.stream import StreamLSL
    from mne_lsl.viewer.theme import ThemeController

# Names no module of 'display/' may reach: the controller subpackage, its model, and the
# low-level LSL layer which only 'backend/' may name.
_FORBIDDEN = ("controller", "ChannelModel", "lsl", "StreamLSL")
# Every module of 'display/', for the import-rule scan. The set is asserted against
# 'pkgutil' below, so a new module cannot silently escape the check.
_MODULES = {"_axis": _axis, "_controls": _controls, "_trace": _trace}


def _press(display: TraceDisplay, key: Qt.Key) -> bool:
    """Send a key press to the display and return whether it was accepted."""
    event = QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    display.keyPressEvent(event)
    return event.isAccepted()


def _spy_get_data(display: TraceDisplay, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Wrap the stream's 'get_data' in a recording proxy and return the calls.

    Patched through 'monkeypatch' rather than assigned on the instance: the stream is
    function-scoped today, thus a permanent rebind is only harmless by accident.
    """
    calls: list[dict] = []
    original = display._stream.get_data

    def _record(winsize=None, picks=None, exclude="bads"):
        calls.append({"winsize": winsize, "picks": picks, "exclude": exclude})
        return original(winsize, picks, exclude)

    monkeypatch.setattr(display._stream, "get_data", _record)
    return calls


def _fake_get_data(
    display: TraceDisplay, monkeypatch: pytest.MonkeyPatch, n_samples: int = 40
) -> None:
    """Replace the fetch with one whose every row *is* its acquisition index.

    A live buffer keeps filling in the acquisition thread, thus a second fetch of the
    same window is not the one the render used. Encoding the identity in the samples
    instead makes 'data[row]' checkable exactly, and the picks are still the ones the
    real code computed.

    The timestamps start at 1, not at 0: a 0.0 timestamp is what the display reads as an
    un-filled buffer, so starting at 0 would make every fetch of this stand-in NaN its
    own first sample and every identity assertion above fail on that one column.
    """

    def _fetch(winsize=None, picks=None, exclude="bads"):
        data = np.repeat(
            np.asarray(picks, dtype=np.float64)[:, None], n_samples, axis=1
        )
        return data, np.arange(1, n_samples + 1, dtype=np.float64) / 100.0

    monkeypatch.setattr(display._stream, "get_data", _fetch)


def _pen_color(display: TraceDisplay, row: int) -> str:
    """Return the name of the pen color of an assigned row."""
    return display._assigned[row].opts["pen"].color().name()


def _mark_bad(display: TraceDisplay, name: str) -> None:
    """Mark a channel bad on the stream info, as the Channels page does."""
    display._stream.info["bads"] = [name]


def _visible_events(display: TraceDisplay) -> int:
    """Return the number of visible event lines, labels included in the count check."""
    lines = sum(line.isVisible() for line in display._event_lines)
    assert lines == sum(label.isVisible() for label in display._event_labels)
    return lines


def _stim_window(
    display: TraceDisplay, samples: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a ``(data, x)`` window whose stim row carries ``samples``."""
    x = np.linspace(0.0, display._winsize, len(samples))
    data = np.zeros((len(display._picks), x.size), dtype=np.float64)
    data[display._event_pos[0]] = samples
    return data, x


def test_construction_defaults(display: TraceDisplay, stream: StreamLSL) -> None:
    """Test that the display starts in acquisition order with everything visible."""
    assert display.n_channels == len(stream.info.ch_names)
    assert display._rows == list(range(display.n_channels))
    assert display.n_rows == display.n_channels
    assert display.n_visible == 20  # the documented default
    assert display.top_offset == 0.0
    assert display.controls.state["rows"] == 20
    # the render clock exists but is the caller's to start.
    assert display.running is False
    assert display._timer.interval() == _RENDER_MS


def test_start_stop_running(display: TraceDisplay) -> None:
    """Test that the render clock is public and both calls are idempotent."""
    display.start()
    assert display.running
    display.start()
    assert display.running
    display.stop()
    assert not display.running
    display.stop()
    assert not display.running


def test_render_draws_data(display: TraceDisplay, push: Callable[..., None]) -> None:
    """Test the end-to-end picks -> data[row] -> setData path, on real samples.

    The rows must not merely carry samples, they must carry *different* samples: every
    channel of the fixture is a distinct sine, thus a fetch which handed the same row
    to every curve would still fill them all.
    """
    push(50)
    display._render()
    assert display._assigned
    with_data = 0
    drawn: set[bytes] = set()
    for curve in display._assigned.values():
        _, y = curve.getData()
        if y is not None and np.any(np.abs(y) > 1e-9):
            with_data += 1
            drawn.add(np.ascontiguousarray(y).tobytes())
    assert with_data > 1
    assert len(drawn) == with_data


def test_every_curve_draws_its_own_channel(
    display: TraceDisplay, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a curve draws the samples of the channel of its own row.

    'data[row]' is the row's samples only because the picks prefix is the layout; a
    fetch row taken from a constant index would draw one channel everywhere, with no
    exception and nothing else to notice it.
    """
    _fake_get_data(display, monkeypatch)
    display.set_channel_layout([5, 2, 7, 0])  # renders at the end
    assert display._assigned
    for row, curve in display._assigned.items():
        assert np.all(curve.yData == display._rows[row]), row


def test_picks_never_none_and_exclude_empty(
    display: TraceDisplay, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the fetch uses explicit integer picks and an empty exclude."""
    calls = _spy_get_data(display, monkeypatch)
    display._render()
    assert len(calls) == 1
    assert calls[0]["picks"] == display._picks
    assert all(isinstance(pick, int) for pick in calls[0]["picks"])
    assert calls[0]["exclude"] == ()
    assert calls[0]["winsize"] == display.controls.state["window"]


def test_picks_prefix_invariant(display: TraceDisplay) -> None:
    """Test that the picks always start with the layout, in order.

    'data[row]' is the row's own samples only because of this; sorting or deduplicating
    the picks would silently draw the wrong channel, with no exception at all.
    """
    assert display._picks[: display.n_rows] == display._rows
    display.set_channel_layout(list(reversed(range(display.n_channels))))
    assert display._picks[: display.n_rows] == display._rows
    display.set_channel_layout([3, 1, 0])
    assert display._picks[: display.n_rows] == display._rows


def test_fixed_x_axis(display: TraceDisplay) -> None:
    """Test that the x range is pinned to [0, W], whatever the data flow."""
    display._render()
    assert display._vb.viewRange()[0] == pytest.approx([0.0, 5.0])
    display.controls.set_window(8.0)
    display._render()
    assert display._vb.viewRange()[0] == pytest.approx([0.0, 8.0])


def test_samples_map_onto_the_window(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that the samples are drawn against relative time, newest at the right edge.

    Pinning the view range to [0, W] is only half of the fixed axis: the samples carry
    absolute LSL timestamps, thus they have to be mapped onto that same [0, W], or every
    trace is drawn far off-screen while the range still reads [0, W].
    """
    push(50)
    display._render()
    # the raw arrays, as 'getData' returns the clipped and downsampled ones.
    x = display._assigned[0].xData
    assert x.max() == pytest.approx(display.controls.state["window"])
    # x must not decrease, which is what clipping and peak downsampling need; it is
    # strictly increasing only over the filled tail, as the yet-unwritten prefix of the
    # buffer carries a zero timestamp and maps to one constant, far-negative x.
    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(x[-50:]) > 0)
    display.controls.set_window(2.0)
    display._render()
    assert display._assigned[0].xData.max() == pytest.approx(2.0)


def test_curves_are_stacked_one_per_row(display: TraceDisplay) -> None:
    """Test that every curve is parked on the y-slot of its own row.

    The stacking is the position of the curve, not a rewrite of its samples: that is
    what makes the pool order-invariant and a layout change free of any allocation.
    """
    display._render()
    assert len(display._assigned) > 1
    for row, curve in display._assigned.items():
        assert curve.pos().x() == pytest.approx(0.0)
        assert curve.pos().y() == pytest.approx(row)


def test_scroll_clamping(display: TraceDisplay) -> None:
    """Test that the offset is clamped to the extent and keeps its fraction."""
    display.scroll_to(-5.0)
    assert display.top_offset == 0.0
    display.scroll_to(1e9)
    assert display.top_offset == max(0.0, display.n_rows - display.n_visible)
    display.controls.set_rows(2)  # 8 channels, 2 rows visible: the extent is 6
    display.scroll_to(3.25)
    assert display.top_offset == pytest.approx(3.25)
    display.scroll_by(1.5)
    assert display.top_offset == pytest.approx(4.75)
    display.scroll_to(1e9)
    assert display.top_offset == pytest.approx(6.0)


def test_scroll_never_exhausts_pool(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable]],
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that the pool covers the whole extent at fractional offsets.

    The pool holds exactly one spare curve over the largest band, thus walking it
    in half rows at the maximum row count is what would surface an off-by-one as an
    'IndexError' from '_free.pop()'.
    """
    stream, _ = lsl_stream(n_channels=80)
    display = make_display(stream)
    display.controls.set_rows(60)
    assert len(display._pool) == 60 + 2 * _OVERSCAN + 2
    offset = 0.0
    while offset <= display.n_rows:
        display.scroll_to(offset)  # clamps, thus the band follows 'top_offset'
        display._render()
        top = display.top_offset
        lo = max(0, int(np.floor(top)) - _OVERSCAN)
        hi = min(display.n_rows, int(np.ceil(top + 60)) + _OVERSCAN)
        assert set(display._assigned) == set(range(lo, hi))
        offset += 0.5


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (Qt.Key.Key_Down, 1.0),
        (Qt.Key.Key_PageDown, 20.0),
        (Qt.Key.Key_End, 40.0),
    ],
)
def test_keyboard_scroll_down(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable]],
    make_display: Callable[..., TraceDisplay],
    key: Qt.Key,
    expected: float,
) -> None:
    """Test that the downward keys move the offset by the documented amount."""
    stream, _ = lsl_stream(n_channels=60)
    display = make_display(stream)
    assert _press(display, key)
    assert display.top_offset == pytest.approx(expected)


def test_keyboard_scroll_up_and_home(display: TraceDisplay) -> None:
    """Test that the upward keys and Home walk back to the top."""
    display.controls.set_rows(2)  # 8 channels, 2 rows visible: the extent is 6
    display.scroll_to(6.0)
    assert _press(display, Qt.Key.Key_Up)
    assert display.top_offset == pytest.approx(5.0)
    assert _press(display, Qt.Key.Key_PageUp)
    assert display.top_offset == pytest.approx(3.0)
    display.scroll_to(3.0)
    assert _press(display, Qt.Key.Key_Home)
    assert display.top_offset == 0.0


def test_keyboard_other_keys_are_not_accepted(display: TraceDisplay) -> None:
    """Test that an unrelated key is left to the parent chain."""
    assert not _press(display, Qt.Key.Key_A)


def test_wheel_scrolls_never_scales(display: TraceDisplay) -> None:
    """Test that a plain wheel notch scrolls only.

    The legacy viewer silently changed the amplitude on a plain wheel event, which is
    the behaviour this display deliberately does not reproduce.
    """
    display.controls.set_rows(4)
    scale = display.controls.state["scale"]
    display.on_wheel(-120, Qt.KeyboardModifier.NoModifier)
    assert display.top_offset == pytest.approx(3.0)
    assert display.controls.state["scale"] == scale
    assert display._amp_mult == scale


def test_ctrl_wheel_scales_only(display: TraceDisplay) -> None:
    """Test that Ctrl+wheel scales through the bar and does not scroll."""
    top = display.top_offset
    display.on_wheel(120, Qt.KeyboardModifier.ControlModifier)
    assert display.controls.state["scale"] > 1.0
    assert display._amp_mult == display.controls.state["scale"]
    assert display.top_offset == top
    display.on_wheel(-120, Qt.KeyboardModifier.ControlModifier)
    assert display.controls.state["scale"] == pytest.approx(1.0)


def test_set_channel_layout_reorder_keeps_pool(display: TraceDisplay) -> None:
    """Test that a reorder repaints the rows without touching the pool.

    A layout change must create, destroy, release and reposition nothing: the curves are
    bound to y-slots, only the identity beneath each row changes.
    """
    display._render()
    pool = list(display._pool)
    assigned = dict(display._assigned)
    first = display.channel_name(0)
    display.set_channel_layout(list(reversed(range(display.n_channels))))
    assert display.channel_name(0) != first
    assert display._pool == pool
    assert set(display._assigned) == set(assigned)
    for row, curve in assigned.items():
        assert display._assigned[row] is curve


def test_set_channel_layout_renders_immediately(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that a layout change never leaves a stale frame on screen.

    Without the render at the end of 'set_channel_layout', a curve would keep the old
    channel's samples under the new channel's pen for up to one render period.

    'equal_nan=True' is mandatory here: fewer samples are pushed than the window holds,
    so the un-filled prefix is drawn as NaN, and 'np.array_equal' is 'False' for two
    *identical* NaN-bearing arrays. Without it this negative assertion holds whatever
    the display does, i.e. it becomes a test which cannot fail.
    """
    push(50)
    display._render()
    before = display._assigned[0].getData()[1].copy()
    display.set_channel_layout(list(reversed(range(display.n_channels))))
    after = display._assigned[0].getData()[1]  # no explicit '_render()' in between
    assert not np.array_equal(before, after, equal_nan=True)


def test_color_invariance_under_layout(display: TraceDisplay) -> None:
    """Test that a channel keeps its color through a reorder and a hide.

    The color is seeded by the acquisition index, never by the row, thus reordering and
    hiding must recolor nothing.
    """
    acq = 3
    expected = trace_color(acq, display._mode).name()
    assert display.color_for(acq).name() == expected
    display.set_channel_layout(list(reversed(range(display.n_channels))))
    assert display.color_for(display._rows.index(acq)).name() == expected
    display.set_channel_layout([5, acq])
    assert display.color_for(1).name() == expected


def test_all_hidden_guard(
    display: TraceDisplay, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that hiding every channel neither renders nor fetches.

    'get_data(picks=[])' raises *and* logs an error asking for a bug report, which at
    30 Hz would fill the user's terminal.
    """
    display._render()
    calls = _spy_get_data(display, monkeypatch)
    display.set_channel_layout([])
    assert display.n_rows == 0
    assert display._assigned == {}
    assert all(not curve.isVisible() for curve in display._pool)
    assert display._empty_label.isVisible()
    assert display._axis.tickValues(0.0, 10.0, 100.0) == []
    display._render()  # must not raise
    assert calls == []


def test_recovery_from_all_hidden(display: TraceDisplay) -> None:
    """Test that a non-empty layout renders again after everything was hidden."""
    display._render()
    display.set_channel_layout([])
    assert len(display._free) == len(display._pool)
    display.set_channel_layout(list(range(display.n_channels)))
    display._render()
    assert display._assigned
    assert not display._empty_label.isVisible()
    assert len(display._free) + len(display._assigned) == len(display._pool)


def test_hidden_channel_shrinks_picks(display: TraceDisplay) -> None:
    """Test that hiding a non-stim channel drops it from the fetch."""
    total = display.n_channels
    display.set_channel_layout([idx for idx in range(total) if idx != 0])
    assert display.n_channels == total  # the status-bar count is unchanged
    assert display.n_rows == total - 1
    assert 0 not in display._picks


def test_events_survive_hidden_stim(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that hiding the stim channel keeps its overlays and drops its trace.

    The overlays belong to the Events page, not to channel visibility.
    """
    stim = display._event_acq[0]
    display.set_channel_layout(
        [idx for idx in range(display.n_channels) if idx != stim]
    )
    assert stim not in display._rows
    assert stim in display._picks
    assert display._event_pos == [len(display._picks) - 1]
    push(50, stim_at=10)
    display._render()
    assert any(line.isVisible() for line in display._event_lines)


def test_event_labels_in_headroom(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that every event label sits in the reserved headroom above the top row."""
    push(50, stim_at=10)
    display._render()
    labels = [label for label in display._event_labels if label.isVisible()]
    assert labels
    # invertY: above the top row means a smaller y.
    assert all(label.pos().y() < display.top_offset - 0.5 for label in labels)


def test_events_toggle_hides_overlays(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that the Events switch hides every line and label."""
    push(50, stim_at=10)
    display._render()
    assert any(line.isVisible() for line in display._event_lines)
    display.controls.set_state({"events": False})
    assert not any(line.isVisible() for line in display._event_lines)
    assert not any(label.isVisible() for label in display._event_labels)
    display._render()  # a tick must not bring them back
    assert not any(line.isVisible() for line in display._event_lines)


def test_event_pool_is_bounded(display: TraceDisplay) -> None:
    """Test that more edges than the pool holds place exactly the pool's worth."""
    x = np.linspace(0.0, 5.0, 200)
    data = np.zeros((len(display._picks), x.size), dtype=np.float32)
    data[display._event_pos[0], ::2] = 5.0  # every other sample is a rising edge
    display._update_events(data, x)
    assert sum(line.isVisible() for line in display._event_lines) == _EVENT_POOL
    assert sum(label.isVisible() for label in display._event_labels) == _EVENT_POOL


def test_refresh_metadata_rename_and_bad(display: TraceDisplay) -> None:
    """Test that the names and the bads are read live rather than snapshotted."""
    display._render()
    name = display.channel_name(0)
    display._stream.rename_channels({name: "renamed"})
    _mark_bad(display, "renamed")
    assert display.channel_name(0) == name  # not yet refreshed
    display.refresh_metadata()
    assert display.channel_name(0) == "renamed"
    assert display.is_bad(0)
    assert display._axis.tickStrings([0.0], 1.0, 1.0) == ["X renamed"]
    assert _pen_color(display, 0) == QColor(tokens(display._mode).bad).name()


def test_refresh_metadata_type_change_retransforms(display: TraceDisplay) -> None:
    """Test that a type change re-applies the transform, as the gain follows it."""
    display._render()
    before = display._assigned[0].transform().m22()
    display._stream.set_channel_types(
        {display.channel_name(0): "ecg"}, on_unit_change="ignore"
    )
    display.refresh_metadata()
    assert display._types[0] == "ecg"
    assert display._assigned[0].transform().m22() != before


def test_gain_is_unit_aware(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable]],
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that the base gain follows the declared unit, and keeps the accepted look.

    A stream declaring volts and one declaring microvolts must render alike, and an
    EEG channel declared in microvolts must land on the fixed gain of the accepted
    prototype, i.e. 0.012.
    """
    micro, _ = lsl_stream(units="uv")
    volt, _ = lsl_stream(units="v")
    displays = (make_display(micro), make_display(volt))
    expected = _ROW_FILL / (_RANGE_SI["eeg"] * 1e6)
    assert expected == pytest.approx(0.012)
    assert displays[0]._gain[0] == pytest.approx(expected)
    assert displays[1]._gain[0] == pytest.approx(expected * 1e6)


def test_gain_reads_the_units_of_every_channel(display: TraceDisplay) -> None:
    """Test that the gains and the types stay aligned on the acquisition indices.

    Both lists are indexed by acquisition index and are read in one pass, thus a
    metadata read which dropped or reordered a channel would pair a gain, or a type,
    with the wrong one -- silently, as nothing downstream can tell a wrong gain from a
    deliberate one. This cannot pin the 'get_channel_units' bad-channel exclusion: the
    metadata read passes explicit integer picks, which bypass 'exclude' entirely, and
    that fix is pinned in 'tests/stream/test_stream_lsl.py'. A bad channel is marked
    here only because it is the state which used to shorten the units list.
    """
    _mark_bad(display, display._stream.info.ch_names[0])
    display.refresh_metadata()
    assert len(display._gain) == display.n_channels
    assert len(display._types) == display.n_channels
    assert display._types[-1] == "stim"
    assert display._gain[0] == pytest.approx(display._gain[1])


def test_bad_and_trace_colors_follow_theme(
    app: QApplication, display: TraceDisplay, controller: ThemeController
) -> None:
    """Test that a theme flip recolors the traces and leaves the layout untouched."""
    controller.install(app, "light")
    _mark_bad(display, display.channel_name(1))
    display.refresh_metadata()
    display._render()
    rows, n_rows = list(display._rows), display.n_rows
    light_trace = _pen_color(display, 0)
    assert _pen_color(display, 1) == QColor(tokens("light").bad).name()

    controller.set_mode("dark")
    assert display._mode == "dark"
    assert _pen_color(display, 0) != light_trace
    assert _pen_color(display, 1) == QColor(tokens("dark").bad).name()
    assert display._rows == rows
    assert display.n_rows == n_rows


def test_theme_flip_retints_bar_icons(
    app: QApplication,
    display: TraceDisplay,
    controller: ThemeController,
    tool_button: Callable[[QWidget, str], QToolButton],
) -> None:
    """Test that a theme flip reaches the bar icons, which bake their color."""
    button = tool_button(display.controls, "Visible rows: increase")
    controller.install(app, "light")
    light = button.icon().pixmap(16, 16).toImage()
    controller.set_mode("dark")
    dark = button.icon().pixmap(16, 16).toImage()
    assert not light.isNull()
    assert light != dark


def _assert_overlay_colors(display: TraceDisplay, mode: str) -> None:
    """Assert that every event overlay and the placeholder are colored for ``mode``."""
    background = QColor(plot_colors(mode)["background"]).name()
    accent = QColor(tokens(mode).success).name()
    assert display._headband.brush().color().name() == background
    assert display._event_lines[0].pen.color().name() == accent
    # the label fill masks the traces behind the value, thus it is the plot background.
    assert display._event_labels[0].fill.color().name() == background
    assert display._event_labels[0].textItem.defaultTextColor().name() == accent
    assert display._empty_label.textItem.defaultTextColor().name() == (
        QColor(tokens(mode).text_secondary).name()
    )


def test_theme_flip_recolors_the_overlays(
    app: QApplication, display: TraceDisplay, controller: ThemeController
) -> None:
    """Test that every event overlay and the placeholder follow the theme.

    All five items are asserted, not only the two which happen to be easiest to read
    back: an overlay left out of ':meth:`_style_overlays`' is readable in one mode only,
    and nothing else in the display would ever notice.
    """
    controller.install(app, "light")
    light_band = display._headband.brush().color().name()
    _assert_overlay_colors(display, "light")
    controller.set_mode("dark")
    assert display._headband.brush().color().name() != light_band
    _assert_overlay_colors(display, "dark")


def test_overlays_ignore_the_pyqtgraph_config(
    app: QApplication,
    controller: ThemeController,
    pg_background: Callable[[str], None],
    stream: StreamLSL,
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that the overlays are colored from the theme, not from pyqtgraph's config.

    A display can be built before a theme was ever applied, i.e. while the pyqtgraph
    configuration still holds its own defaults; reading the background from there would
    then mask the traces with a color from another mode.
    """
    controller.install(app, "light")
    pg_background("#ff00ff")  # what the theme never sets
    display = make_display(stream)
    _assert_overlay_colors(display, "light")


def test_rows_change_resizes_pool(display: TraceDisplay) -> None:
    """Test that the row count resizes the pool and leaves the layout alone."""
    rows = list(display._rows)
    display.scroll_to(1e9)
    display.controls.set_rows(30)
    assert len(display._pool) == 30 + 2 * _OVERSCAN + 2
    assert display.n_visible == 30
    assert display._rows == rows
    assert display.top_offset == max(0.0, display.n_rows - 30)


def test_stopped_clock_follows_the_row_count(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that a row-count change repaints while the render clock is stopped.

    A stopped clock is what the document's Freeze is. '_on_rows' rebuilds the pool, thus
    without a repaint from the retained window the plot is left with **zero** curves --
    an empty viewport, while the status bar still reports 'n/N ch'.
    """
    push(50)
    display._render()
    display.stop()
    assert len(display._assigned) == display.n_rows
    display.controls.set_rows(4)
    assert display._assigned
    assert set(display._assigned) == set(range(min(display.n_rows, 4 + _OVERSCAN)))
    for row, curve in display._assigned.items():
        assert curve.getData()[1] is not None, row
        assert curve.getData()[1].size, row


def test_stopped_clock_follows_a_scroll(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable]],
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that a scroll repaints the band while the render clock is stopped.

    Nothing repaints on a scroll by itself, thus a frozen viewport scrolled onto rows it
    had not banded yet showed them blank until it was unfrozen.
    """
    stream, push = lsl_stream(n_channels=16)
    display = make_display(stream)
    display.controls.set_rows(4)
    push(50)
    display.scroll_to(0.0)
    display._render()
    display.stop()
    display.scroll_by(6)
    assert display.top_offset == pytest.approx(6.0)
    for row in range(6, 10):  # the four rows the viewport now shows
        assert row in display._assigned, row
        assert display._assigned[row].getData()[1].size, row


def test_stopped_clock_repaints_the_same_window(
    display: TraceDisplay, push: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a repaint over a stopped clock never polls the stream again.

    The whole point of the retained window: a poll would advance the viewport to the
    newest samples, so a frozen document would jump forward on any interaction at all.

    'equal_nan=True' is mandatory on both comparisons: fewer samples are pushed than the
    window holds, so the un-filled prefix is drawn as NaN, and 'np.array_equal' is
    'False' for two *identical* NaN-bearing arrays -- which fails the positive assertion
    and makes the negative one hold whatever the display does.
    """
    push(50)
    display._render()
    display.stop()
    acq = display._rows[0]
    before = display._assigned[0].getData()[1].copy()
    calls = _spy_get_data(display, monkeypatch)
    push(50)
    display.scroll_by(1)
    display.controls.set_rows(4)
    display.controls.set_window(1.0)
    assert calls == []
    assert np.array_equal(display._assigned[0].getData()[1], before, equal_nan=True)
    # a hide is a subset of the retained picks, thus it too repaints without a poll --
    # and the channel which moved onto row 0 draws its own samples out of that window.
    display.set_channel_layout(list(range(1, display.n_channels)))
    assert calls == []
    assert display._rows[0] != acq
    assert not np.array_equal(display._assigned[0].getData()[1], before, equal_nan=True)


def test_disconnected_stream_render_is_noop(display: TraceDisplay) -> None:
    """Test that a render tick on a disconnected stream returns quietly."""
    display._stream.disconnect()
    display._render()  # the interface response to a lost stream is Phase G's


def test_close_does_not_disconnect_stream(
    display: TraceDisplay, stream: StreamLSL
) -> None:
    """Test that closing the display leaves the borrowed stream connected."""
    display.start()
    display.close()
    assert not display.running
    assert stream.connected


def test_close_is_idempotent(display: TraceDisplay) -> None:
    """Test that closing twice does not raise on the dropped theme connection."""
    display.close()
    display.close()


# -- render clock ----------------------------------------------------------------------
def test_render_clock_repaints_and_stops(
    qtbot: QtBot,
    display: TraceDisplay,
    push: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the clock really drives '_render', and that 'stop' really stops it.

    Every other test of this module calls '_render' by hand, thus the one connection
    which makes the display repaint on its own is exercised nowhere else: dropping it
    leaves a frozen plot and no failing assertion at all. A completed fetch is the proof
    a whole render ran, not merely that the timer fired.
    """
    push(50)
    calls = _spy_get_data(display, monkeypatch)
    display.start()
    qtbot.waitUntil(lambda: len(calls) > 0, timeout=5000)
    display.stop()
    qtbot.wait(_RENDER_MS * 4)  # drain whatever the clock had already queued
    settled = len(calls)
    qtbot.wait(_RENDER_MS * 4)
    assert len(calls) == settled


# -- the un-filled window, and the polling report --------------------------------------
def test_unfilled_window_is_drawn_as_nan(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that the un-filled part of the buffer is NaN and the real samples are not.

    'connect()' allocates the timestamps with 'np.zeros', so an un-filled region reads
    exactly 0.0 while a real timestamp never does. Drawn as samples it joins across the
    outage instead of leaving a gap. Kills deleting the rule, kills NaN-ing the filled
    samples too, and kills matching on anything other than an exact 0.0.
    """
    push(50)
    display._render()
    _, data, _ = display._frame
    _, ts = display._stream.get_data(display.controls.state["window"], picks=[0])
    unfilled = ts == 0.0
    assert unfilled.sum() == 150  # 2 s at 100 Hz, 50 samples pushed
    assert np.all(np.isnan(data[:, unfilled]))
    assert not np.any(np.isnan(data[:, ~unfilled]))


def test_unfilled_window_of_an_integer_stream(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable[..., None]]],
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that an integer stream renders and gets a float frame with NaNs in it.

    'NaN' cannot be stored in an integer array, so without the promotion the assignment
    raises 'ValueError' on every tick of an int8/16/32/64 stream -- inside a Qt slot, at
    30 Hz. Kills dropping the dtype guard.
    """
    stream, push = lsl_stream(dtype="int32")
    assert np.dtype(stream.dtype) == np.int32
    display = make_display(stream)
    push(50)
    display._render()  # would raise 'ValueError' without the promotion
    _, data, _ = display._frame
    assert data.dtype.kind == "f"
    assert np.isnan(data).any()
    assert not np.isnan(data[:, -1]).any()  # the real samples survived the promotion


def test_unfilled_zeros_are_not_a_rising_stim_edge(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that the un-filled buffer never produces an event overlay.

    The edge rule is an exact ``== 0`` to non-zero transition, and an un-filled buffer
    reads exactly 0.0 -- so a stim sample which is high in the *first* pushed chunk sits
    immediately after the un-filled region and used to be reported as a rising edge,
    with a line and a value label, for a transition which never happened on the wire.
    The NaN rule is what removes it: ``NaN == 0`` is 'False'.

    The two shipped NaN tests cannot see this, because the fixture pushes zeros into the
    stim channel and they assert on the frame rather than on the overlays. Both edges
    live in one window here, so the same fetch shows the false one suppressed and the
    real one kept -- an assertion which cannot pass by drawing nothing at all.
    """
    push(50, stim_at=0)  # high on the very first sample which exists
    display._render()
    _, data, _ = display._frame
    stim = data[display._event_pos[0]]
    first = int(np.flatnonzero(stim > 0)[0])
    assert np.isnan(stim[first - 1])  # the un-filled sample is not a zero
    assert _visible_events(display) == 0

    push(50, stim_at=10)  # a real 0 -> 3 transition, inside the filled region
    display._render()
    assert _visible_events(display) == 1


def test_polled_is_emitted_on_every_branch(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that a tick reports itself whether or not it drew anything.

    'polled' is the only clock a consumer gets out of this widget, so a branch which
    does not emit is a consumer which stops being told anything precisely when something
    went wrong. Kills moving the emit inside any branch of '_render'.
    """
    ticks: list[int] = []
    display.polled.connect(lambda: ticks.append(1))
    push(50)
    display._render()  # connected and drawing
    assert len(ticks) == 1
    display.set_channel_layout([])  # all hidden: the fetch is skipped
    ticks.clear()
    display._render()
    assert len(ticks) == 1
    display._stream.disconnect()
    ticks.clear()
    display._render()
    assert len(ticks) == 1


def test_poll_survives_a_stream_reset_under_the_fetch(
    display: TraceDisplay,
    push: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a stream lost under the fetch ends the tick instead of raising out.

    The connection guard cannot cover the read which follows it: the acquisition thread
    resets the stream on its own account, so a source lost in between raises out of a
    'QTimer' slot 30 times a second, for a routine disconnection the next tick handles
    on the guard. A failure over a stream which is *still* connected is re-raised
    instead, which is the half a bare 'except' would swallow -- it is the one 'get_data'
    logs a bug report for.
    """
    push(50)
    display._render()

    def _raise(winsize=None, picks=None, exclude="bads"):
        raise RuntimeError("The Stream is not connected.")

    monkeypatch.setattr(display._stream, "get_data", _raise)
    with pytest.raises(RuntimeError, match="not connected"):
        display._render()

    def _lose(winsize=None, picks=None, exclude="bads"):
        display._stream.disconnect()  # the reset lands between the guard and the read
        raise RuntimeError("The Stream is not connected.")

    monkeypatch.setattr(display._stream, "get_data", _lose)
    display._render()
    assert not display._stream.connected


def test_last_timestamp(display: TraceDisplay, push: Callable[..., None]) -> None:
    """Test the newest timestamp of the last fetched window, over its three states.

    Kills reading it off 'relative', which ends at 0.0 by construction, and kills
    setting it after the trim, which would report 'nan' on a partially filled buffer.
    """
    assert display.last_timestamp is None  # nothing fetched yet
    display._render()
    assert display.last_timestamp == 0.0  # nothing pushed: the buffer is all zeros
    push(50)
    display._render()
    first = display.last_timestamp
    assert first > 0.0
    display._render()  # a re-poll with no new samples must not move it
    assert display.last_timestamp == first


# -- amplitude transform ---------------------------------------------------------------
def test_amp_transform_is_signed_and_y_only(display: TraceDisplay) -> None:
    """Test that the transform scales y alone, negatively, by gain x multiplier.

    The minus is what compensates 'invertY(True)': without it every trace is drawn
    upside down, which no assertion on a magnitude can see.
    """
    display._render()
    mult = display.controls.state["scale"]
    assert display._assigned
    for row, curve in display._assigned.items():
        transform = curve.transform()
        assert transform.m11() == pytest.approx(1.0)  # x untouched: clipping needs it
        assert transform.m22() == pytest.approx(
            -display._gain[display._rows[row]] * mult
        )


def test_scale_change_retransforms_every_curve(display: TraceDisplay) -> None:
    """Test that a new amplitude multiplier reaches every assigned curve.

    The bar publishes the multiplier and the display caches it, thus the stepper and
    Ctrl+wheel are cosmetic unless the transforms are re-applied.
    """
    display._render()
    before = {row: curve.transform().m22() for row, curve in display._assigned.items()}
    assert before
    display.controls.set_scale(2.0)
    assert display._amp_mult == 2.0
    for row, curve in display._assigned.items():
        assert curve.transform().m22() == pytest.approx(before[row] * 2.0), row


def test_pen_is_one_pixel_wide_and_solid(display: TraceDisplay) -> None:
    """Test that a trace is drawn hairline and solid, as a wall of them has to be."""
    display._render()
    pen = display._assigned[0].opts["pen"]
    assert pen.width() == 1
    assert pen.style() == Qt.PenStyle.SolidLine


def test_curves_clip_and_downsample(display: TraceDisplay) -> None:
    """Test that every pooled curve clips to the view and peak-downsamples."""
    assert display._pool
    for curve in display._pool:
        assert curve.opts["clipToView"] is True
        assert curve.opts["autoDownsample"] is True
        assert curve.opts["downsampleMethod"] == "peak"


# -- color modes -----------------------------------------------------------------------
def test_type_color_mode_repens_every_curve(display: TraceDisplay) -> None:
    """Test that the type mode pens every curve with the color of its channel type.

    Nothing else puts a *display* into type mode, thus a fixed color, or a color mode
    change which never re-pens, would go unseen: the bar would publish the mode and the
    plot would keep the channel colors.
    """
    display._render()
    channel_pens = {row: _pen_color(display, row) for row in display._assigned}
    display.controls.set_state({"color_mode": "type"})
    assert display._color_mode == "type"
    for row in display._assigned:
        acq = display._rows[row]
        assert _pen_color(display, row) == type_color(display._types[acq]).name()
    # the two types of the fixture are colored differently, i.e. the mode is not one
    # constant color, and the eeg rows genuinely moved off their per-channel color.
    stim_row = display._rows.index(display._event_acq[0])
    assert _pen_color(display, 0) != _pen_color(display, stim_row)
    assert _pen_color(display, 0) != channel_pens[0]


def test_bad_color_wins_over_the_type_color(display: TraceDisplay) -> None:
    """Test that a bad channel keeps the bad color in type mode.

    The bad state is the more urgent cue of the two and must not be swallowed by the
    color mode, which is the only thing distinguishing the two orderings of the test.
    """
    _mark_bad(display, display.channel_name(1))
    display.refresh_metadata()
    display.controls.set_state({"color_mode": "type"})
    display._render()
    bad = QColor(tokens(display._mode).bad).name()
    assert _pen_color(display, 1) == bad
    assert _pen_color(display, 0) == type_color(display._types[display._rows[0]]).name()
    assert _pen_color(display, 0) != bad


# -- scrollbar -------------------------------------------------------------------------
def test_scrollbar_tracks_the_offset(display: TraceDisplay) -> None:
    """Test that the thumb, the range and both steps follow the display.

    The scrollbar is the only visible read-out of the vertical position: a thumb which
    never moves, or a page step left at the Qt default of 10 units, i.e. a tenth of a
    row, is a control which lies about where the viewport is.
    """
    display.controls.set_rows(4)  # 8 channels, 4 visible: the extent is 4 rows
    display.scroll_to(2.5)
    assert display._scroll.value() == 250
    assert display._scroll.minimum() == 0
    assert display._scroll.maximum() == 400
    assert display._scroll.pageStep() == 4 * _SB_RES
    assert display._scroll.singleStep() == _SB_RES


def test_scrollbar_drag_scrolls_the_display(display: TraceDisplay) -> None:
    """Test that moving the thumb moves the viewport, in sub-row units."""
    display.controls.set_rows(4)
    display._scroll.setValue(275)
    assert display.top_offset == pytest.approx(2.75)
    display._scroll.setValue(0)
    assert display.top_offset == 0.0


def test_scrollbar_refresh_does_not_feed_back(display: TraceDisplay) -> None:
    """Test that pushing the offset to the scrollbar does not quantize it.

    '_apply_scroll' writes the thumb, whose 'valueChanged' is wired back to
    '_on_scrollbar' and thence to '_apply_scroll'. The signal block is the only thing
    breaking that loop; without it the round trip snaps the offset onto whole scrollbar
    units, and fractional scrolling -- the whole point of the overscan band -- silently
    disappears.
    """
    display.controls.set_rows(4)
    display.scroll_to(1.3333)
    assert display.top_offset == pytest.approx(1.3333)
    assert display._scroll.value() == round(1.3333 * _SB_RES)


# -- event overlays --------------------------------------------------------------------
def test_event_line_and_label_placement(display: TraceDisplay) -> None:
    """Test that an edge places its line and its value label on that very sample.

    Both the x of the line and the text of the label are write-only as far as the rest
    of the display is concerned: lines stacked at the origin, or blank labels, keep
    every visibility assertion green.
    """
    data, x = _stim_window(display, [0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    display._update_events(data, x)
    assert _visible_events(display) == 1
    assert display._event_lines[0].value() == pytest.approx(x[3])
    assert display._event_labels[0].pos().x() == pytest.approx(x[3])
    assert display._event_labels[0].textItem.toPlainText() == "3"
    # the label is parked in the middle of the reserved headroom above the top row.
    assert display._event_labels[0].pos().y() == pytest.approx(
        display.top_offset - 0.5 - _EVENT_LABEL_MARGIN / 2
    )


def test_event_edge_is_a_zero_to_nonzero_transition(display: TraceDisplay) -> None:
    """Test that a plateau is one edge, not one edge per sample above zero.

    A plateau is what a real trigger looks like: an alternating pattern overflows the
    overlay pool under either semantics and therefore cannot tell them apart.
    """
    data, x = _stim_window(display, [0.0, 3.0, 3.0, 0.0, 3.0])
    display._update_events(data, x)
    assert _visible_events(display) == 2
    assert display._event_lines[0].value() == pytest.approx(x[1])
    assert display._event_lines[1].value() == pytest.approx(x[4])


def test_infinite_stim_sample_is_not_an_edge(display: TraceDisplay) -> None:
    """Test that a non-finite stim sample raises nothing and leaves nothing behind.

    'inf > 0' is true, thus an infinite sample used to be detected as an edge and then
    raised on the label conversion -- from inside the render tick, which froze the
    previous frame's overlays on screen and repeated the failure at every tick.
    """
    data, x = _stim_window(display, [0.0, 0.0, 3.0, 0.0, 0.0])
    display._update_events(data, x)
    assert _visible_events(display) == 1
    data, x = _stim_window(display, [0.0, 0.0, 0.0, np.inf, 0.0])
    display._update_events(data, x)  # must not raise
    assert _visible_events(display) == 0


def test_all_hidden_clears_the_events(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that hiding every channel takes the event overlays down with the traces.

    '_render' returns before it reaches the overlays while everything is hidden, thus
    hiding is the only chance to clear them: otherwise the last frame's lines stay
    frozen over the placeholder for as long as the layout stays empty.
    """
    push(50, stim_at=10)
    display._render()
    assert _visible_events(display) > 0
    display.set_channel_layout([])
    assert _visible_events(display) == 0
    display._render()  # returns early, thus nothing else could clear them
    assert _visible_events(display) == 0


def test_visible_stim_is_not_picked_twice(display: TraceDisplay) -> None:
    """Test that a visible stim channel is fetched once, not once more for its events.

    A duplicated pick appends a column the layout does not own, which breaks the
    'picks[:n_rows] == rows' prefix invariant the whole render loop rests on.
    """
    stim = display._event_acq[0]
    assert stim in display._rows
    assert display._picks == display._rows
    assert display._event_pos == [display._rows.index(stim)]


# -- layout changes --------------------------------------------------------------------
def test_reorder_restyles_the_assigned_curves(display: TraceDisplay) -> None:
    """Test that a reorder re-pens and re-transforms every row it kept.

    '_render' only styles a *newly* banded row, and a reorder bands nothing new, thus
    without the explicit restyle every curve keeps the previous channel's color and gain
    while drawing the new channel's samples.
    """
    display._render()
    display.set_channel_layout(list(reversed(range(display.n_channels))))
    mult = display.controls.state["scale"]
    assert display._assigned
    for row, curve in display._assigned.items():
        acq = display._rows[row]
        assert curve.opts["pen"].color().name() == display.color_for(row).name(), row
        assert curve.transform().m22() == pytest.approx(-display._gain[acq] * mult), row
    # the reversal puts the stim channel, whose gain differs from an eeg one, on row 0.
    assert display._rows[0] == display._event_acq[0]
    assert display._gain[display._rows[0]] != display._gain[display._rows[1]]


@pytest.mark.parametrize("rows", [[-1], [0, 1, 99], [8], [0, -2, 1]])
def test_set_channel_layout_rejects_invalid_rows(
    display: TraceDisplay, rows: list[int]
) -> None:
    """Test that a bad acquisition index raises before anything is mutated.

    An out-of-range index used to raise 'IndexError' half-way and leave the display
    unusable, and a negative one silently drew a different channel -- the one failure
    mode the index model exists to rule out.
    """
    before, picks = list(display._rows), list(display._picks)
    with pytest.raises(ValueError, match="acquisition indices"):
        display.set_channel_layout(rows)
    assert display._rows == before
    assert display._picks == picks


def test_band_reserves_the_event_label_headroom(display: TraceDisplay) -> None:
    """Test that the view range and the mask band both hold the label headroom.

    The headroom is where the event values are drawn; losing it draws them over the top
    trace, and a zero-height mask band lets the overscan rows peek into it.
    """
    display.controls.set_rows(4)
    display.scroll_to(2.0)
    band_top = 2.0 - 0.5 - _EVENT_LABEL_MARGIN
    assert display._vb.viewRange()[1] == pytest.approx([band_top, 2.0 + 4 - 0.5])
    rect = display._headband.rect()
    assert rect.top() == pytest.approx(band_top)
    assert rect.height() == pytest.approx(_EVENT_LABEL_MARGIN)
    # deliberately wider than any window: the rect is clipped to the view.
    assert rect.width() > display._winsize


def test_placeholder_stays_inside_the_view_range(display: TraceDisplay) -> None:
    """Test that the all-hidden placeholder is centred on the range the view has.

    '_render' owns the x range and returns early while everything is hidden, thus the
    placeholder is the one thing left to show and the one thing which lands off-screen
    if its position, or the range, is not refreshed here.
    """
    display.set_channel_layout([])
    assert display._empty_label.isVisible()
    (x0, x1), (y0, y1) = display._vb.viewRange()
    pos = display._empty_label.pos()
    assert (x0, x1) == pytest.approx((0.0, 5.0))
    assert pos.x() == pytest.approx(2.5)
    assert x0 <= pos.x() <= x1
    assert y0 <= pos.y() <= y1
    display.controls.set_window(20.0)  # widened with nothing to render
    (x0, x1), (y0, y1) = display._vb.viewRange()
    pos = display._empty_label.pos()
    assert (x0, x1) == pytest.approx((0.0, 20.0))
    assert pos.x() == pytest.approx(10.0)
    assert x0 <= pos.x() <= x1
    assert y0 <= pos.y() <= y1


# -- metadata robustness ---------------------------------------------------------------
def test_metadata_read_needs_a_connected_stream(display: TraceDisplay) -> None:
    """Test that a metadata refresh on a lost stream keeps the last known channels.

    'stream.info' raises on a disconnected stream, thus without the guard a refresh
    racing a disconnection raises instead of leaving the display on its last frame.
    """
    names, gains = list(display._names), list(display._gain)
    display._stream.disconnect()
    display.refresh_metadata()  # must not raise
    assert display._names == names
    assert display._gain == gains


def test_refresh_metadata_survives_a_shrinking_stream(
    display: TraceDisplay, push: Callable[..., None]
) -> None:
    """Test that a narrower channel set drops the rows which no longer exist.

    A structural change which shrank the stream leaves the layout holding indices past
    the end; this used to raise half-way and then made '_render' raise on every tick,
    logging the 'open an issue on GitHub' error 30 times a second.
    """
    push(50)
    display._render()
    display._stream.pick(display._stream.info.ch_names[:4])
    display.refresh_metadata()  # must not raise
    assert display.n_channels == 4
    assert display._rows == [0, 1, 2, 3]
    assert display._picks == [0, 1, 2, 3]
    assert len(display._gain) == 4
    display._render()  # must not raise either
    assert set(display._assigned) == {0, 1, 2, 3}


def test_render_survives_an_irregularly_sampled_stream(
    lsl_stream: Callable[..., tuple[StreamLSL, Callable]],
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that a stream declaring 'sfreq == 0' renders instead of raising every tick.

    'get_data' reads its window as a *sample count* for an irregular stream, thus the
    float window of the bar raised a 'TypeError' from inside the render tick.
    """
    stream, push = lsl_stream(sfreq=0.0, bufsize=50)
    display = make_display(stream)
    assert display._stream.info["sfreq"] == 0
    push(50)
    display._render()  # must not raise
    assert display._assigned
    assert display._vb.viewRange()[0] == pytest.approx([0.0, display._winsize])


def test_channel_type_tables_cover_the_channel_model() -> None:
    """Test that every type the Channels page can set has a range and a color.

    A type absent from the range tables falls back to a range of 1.0, i.e. the channel
    becomes unit-blind and a microvolt-declared trace draws dead flat. A type absent
    from the color table silently collapses onto the misc gray in type mode.
    """
    assert set(CH_TYPES) <= set(_RANGE_SI) | set(_RANGE_NATIVE)
    assert set(CH_TYPES) <= set(_TYPE_COLORS)


# -- lifecycle -------------------------------------------------------------------------
def test_show_grabs_the_keyboard_focus(shown_display: TraceDisplay) -> None:
    """Test that showing the display focuses it, so the arrow and page keys scroll.

    The plot widget explicitly refuses the focus, thus without this grab the scroll keys
    reach whatever the shell focused last and the display looks unresponsive.
    """
    assert shown_display.hasFocus()


def test_close_drops_the_theme_connection(
    app: QApplication,
    controller: ThemeController,
    stream: StreamLSL,
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that a closed display stops following the theme.

    The controller is a process singleton, thus a display which never disconnects is
    restyled -- and its control bar re-iconed -- for the rest of the process.
    """
    controller.install(app, "light")
    display = make_display(stream)
    assert display._mode == "light"
    display.close()
    controller.set_mode("dark")
    assert display._mode == "light"


def test_reopen_restores_the_theme_connection_and_the_clock(
    app: QApplication,
    controller: ThemeController,
    stream: StreamLSL,
    make_display: Callable[..., TraceDisplay],
) -> None:
    """Test that a reopened display follows the theme again and resumes rendering.

    The counterpart of the close, which drops both. Without it a display closed and
    reopened keeps the previous mode's baked pens and bar icons for the rest of the
    process, and -- worse -- sits frozen on the last frame it drew, as nothing restarts
    the render clock.
    """
    controller.install(app, "light")
    display = make_display(stream)
    display.start()
    display.close()
    assert not display.running
    controller.set_mode("dark")  # flipped while closed, so the pens are now stale
    assert display._mode == "light"
    display.show()
    app.processEvents()
    assert display.running  # the clock came back with the window
    assert display._mode == "dark"  # and the missed flip was caught up
    controller.set_mode("light")
    assert display._mode == "light"  # and it follows again
    display.close()


def test_reopening_a_stopped_display_leaves_the_clock_stopped(
    app: QApplication, stream: StreamLSL, make_display: Callable[..., TraceDisplay]
) -> None:
    """Test that showing a display which was never started does not start it.

    The clock is resumed only if the close stopped a running one; whoever owns the
    display decides when it renders.
    """
    display = make_display(stream)
    display.close()
    display.show()
    app.processEvents()
    assert not display.running
    display.close()


def test_no_controller_import(
    module_scan: Callable[[ModuleType], tuple[set[str], set[str]]],
) -> None:
    """Test that no module of 'display/' reaches the controller or the LSL layer.

    A source-level check, as importing any viewer module necessarily imports 'mne_lsl'
    and therefore 'mne_lsl.lsl'. The identifiers come from the syntax tree, so that a
    docstring mentioning a forbidden name is documentation and not a dependency. The
    scan is the shared 'module_scan' fixture, the one the 'backend/' rules also use.
    """
    found = {
        module.name for module in pkgutil.iter_modules(mne_lsl.viewer.display.__path__)
    }
    assert found == set(_MODULES)  # a new module must join the scan
    for name, module in _MODULES.items():
        imports, identifiers = module_scan(module)
        # every segment of an import path, so 'from ..controller import X' is caught by
        # its package as well as by the name it binds.
        segments = {segment for path in imports for segment in path.split(".")}
        assert (segments | identifiers).isdisjoint(_FORBIDDEN), name
