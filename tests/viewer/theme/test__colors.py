from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from qtpy.QtGui import QColor

from mne_lsl.viewer.theme import (
    apply_theme,
    channel_color,
    contrast_ratio,
    icon,
    plot_colors,
    tokens,
    trace_color,
    type_color,
)
from mne_lsl.viewer.theme._colors import _TYPE_COLORS

if TYPE_CHECKING:
    from qtpy.QtWidgets import QApplication

_MODES = ("light", "dark")


def test_type_color() -> None:
    """Test the fixed channel-type colors and the misc fallback."""
    assert type_color("eeg") == QColor("#6098bf")
    for ch_type, value in _TYPE_COLORS.items():
        assert type_color(ch_type) == QColor(value), ch_type
        assert type_color(ch_type).isValid()
    for unknown in ("", "eyetrack", "dbs"):
        assert type_color(unknown) == QColor(_TYPE_COLORS["misc"]), unknown


def test_channel_color() -> None:
    """Test that channel colors are deterministic and spread in hue."""
    assert channel_color(3) == channel_color(3)
    hues = [channel_color(i).hueF() for i in range(16)]
    distances = [
        min(abs(a - b), 1 - abs(a - b))
        for i, a in enumerate(hues)
        for b in hues[i + 1 :]
    ]
    assert min(distances) > 0.02, min(distances)


def test_trace_color() -> None:
    """Test that the dark mode reuses 'channel_color' and the light one deepens it."""
    for i in range(8):
        assert trace_color(i, "dark") == channel_color(i)
        assert trace_color(i, "light") != trace_color(i, "dark")
        light = trace_color(i, "light")
        # HSV round-trips through 8 bits, hence the tolerance.
        assert light.saturationF() == pytest.approx(0.62, abs=0.01)
        assert light.valueF() == pytest.approx(0.60, abs=0.01)


@pytest.mark.parametrize("mode", _MODES)
def test_trace_color_legibility(mode: str) -> None:
    """Test that every trace color clears 3:1 on the mode's plot background.

    Deliberately tight, the measured minima being 3.08 (light) and 3.23 (dark): it pins
    the saturation / value pair so that brightening the traces cannot silently break
    their legibility.
    """
    background = QColor(tokens(mode).plot_bg)
    for i in range(64):
        ratio = contrast_ratio(trace_color(i, mode), background)
        assert ratio >= 3.0, (mode, i, ratio)


def test_plot_colors() -> None:
    """Test that the pyqtgraph colors mirror the tokens and differ per mode."""
    for mode in _MODES:
        colors = plot_colors(mode)
        assert set(colors) == {"background", "foreground", "grid"}
        t = tokens(mode)
        assert colors == {
            "background": t.plot_bg,
            "foreground": t.plot_fg,
            "grid": t.grid,
        }
    assert plot_colors("light") != plot_colors("dark")


def test_contrast_ratio() -> None:
    """Test the bounds and the symmetry of the WCAG contrast ratio."""
    white, black = QColor("#ffffff"), QColor("#000000")
    assert contrast_ratio(white, black) == pytest.approx(21.0)
    assert contrast_ratio(white, white) == 1.0
    assert contrast_ratio(white, black) == contrast_ratio(black, white)
    grey = QColor("#808080")
    assert contrast_ratio(grey, white) == contrast_ratio(white, grey)


def _paints_with(name: str, color: str) -> bool:
    """Return whether the rendered icon has an opaque pixel of that color."""
    image = icon(name).pixmap(16, 16).toImage()
    expected = QColor(color).rgb()
    return any(
        image.pixelColor(x, y).alpha() > 200
        and image.pixelColor(x, y).rgb() == expected
        for x in range(image.width())
        for y in range(image.height())
    )


def test_icon(app: QApplication) -> None:
    """Test that an icon is rendered and follows the theme's icon token.

    The icon is deliberately built *before* the first flip: 'qtawesome.icon' memoizes on
    the name and the explicit keyword arguments only, thus this is what catches
    'apply_theme' forgetting to drop that cache and handing back the previous colors.
    """
    assert not icon("mdi6.close").isNull()
    for mode in _MODES:
        apply_theme(app, mode)
        assert _paints_with("mdi6.close", tokens(mode).icon), mode


def test_icon_requires_application() -> None:
    """Test that 'icon' raises without an application, in a fresh interpreter.

    QtAwesome caches a font-less icon set when it is first used without an application,
    which poisons every later icon of the process. The guard can therefore not be tested
    in-process: the session-wide 'app' fixture may or may not have been built yet,
    depending on the order 'pytest-randomly' picks.
    """
    code = (
        "from mne_lsl.viewer.theme import icon\n"
        "try:\n"
        "    icon('mdi6.close')\n"
        "except RuntimeError as error:\n"
        "    assert 'requires a running QApplication' in str(error), str(error)\n"
        "else:\n"
        "    raise AssertionError('icon() did not raise without a QApplication')\n"
    )
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
