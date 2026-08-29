from __future__ import annotations

import re
from dataclasses import fields
from typing import TYPE_CHECKING

import darkdetect
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from mne_lsl.viewer.theme import Tokens, contrast_ratio, resolve_mode, tokens
from mne_lsl.viewer.theme._tokens import _DARK, _LIGHT

if TYPE_CHECKING:
    from qtpy.QtWidgets import QApplication

_HEX = re.compile(r"^#[0-9a-f]{6}$")
_MODES = ("light", "dark")


def test_tables_are_hex_colors() -> None:
    """Test that every token of both tables is a lowercase '#rrggbb' string."""
    names = [field.name for field in fields(Tokens)]
    assert len(names) == 21
    for table in (_LIGHT, _DARK):
        for name in names:
            value = getattr(table, name)
            assert _HEX.match(value), (name, value)
            assert QColor(value).isValid(), (name, value)


def test_tables_differ() -> None:
    """Test that the tables differ on the tokens which define the mode."""
    for name in ("window", "base", "text", "selection", "plot_bg", "bad"):
        assert getattr(_LIGHT, name) != getattr(_DARK, name), name


def test_tokens() -> None:
    """Test that 'tokens' returns the table of the resolved mode."""
    assert tokens("light") is _LIGHT
    assert tokens("dark") is _DARK
    assert tokens() in (_LIGHT, _DARK)


def test_resolve_mode() -> None:
    """Test that a concrete mode resolves to itself and 'auto' to either."""
    assert resolve_mode("light") == "light"
    assert resolve_mode("dark") == "dark"
    # never an exact value: the offscreen platform reports 'ColorScheme.Unknown', thus
    # 'auto' falls through to darkdetect and the result is machine-dependent.
    assert resolve_mode("auto") in _MODES


@pytest.mark.parametrize("mode", ["Dark", "", None, "system", 0])
def test_resolve_mode_invalid(mode: object) -> None:
    """Test that an unknown mode raises."""
    with pytest.raises(ValueError, match="Invalid value for the 'mode' parameter"):
        resolve_mode(mode)


def test_resolve_mode_darkdetect_fallback(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that 'auto' consults darkdetect when Qt does not know the OS scheme."""
    if app.styleHints().colorScheme() is not Qt.ColorScheme.Unknown:
        pytest.skip("Qt reports the OS color scheme, darkdetect is never consulted.")
    monkeypatch.setattr(darkdetect, "theme", lambda: "Dark")
    assert resolve_mode("auto") == "dark"
    monkeypatch.setattr(darkdetect, "theme", lambda: "Light")
    assert resolve_mode("auto") == "light"


def test_resolve_mode_darkdetect_failure(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a failing darkdetect backend is not fatal."""

    def _raise() -> str:
        raise RuntimeError("no darkdetect backend on this platform")

    # 'app' is requested, not used: the fallback must be exercised with an application
    # present, as in production.
    monkeypatch.setattr(darkdetect, "theme", _raise)
    assert resolve_mode("auto") in _MODES


@pytest.mark.parametrize("mode", _MODES)
def test_contrast_text(mode: str) -> None:
    """Test that the text tokens clear WCAG 4.5:1 on their backgrounds."""
    t = tokens(mode)
    for bg in ("window", "base", "surface", "raised"):
        ratio = contrast_ratio(QColor(t.text), QColor(getattr(t, bg)))
        assert ratio >= 4.5, (mode, bg, ratio)
    # 'text_secondary' is 'palette(mid)', used for hint text and dividers. It is not
    # asserted on 'raised': it reads 4.43:1 there in dark mode, and no hint text is
    # drawn on a button face.
    for bg in ("window", "base", "surface"):
        ratio = contrast_ratio(QColor(t.text_secondary), QColor(getattr(t, bg)))
        assert ratio >= 4.5, (mode, bg, ratio)


@pytest.mark.parametrize("mode", _MODES)
def test_contrast_status(mode: str) -> None:
    """Test that the status tokens clear WCAG 4.5:1 on window and base."""
    t = tokens(mode)
    # 'surface' is deliberately excluded: light 'success' reads 4.21:1 on it, thus
    # status text must be drawn on 'window' or 'base'.
    for name in ("error", "warning", "success"):
        for bg in ("window", "base"):
            ratio = contrast_ratio(QColor(getattr(t, name)), QColor(getattr(t, bg)))
            assert ratio >= 4.5, (mode, name, bg, ratio)


@pytest.mark.parametrize("mode", _MODES)
def test_contrast_selection_accent_bad(mode: str) -> None:
    """Test the selection text, the accent and the bad-channel token."""
    t = tokens(mode)
    assert contrast_ratio(QColor(t.selection_text), QColor(t.selection)) >= 4.5
    # 3:1 is the non-text threshold; the accent reads 4.17:1 in light mode.
    assert contrast_ratio(QColor(t.accent), QColor(t.window)) >= 3.0
    # the same color paints the bad trace and its axis label, thus it must clear the
    # text threshold on the plot canvas, and stay near-neutral so that "greyed out"
    # reads as a category difference and not as one more hue among the traces.
    assert contrast_ratio(QColor(t.bad), QColor(t.plot_bg)) >= 4.5
    assert QColor(t.bad).saturation() <= 60
