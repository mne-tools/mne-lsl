from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import qInstallMessageHandler
from qtpy.QtWidgets import QWidget

from mne_lsl.viewer._bootstrap import import_ads
from mne_lsl.viewer.theme import _ADS_ICONS, _ADS_QSS, _QSS

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtWidgets import QApplication

_HEX = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
_ROLE = re.compile(r"palette\(([^)]*)\)")
# every role spelling the QSS 'palette()' function understands. Qt silently ignores an
# unknown role, painting black, thus nothing but this list catches a typo.
_ROLES = frozenset(
    {
        "accent",
        "alternate-base",
        "base",
        "bright-text",
        "button",
        "button-text",
        "dark",
        "highlight",
        "highlighted-text",
        "light",
        "link",
        "link-visited",
        "mid",
        "midlight",
        "placeholder-text",
        "shadow",
        "text",
        "tooltip-base",
        "tooltip-text",
        "window",
        "window-text",
    }
)
_SHEETS = {"_QSS": _QSS, "_ADS_QSS": _ADS_QSS}


@pytest.mark.parametrize("name", list(_SHEETS))
def test_sheets_have_no_color_literal(name: str) -> None:
    """Test that the style sheets are non-empty and free of hardcoded colors."""
    sheet = _SHEETS[name]
    assert sheet.strip()
    assert _HEX.search(sheet) is None, _HEX.findall(sheet)


@pytest.mark.parametrize("name", list(_SHEETS))
def test_sheets_palette_roles(name: str) -> None:
    """Test that every 'palette(<role>)' reference names an existing role."""
    references = _ROLE.findall(_SHEETS[name])
    assert references
    for reference in references:
        assert reference.strip() in _ROLES, reference


def test_ads_icons() -> None:
    """Test that the Qt-ADS icon slots exist and map to QtAwesome names."""
    ads = import_ads()
    assert set(_ADS_ICONS) == {
        "TabCloseIcon",
        "DockAreaCloseIcon",
        "DockAreaMenuIcon",
        "DockAreaUndockIcon",
    }
    for slot, name in _ADS_ICONS.items():
        assert hasattr(ads.eIcon, slot), slot
        assert name.startswith("mdi6."), name


def test_sheets_parse(app: QApplication, flush_deletes: Callable[..., None]) -> None:
    """Test that Qt parses both style sheets without complaining."""
    messages: list[str] = []

    def _handler(mode: object, context: object, message: str) -> None:
        messages.append(message)

    previous_sheet = app.styleSheet()
    previous_handler = qInstallMessageHandler(_handler)
    try:
        app.setStyleSheet(_QSS)
        widget = QWidget()
        widget.setStyleSheet(_ADS_QSS)
        widget.show()
        widget.ensurePolished()
        app.processEvents()
        widget.close()
        flush_deletes(widget)
    finally:
        qInstallMessageHandler(previous_handler)
        app.setStyleSheet(previous_sheet)
    faulty = [
        message
        for message in messages
        if "Unknown property" in message or "Could not parse" in message
    ]
    assert not faulty, faulty
