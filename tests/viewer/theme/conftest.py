from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from mne_lsl.viewer.theme import ThemeController


@pytest.fixture
def recorder(controller: ThemeController) -> Generator[list[str], None, None]:
    """Yield the list of modes emitted by 'theme_changed', disconnected afterwards."""
    received: list[str] = []

    def _record(mode: str) -> None:
        received.append(mode)

    controller.theme_changed.connect(_record)
    yield received
    controller.theme_changed.disconnect(_record)
