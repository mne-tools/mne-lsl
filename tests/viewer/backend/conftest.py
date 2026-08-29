from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from mne_lsl.stream import BaseStream


@pytest.fixture
def streams() -> Generator[list[BaseStream], None, None]:
    """Yield a list of streams to disconnect at teardown.

    Every test which obtains a connected stream appends it here instead of disconnecting
    it itself: a failing assertion would otherwise leave a live inlet and its
    acquisition thread behind.

    Nothing is suppressed around the disconnection: 'connected' reads a partially set
    state as not connected and 'disconnect' is idempotent, so a stream whose connection
    raised halfway through is torn down here without raising.
    """
    connected: list[BaseStream] = []
    yield connected
    for stream in reversed(connected):
        stream.disconnect()
    connected.clear()
