from __future__ import annotations

import time

from mne_lsl.utils._time import high_precision_sleep


def test_high_precision_sleep() -> None:
    """Test high precision sleep function."""
    start = time.perf_counter()
    high_precision_sleep(0.4)
    end = time.perf_counter()
    assert 0.4 <= end - start
    # test value which should return right away
    high_precision_sleep(0)
    high_precision_sleep(-1)
