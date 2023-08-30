from __future__ import annotations  # c.f. PEP 563, PEP 649

from collections import Counter
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose


if TYPE_CHECKING:
    from typing import Callable

    from mne.io import BaseRaw
    from numpy.typing import NDArray


def match_stream_and_raw_data(
    data: NDArray[float], raw: BaseRaw, n_channels: int
) -> None:
    """Check if the data array is part of the provided raw."""
    idx = [np.where(raw[:, :][0] == data[k, 0])[1] for k in range(data.shape[0])]
    idx = np.concatenate(idx)
    counter = Counter(idx)
    idx, n_channels_ = counter.most_common()[0]
    assert n_channels_ == data.shape[0]
    assert n_channels_ == n_channels
    start = idx
    stop = start + data.shape[1]
    if stop <= raw.times.size:
        assert_allclose(data, raw[:, start:stop][0])
    else:
        raw_data = np.hstack((raw[:, start:][0], raw[:, :][0]))[:, : stop - start]
        assert_allclose(data, raw_data)


def requires_module(function: Callable, name: str):
    """Skip a test if package is not available (decorator)."""
    try:
        import_module(name)
        skip = False
    except ImportError:
        skip = True
    reason = f"Test {function.__name__} skipped, requires {name}."
    return pytest.mark.skipif(skip, reason=reason)(function)
