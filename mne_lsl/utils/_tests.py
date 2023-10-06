from __future__ import annotations  # c.f. PEP 563, PEP 649

import hashlib
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne.utils import assert_object_equal
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from typing import Callable

    from mne.io import BaseRaw
    from numpy.typing import NDArray


def sha256sum(fname):
    """Efficiently hash a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(fname, "rb", buffering=0) as file:
        while n := file.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def match_stream_and_raw_data(data: NDArray[float], raw: BaseRaw) -> None:
    """Check if the data array is part of the provided raw."""
    for start in range(raw.times.size):
        if np.allclose(np.squeeze(raw[:, start][0]), data[:, 0], atol=0, rtol=1e-8):
            break
    else:
        raise RuntimeError("Could not find match between data and raw.")
    stop = start + data.shape[1]
    if stop <= raw.times.size:
        assert_allclose(data, raw[:, start:stop][0])
    else:
        raw_data = raw[:, start:][0]
        while raw_data.shape[1] != data.shape[1]:
            if raw.times.size <= data.shape[1] - raw_data.shape[1]:
                raw_data = np.hstack((raw_data, raw[:, :][0]))
            else:
                raw_data = np.hstack(
                    (raw_data, raw[:, : data.shape[1] - raw_data.shape[1]][0])
                )
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


def compare_infos(info1, info2):
    """Check that 2 infos are similar, even if some minor attribute deviate."""
    assert info1["ch_names"] == info2["ch_names"]
    assert info1["highpass"] == info2["highpass"]
    assert info2["lowpass"] == info2["lowpass"]
    assert info1["sfreq"] == info2["sfreq"]
    assert_object_equal(info1["projs"], info2["projs"])
    assert_object_equal(info1["dig"], info2["dig"])
    chs1 = [
        {
            key: value
            for key, value in elt.items()
            if key
            in (
                "kind",
                "coil_type",
                "loc",
                "unit",
                "unit_mul",
                "ch_name",
                "coord_frame",
            )
        }
        for elt in info1["chs"]
    ]
    chs2 = [
        {
            key: value
            for key, value in elt.items()
            if key
            in (
                "kind",
                "coil_type",
                "loc",
                "unit",
                "unit_mul",
                "ch_name",
                "coord_frame",
            )
        }
        for elt in info2["chs"]
    ]
    assert_object_equal(chs1, chs2)
    range_cal1 = [elt["range"] * elt["cal"] for elt in info1["chs"]]
    range_cal2 = [elt["range"] * elt["cal"] for elt in info2["chs"]]
    assert_allclose(range_cal1, range_cal2)
