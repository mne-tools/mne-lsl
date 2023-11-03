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
    got_data = raw[:][0]
    # Sample numbers should be in the last row of the data, but could loop so are
    # not necessarily always increasing by 1
    good = np.isclose(got_data, data[:, :1], atol=1e-10, rtol=1e-8).all(axis=0)
    good = np.where(good)[0]
    if len(good) != 1:
        raise RuntimeError(
            "Could not find match between data and raw " f"(found {len(good)} options)."
        )
    start = good[0]
    del good
    stop = start + data.shape[1]
    n_fetch = 1
    if stop <= raw.times.size:
        raw_data = raw[:, start:stop][0]
    else:
        raw_data = raw[:, start:][0]
        while raw_data.shape[1] != data.shape[1]:
            if raw.times.size <= data.shape[1] - raw_data.shape[1]:
                raw_data = np.hstack((raw_data, raw[:, :][0]))
            else:
                raw_data = np.hstack(
                    (raw_data, raw[:, : data.shape[1] - raw_data.shape[1]][0])
                )
        n_fetch += 1
    err_msg = f"data mismatch after {n_fetch} fetch(es)"
    assert_allclose(data, raw_data, rtol=0.001, err_msg=err_msg)


def requires_module(name: str):  # pragma: no cover
    """Skip a test if package is not available (decorator)."""
    try:
        import_module(name)
        skip = False
    except ImportError:
        skip = True

    def decorator(function: Callable):
        return pytest.mark.skipif(
            skip, reason=f"Test {function.__name__} skipped, requires {name}."
        )(function)

    return decorator


def compare_infos(info1, info2):
    """Check that 2 infos are similar, even if some minor attribute deviate."""
    assert info1["ch_names"] == info2["ch_names"]
    assert info1["highpass"] == info2["highpass"]
    assert info2["lowpass"] == info2["lowpass"]
    assert info1["sfreq"] == info2["sfreq"]
    # projectors
    assert len(info1["projs"]) == len(info2["projs"])
    projs1 = sorted(info1["projs"], key=lambda x: x["desc"])
    projs2 = sorted(info2["projs"], key=lambda x: x["desc"])
    for proj1, proj2 in zip(projs1, projs2):
        assert proj1["desc"] == proj2["desc"]
        assert proj1["kind"] == proj2["kind"]
        assert proj1["data"]["nrow"] == proj2["data"]["nrow"]
        assert proj1["data"]["ncol"] == proj2["data"]["ncol"]
        assert proj1["data"]["row_names"] == proj2["data"]["row_names"]
        assert proj1["data"]["col_names"] == proj2["data"]["col_names"]
        assert_allclose(proj1["data"]["data"], proj2["data"]["data"])
    # digitization
    if info1["dig"] is not None and info2["dig"] is not None:
        assert len(info1["dig"]) == len(info2["dig"])
        digs1 = sorted(info1["dig"], key=lambda x: (x["kind"], x["ident"]))
        digs2 = sorted(info2["dig"], key=lambda x: (x["kind"], x["ident"]))
        for dig1, dig2 in zip(digs1, digs2):
            assert dig1["kind"] == dig2["kind"]
            assert dig1["ident"] == dig2["ident"]
            assert dig1["coord_frame"] == dig2["coord_frame"]
            assert_allclose(dig1["r"], dig2["r"])
    elif info1["dig"] is None and info2["dig"] is None:
        pass
    else:
        raise AssertionError
    # channel information, strict order
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
