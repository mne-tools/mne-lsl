from __future__ import annotations  # c.f. PEP 563, PEP 649

from copy import deepcopy
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne.filter import create_filter
from scipy.signal import sosfilt_zi

from mne_lsl.stream._filters import (
    StreamFilter,
    _combine_filters,
    _sanitize_filters,
    _uncombine_filters,
)

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def iir_params() -> dict[str, Any]:
    """Return a dictionary with valid IIR parameters."""
    return dict(order=4, ftype="butter", output="sos")


@pytest.fixture(scope="module")
def sfreq() -> float:
    """Return a valid sampling frequency."""
    return 1000.0


@pytest.fixture(scope="module")
def picks() -> NDArray[np.int8]:
    """Return a valid selection of channels."""
    return np.arange(0, 10, dtype=np.int8)


@pytest.fixture(scope="function")
def filter1(
    iir_params: dict[str, Any], sfreq: float, picks: NDArray[np.int8]
) -> StreamFilter:
    """Create a filter."""
    l_freq = 1.0
    h_freq = 40.0
    filt = create_filter(
        data=None,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        phase="forward",
        verbose="CRITICAL",
    )
    filt.update(
        zi_coeff=sosfilt_zi(filt["sos"])[..., np.newaxis],
        zi=None,
        l_freq=l_freq,
        h_freq=h_freq,
        iir_params=iir_params,
        sfreq=sfreq,
        picks=picks,
    )
    del filt["order"]
    del filt["ftype"]
    return StreamFilter(filt)


@pytest.fixture(scope="function")
def filter2(
    iir_params: dict[str, Any], sfreq: float, picks: NDArray[np.int8]
) -> StreamFilter:
    """Create a filter."""
    l_freq = 2.0
    h_freq = None
    filt = create_filter(
        data=None,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        phase="forward",
        verbose="CRITICAL",
    )
    filt.update(
        zi_coeff=sosfilt_zi(filt["sos"])[..., np.newaxis],
        zi=None,
        l_freq=l_freq,
        h_freq=h_freq,
        iir_params=iir_params,
        sfreq=sfreq,
        picks=picks,
    )
    del filt["order"]
    del filt["ftype"]
    return StreamFilter(filt)


@pytest.fixture(scope="function")
def filter3(sfreq: float, picks: NDArray[np.int8]) -> StreamFilter:
    """Create a filter."""
    l_freq = None
    h_freq = 80.0
    iir_params = dict(order=2, ftype="bessel", output="sos")
    filt = create_filter(
        data=None,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        phase="forward",
        verbose="CRITICAL",
    )
    filt.update(
        zi_coeff=sosfilt_zi(filt["sos"])[..., np.newaxis],
        zi=None,
        l_freq=l_freq,
        h_freq=h_freq,
        iir_params=iir_params,
        sfreq=sfreq,
        picks=picks,
    )
    del filt["order"]
    del filt["ftype"]
    return StreamFilter(filt)


def test_combine_uncombine_filters(
    filter1: StreamFilter,
    filter2: StreamFilter,
    filter3: StreamFilter,
    picks: NDArray[np.int8],
):
    """Test (un)combination of filters."""
    # uncombine self
    filt = _uncombine_filters(filter1)
    assert filter1 == filt[0]

    # combine 2 filters
    filt = _combine_filters(filter1, filter2, picks)
    assert np.array_equal(filt["sos"], np.vstack((filter1["sos"], filter2["sos"])))
    assert filt["sos"].shape[-1] == 6
    assert filt["l_freq"] == (filter1["l_freq"], filter2["l_freq"])
    assert filt["h_freq"] == (filter1["h_freq"], filter2["h_freq"])
    assert not np.array_equal(filt["zi_coeff"], filter1["zi_coeff"])
    assert not np.array_equal(filt["zi_coeff"], filter2["zi_coeff"])
    assert filt["zi"] is None
    assert np.array_equal(filt["picks"], filter1["picks"])
    assert np.array_equal(filt["picks"], filter2["picks"])
    assert filt["sfreq"] == filter1["sfreq"] == filter2["sfreq"]
    assert filt["iir_params"] == (filter1["iir_params"], filter2["iir_params"])
    filt1, filt2 = _uncombine_filters(filt)
    assert filt1 == filter1
    assert filt2 == filter2

    # add initial conditions
    filter2["zi"] = filter2["zi_coeff"] * 5
    assert filt2 != filter2
    filt = _combine_filters(filter1, filter2, picks)
    assert np.array_equal(filt["sos"], np.vstack((filter1["sos"], filter2["sos"])))
    assert filt["sos"].shape[-1] == 6
    assert filt["l_freq"] == (filter1["l_freq"], filter2["l_freq"])
    assert filt["h_freq"] == (filter1["h_freq"], filter2["h_freq"])
    assert not np.array_equal(filt["zi_coeff"], filter1["zi_coeff"])
    assert not np.array_equal(filt["zi_coeff"], filter2["zi_coeff"])
    assert filt["zi"] is None
    assert np.array_equal(filt["picks"], filter1["picks"])
    assert np.array_equal(filt["picks"], filter2["picks"])
    assert filt["sfreq"] == filter1["sfreq"] == filter2["sfreq"]
    assert filt["iir_params"] == (filter1["iir_params"], filter2["iir_params"])
    filt1, filt2 = _uncombine_filters(filt)
    assert filt1 == filter1
    assert filt2 != filter2
    filter2["zi"] = None
    assert filt2 == filter2

    # test with different filter type
    filt = _combine_filters(filter1, filter3, picks)
    assert np.array_equal(filt["sos"], np.vstack((filter1["sos"], filter3["sos"])))
    assert filt["sos"].shape[-1] == 6
    assert filt["l_freq"] == (filter1["l_freq"], filter3["l_freq"])
    assert filt["h_freq"] == (filter1["h_freq"], filter3["h_freq"])
    assert not np.array_equal(filt["zi_coeff"], filter1["zi_coeff"])
    assert not np.array_equal(filt["zi_coeff"], filter3["zi_coeff"])
    assert filt["zi"] is None
    assert np.array_equal(filt["picks"], filter1["picks"])
    assert np.array_equal(filt["picks"], filter3["picks"])
    assert filt["sfreq"] == filter1["sfreq"] == filter3["sfreq"]
    assert filt["iir_params"] == (filter1["iir_params"], filter3["iir_params"])
    filt1, filt3 = _uncombine_filters(filt)
    assert filt1 == filter1
    assert filt3 == filter3

    # test combination of 3 filters
    filt_ = _combine_filters(filt, filter2, picks)
    assert np.array_equal(
        filt_["sos"], np.vstack((filter1["sos"], filter3["sos"], filter2["sos"]))
    )
    assert np.array_equal(filt_["sos"], np.vstack((filt["sos"], filter2["sos"])))
    assert filt_["sos"].shape[-1] == 6
    assert filt_["l_freq"] == (filter1["l_freq"], filter3["l_freq"], filter2["l_freq"])
    assert filt_["h_freq"] == (filter1["h_freq"], filter3["h_freq"], filter2["h_freq"])
    assert not np.array_equal(filt_["zi_coeff"], filt["zi_coeff"])
    assert not np.array_equal(filt_["zi_coeff"], filter1["zi_coeff"])
    assert not np.array_equal(filt_["zi_coeff"], filter2["zi_coeff"])
    assert not np.array_equal(filt_["zi_coeff"], filter3["zi_coeff"])
    assert filt_["zi"] is None
    assert np.array_equal(filt_["picks"], picks)
    assert filt_["sfreq"] == filter1["sfreq"] == filter2["sfreq"] == filter3["sfreq"]
    assert filt_["iir_params"] == (
        filter1["iir_params"],
        filter3["iir_params"],
        filter2["iir_params"],
    )
    filt1, filt3, filt2 = _uncombine_filters(filt_)
    assert filt1 == filter1
    assert filt2 == filter2  # zi already set to None
    assert filt3 == filter3


def test_invalid_uncombine_filters(filter1, filter2, picks):
    """Test error raising in uncombine filters."""
    filt = _combine_filters(filter1, filter2, picks)
    filt["l_freq"] = filt["l_freq"][0]
    with pytest.raises(RuntimeError, match="as both tuple and non-tuple"):
        _uncombine_filters(filt)
    filt["h_freq"] = filt["h_freq"][0]
    with pytest.raises(RuntimeError, match="as both tuple and non-tuple"):
        _uncombine_filters(filt)
    filt["iir_params"] = filt["iir_params"][0]
    filt2 = _uncombine_filters(filt)
    assert filt == filt2[0]


@pytest.fixture(scope="function")
def filters(iir_params: dict[str, Any], sfreq: float) -> list[StreamFilter]:
    """Create a list of valid filters."""
    l_freqs = (1, 1, 0.1)
    h_freqs = (40, 15, None)
    picks = (np.arange(0, 10), np.arange(10, 20), np.arange(20, 30))
    filters = [
        create_filter(
            data=None,
            sfreq=sfreq,
            l_freq=lfq,
            h_freq=hfq,
            method="iir",
            iir_params=iir_params,
            phase="forward",
            verbose="CRITICAL",  # disable logs
        )
        for lfq, hfq in zip(l_freqs, h_freqs)
    ]
    for k, (filt, lfq, hfq, picks_) in enumerate(zip(filters, l_freqs, h_freqs, picks)):
        zi_coeff = sosfilt_zi(filt["sos"])[..., np.newaxis]
        filt.update(
            zi_coeff=zi_coeff,
            zi=zi_coeff * k,
            l_freq=lfq,
            h_freq=hfq,
            iir_params=iir_params,
            sfreq=sfreq,
            picks=picks_,
        )
        del filt["order"]
        del filt["ftype"]
    all_picks = np.hstack([filt["picks"] for filt in filters])
    assert np.unique(all_picks).size == all_picks.size  # sanity-check
    return [StreamFilter(filt) for filt in filters]


def test_StreamFilter(filters: StreamFilter):
    """Test the StreamFilter class."""
    filter2 = deepcopy(filters[0])
    assert filter2 == filters[0]
    assert filters[0] != filters[1]
    assert filters[0] != filters[2]
    # test different key types
    filter2["l_freq"] = str(filter2["l_freq"])  # force different type
    with pytest.warns(RuntimeWarning, match="type of the key 'l_freq' is different"):
        assert filter2 != filters[0]
    # test with nans
    filter2 = deepcopy(filters[0])
    filter3 = deepcopy(filters[0])
    filter2["sos"][0, 0] = np.nan
    assert filter2 != filter3
    filter3["sos"][0, 0] = np.nan
    assert filter2 == filter3
    # test absent key
    filter2 = deepcopy(filters[0])
    del filter2["sos"]
    assert filter2 != filters[0]
    # test representation
    assert f"({filters[0]['l_freq']}, {filters[0]['h_freq']})" in repr(filters[0])


@pytest.fixture(
    scope="function",
    params=[
        (None, 100, np.arange(30, 40)),
        (10, 100, np.arange(30, 40)),
        (50, None, np.arange(30, 40)),
        (None, 100, np.arange(0, 10)),
        (10, 100, np.arange(0, 10)),
        (50, None, np.arange(0, 10)),
        (None, 100, np.arange(5, 15)),
        (10, 100, np.arange(5, 15)),
        (50, None, np.arange(5, 15)),
        (None, 100, np.arange(5, 10)),
        (10, 100, np.arange(5, 10)),
        (50, None, np.arange(5, 10)),
        (None, 100, np.arange(5, 25)),
        (10, 100, np.arange(5, 25)),
        (50, None, np.arange(5, 25)),
    ],
)
def filter_(request, iir_params: dict[str, Any], sfreq: float) -> StreamFilter:
    """Create a filter."""
    l_freq, h_freq, picks = request.param
    filt = create_filter(
        data=None,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        phase="forward",
        verbose="CRITICAL",
    )
    filt.update(
        zi_coeff=sosfilt_zi(filt["sos"])[..., np.newaxis],
        zi=None,
        l_freq=l_freq,
        h_freq=h_freq,
        iir_params=iir_params,
        sfreq=sfreq,
        picks=picks,
    )
    del filt["order"]
    del filt["ftype"]
    return StreamFilter(filt)


def test_sanitize_filters(filters: list[StreamFilter], filter_: StreamFilter):
    """Test clean-up of filter list to ensure non-overlap between channels."""
    # look for overlapping channels
    overlap = [np.intersect1d(filt["picks"], filter_["picks"]) for filt in filters]
    # sanitize and validate output
    filts = _sanitize_filters(filters, filter_)
    if all(ol.size == 0 for ol in overlap):
        assert filts == filters + [filter_]
    else:
        picks = list(chain(*(filt["picks"] for filt in filts)))
        assert np.unique(picks).size == len(picks)  # ensure no more overlap
        # find pairs of filters that have been combined
        idx = [k for k, ol in enumerate(overlap) if ol.size != 0]
        for k in idx:
            filt = _combine_filters(filters[k], filter_, overlap[k])
            assert filt in filts
            assert filters[k] not in filts
        assert filter_ not in filts
