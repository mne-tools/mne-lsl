from __future__ import annotations  # c.f. PEP 563, PEP 649

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne.filter import create_filter
from scipy.signal import sosfilt_zi

from mne_lsl.stream._base import StreamFilter, _sanitize_filters

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture(scope="module")
def iir_params() -> dict[str, Any]:
    """Return a dictionary with valid IIR parameters."""
    return dict(order=4, ftype="butter", output="sos")


@pytest.fixture(scope="module")
def sfreq() -> int:
    """Return a valid sampling frequency."""
    return 1000


@pytest.fixture(scope="function")
def filters(iir_params, sfreq) -> list[dict[str, Any]]:
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
        for lfq, hfq in zip(l_freqs, h_freqs, strict=True)
    ]
    for k, (filt, lfq, hfq, picks_) in enumerate(
        zip(filters, l_freqs, h_freqs, picks, strict=True)
    ):
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


def test_StreamFilter(filters):
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


def test_sanitize_filters_no_overlap(filters):
    """Test clean-up of filter list to ensure non-overlap between channels."""
    filter_ = create_filter(
        data=None,
        sfreq=1000,
        l_freq=None,
        h_freq=100,
        method="iir",
        iir_params=dict(order=4, ftype="butter", output="sos"),
        phase="forward",
        verbose="CRITICAL",
    )
    filter_["zi"] = None
    filter_["zi_coeff"] = sosfilt_zi(filter_["sos"])
    filter_["picks"] = np.arange(30, 40)
    filter_["l_freq"] = None
    filter_["h_freq"] = 100
    filter_ = StreamFilter(filter_)
    all_picks = np.hstack([filt["picks"] for filt in filters + [filter_]])
    assert np.unique(all_picks).size == all_picks.size
    filters_clean = _sanitize_filters(filters, filter_)
    assert len(filters) == 3
    assert len(filters_clean) == 4
    assert filters == filters_clean[:3]
    assert filters_clean[-1] not in filters
    assert filters_clean[-1]["l_freq"] is None
    assert filters_clean[-1]["h_freq"] == 100
    assert np.array_equal(filters_clean[-1]["picks"], np.arange(30, 40))
    assert filters_clean[-1]["order"] == 4
    assert filters_clean[-1]["sos"].shape == (2, 6)


def test_sanitize_filters_partial_overlap(filters):
    """Test clean-up of filter list to ensure non-overlap between channels."""
    filter_ = create_filter(
        data=None,
        sfreq=1000,
        l_freq=None,
        h_freq=100,
        method="iir",
        iir_params=dict(order=4, ftype="butter", output="sos"),
        phase="forward",
        verbose="CRITICAL",
    )
    filter_["zi"] = None
    filter_["zi_coeff"] = sosfilt_zi(filter_["sos"])
    filter_["picks"] = np.arange(5, 15)
    filter_["l_freq"] = None
    filter_["h_freq"] = 100
    filter_ = StreamFilter(filter_)
    filters_clean = _sanitize_filters(filters, filter_)
    assert len(filters) == 3
    assert len(filters_clean) == 5
    # filter 0 and 1 are overlapping with filter_, thus we should have 2 new filters at
    # the end of the list, and only filter 2 should be preserved.
    assert filters[2] == filters_clean[2]
    assert filters[0] not in filters_clean
    assert filters[1] not in filters_clean
    # filter 0 and 1 should be lacking some channels
    for k, pick in enumerate((np.arange(0, 5), np.arange(15, 20))):
        assert np.array_equal(filters_clean[k]["picks"], pick)
        assert np.array_equal(filters_clean[k]["sos"], filters[k]["sos"])
        assert np.array_equal(filters_clean[k]["zi_coeff"], filters[k]["zi_coeff"])
        assert filters_clean[k]["zi"] is None
    # filter 3 should have the intersection with filter 0 and filter 4 with filter 1
    assert np.array_equal(filters_clean[3]["picks"], np.arange(5, 10))
    assert np.array_equal(
        filters_clean[3]["sos"], np.vstack((filters[0]["sos"], filter_["sos"]))
    )
    assert not np.array_equal(filters_clean[3]["zi_coeff"], filters[0]["zi_coeff"])
    assert not np.array_equal(filters_clean[3]["zi_coeff"], filter_["zi_coeff"])
    assert filters_clean[3]["zi"] is None
    assert np.array_equal(filters_clean[4]["picks"], np.arange(10, 15))
    assert np.array_equal(
        filters_clean[4]["sos"], np.vstack((filters[1]["sos"], filter_["sos"]))
    )
    assert not np.array_equal(filters_clean[4]["zi_coeff"], filters[1]["zi_coeff"])
    assert not np.array_equal(filters_clean[4]["zi_coeff"], filter_["zi_coeff"])
    assert filters_clean[4]["zi"] is None
    # check representation on combined filters
    assert filters_clean[3]["l_freq"] == (filters[0]["l_freq"], filter_["l_freq"])
    assert filters_clean[3]["h_freq"] == (filters[0]["h_freq"], filter_["h_freq"])
    assert f"({filters[0]['l_freq']}, {filter_['l_freq']})" in repr(filters_clean[3])
    assert f"({filters[0]['h_freq']}, {filter_['h_freq']})" in repr(filters_clean[3])
    assert filters_clean[4]["l_freq"] == (filters[1]["l_freq"], filter_["l_freq"])
    assert filters_clean[4]["h_freq"] == (filters[1]["h_freq"], filter_["h_freq"])
    assert f"({filters[1]['l_freq']}, {filter_['l_freq']})" in repr(filters_clean[4])
    assert f"({filters[1]['h_freq']}, {filter_['h_freq']})" in repr(filters_clean[4])


def test_sanitize_filters_full_overlap(filters):
    """Test clean-up of filter list to ensure non-overlap between channels."""
    filter_ = create_filter(
        data=None,
        sfreq=1000,
        l_freq=None,
        h_freq=100,
        method="iir",
        iir_params=dict(order=4, ftype="butter", output="sos"),
        phase="forward",
        verbose="CRITICAL",
    )
    filter_["zi"] = None
    filter_["zi_coeff"] = sosfilt_zi(filter_["sos"])
    filter_["picks"] = np.arange(0, 10)
    filter_["l_freq"] = None
    filter_["h_freq"] = 100
    filter_ = StreamFilter(filter_)
    filters_clean = _sanitize_filters(filters, filter_)
    assert len(filters) == 3
    assert len(filters_clean) == 3
    # filter 0 and filter_ fully overlap, thus filter 0 will be removed and the combined
    # filter is added to the end of the list -> order is not preserved.
    assert filters[1:] == filters_clean[:2]
    assert filters[0]["l_freq"] in filters_clean[-1]["l_freq"]
    assert filters[0]["h_freq"] in filters_clean[-1]["h_freq"]
    assert filter_["l_freq"] in filters_clean[-1]["l_freq"]
    assert filter_["h_freq"] in filters_clean[-1]["h_freq"]
    assert np.array_equal(filters_clean[-1]["picks"], np.arange(0, 10))
    assert filters_clean[-1]["zi"] is None
    assert not np.array_equal(filters_clean[-1]["zi_coeff"], filters[0]["zi_coeff"])
    assert not np.array_equal(filters_clean[-1]["zi_coeff"], filter_["zi_coeff"])
    assert np.array_equal(
        np.vstack((filters[0]["sos"], filter_["sos"])), filters_clean[-1]["sos"]
    )
