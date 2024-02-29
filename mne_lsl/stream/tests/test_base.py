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


@pytest.fixture(scope="function")
def filters() -> list[dict[str, Any]]:
    """Create a list of valid filters."""
    l_freqs = (1, 1, 0.1)
    h_freqs = (40, 15, None)
    picks = (np.arange(0, 10), np.arange(10, 20), np.arange(20, 30))
    filters = [
        create_filter(
            data=None,
            sfreq=1000,
            l_freq=lfq,
            h_freq=hfq,
            method="iir",
            iir_params=dict(order=4, ftype="butter", output="sos"),
            phase="forward",
            verbose="ERROR",
        )
        for lfq, hfq in zip(l_freqs, h_freqs, strict=True)
    ]
    for filt, l_fq, h_fq, pick in zip(filters, l_freqs, h_freqs, picks, strict=True):
        filt["zi"] = None
        filt["zi_coeff"] = sosfilt_zi(filt["sos"])
        filt["picks"] = pick
        filt["l_freq"] = l_fq
        filt["h_freq"] = h_fq
    all_picks = np.hstack([filt["picks"] for filt in filters])
    assert np.unique(all_picks).size == all_picks.size  # sanity-check
    return [StreamFilter(filter_) for filter_ in filters]


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
        verbose="ERROR",
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


def test_StreamFilter(filters):
    """Test the StreamFilter class."""
    filter2 = deepcopy(filters[0])
    assert filter2 == filters[0]
    assert filters[0] != filters[1]
    assert filters[0] != filters[2]
    # test different key types
    filter2["order"] = str(filter2["order"])  # force different type
    with pytest.warns(RuntimeWarning, match="type of the key 'order' is different"):
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
