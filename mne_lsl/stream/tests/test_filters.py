from __future__ import annotations  # c.f. PEP 563, PEP 649

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne.filter import create_filter
from scipy.signal import sosfilt_zi

from mne_lsl.stream._filters import StreamFilter

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture(scope="module")
def iir_params() -> dict[str, Any]:
    """Return a dictionary with valid IIR parameters."""
    return dict(order=4, ftype="butter", output="sos")


@pytest.fixture(scope="module")
def sfreq() -> float:
    """Return a valid sampling frequency."""
    return 1000.0


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
