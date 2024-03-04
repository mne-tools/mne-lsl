from __future__ import annotations  # c.f. PEP 563, PEP 649

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mne_lsl.stream._filters import StreamFilter, create_filter

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


@pytest.fixture(scope="module")
def filters(iir_params: dict[str, Any], sfreq: float) -> list[StreamFilter]:
    """Create a list of valid filters."""
    l_freqs = (1, 1, 0.1)
    h_freqs = (40, 15, None)
    picks = (np.arange(0, 10), np.arange(10, 20), np.arange(20, 30))
    filters = list()
    for k, (lfq, hfq, picks_) in enumerate(zip(l_freqs, h_freqs, picks)):
        filt = create_filter(
            sfreq=sfreq,
            l_freq=lfq,
            h_freq=hfq,
            iir_params=iir_params,
        )
        filt.update(picks=picks_)
        filt["zi"] = k * filt["zi_unit"]
        del filt["order"]
        del filt["ftype"]
        filters.append(StreamFilter(filt))
    return filters


def test_StreamFilter(iir_params: dict[str, Any], sfreq: float):
    """Test StreamFilter creation."""
    # test deletion of duplicates
    filt = create_filter(
        sfreq=sfreq,
        l_freq=1,
        h_freq=101,
        iir_params=iir_params,
    )
    filt.update(picks=np.arange(5, 15))
    filt = StreamFilter(filt)
    assert "order" not in filt
    assert "order" in filt["iir_params"]
    assert "ftype" not in filt
    assert "ftype" in filt["iir_params"]
    # test creation from self
    filt2 = StreamFilter(filt)
    assert filt == filt2


def test_StreamFilter_comparison(filters: StreamFilter):
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


def test_StreamFilter_repr(filters):
    """Test the representation."""
    assert f"({filters[0]['l_freq']}, {filters[0]['h_freq']})" in repr(filters[0])
    assert str(filters[0]["iir_params"]["order"]) in repr(filters[0])
