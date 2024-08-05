from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne.filter import create_filter as create_filter_mne
from numpy.testing import assert_allclose

from mne_lsl.stream._filters import StreamFilter, create_filter, ensure_sos_iir_params

if TYPE_CHECKING:
    from typing import Any, Optional


@pytest.fixture(scope="module")
def iir_params() -> dict[str, Any]:
    """Return a dictionary with valid IIR parameters."""
    return dict(order=4, ftype="butter", output="sos")


@pytest.fixture(scope="module")
def sfreq() -> float:
    """Return a valid sampling frequency."""
    return 1000.0


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
    # test invalid creation
    del filt2["iir_params"]
    with pytest.warns(RuntimeWarning, match=" 'iir_params' key is missing"):
        StreamFilter(filt2)
    filt2["iir_params"] = filt["iir_params"]
    filt2["order"] = 101
    with pytest.raises(RuntimeError, match="inconsistent"):
        StreamFilter(filt2)
    # test lack of keys in 'iir_params'
    filt = dict(filt)  # cast to parent class should work
    assert "order" not in filt
    filt["order"] = filt["iir_params"]["order"]
    del filt["iir_params"]["order"]
    filt = StreamFilter(filt)
    assert "order" not in filt
    assert "order" in filt["iir_params"]


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


def test_StreamFilter_comparison(filters: list[StreamFilter]):
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
    # test with and without zis
    filter2 = deepcopy(filters[0])
    filter2["zi"] = None
    assert filters[0]["zi"] is not None
    assert filter2 != filters[0]


def test_StreamFilter_repr(filters: list[StreamFilter]):
    """Test the representation."""
    assert f"({filters[0]['l_freq']}, {filters[0]['h_freq']})" in repr(filters[0])
    assert str(filters[0]["iir_params"]["order"]) in repr(filters[0])


@pytest.mark.parametrize(("l_freq", "h_freq"), [(1, 40), (None, 15), (0.1, None)])
def test_create_filter(
    iir_params: dict[str, Any],
    sfreq: float,
    l_freq: Optional[float],
    h_freq: Optional[float],
):
    """Test create_filter conformity with MNE."""
    filter1 = create_filter(
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        iir_params=iir_params,
    )
    filter2 = create_filter_mne(
        data=None,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        phase="forward",
        verbose="CRITICAL",
    )
    assert_allclose(filter1["sos"], filter2["sos"])


def test_ensure_sos_iir_params():
    """Test validation of IIR params."""
    assert isinstance(ensure_sos_iir_params(None), dict)
    with pytest.raises(TypeError, match="must be an instance of"):
        ensure_sos_iir_params("101")
    iir_params = dict(order=8, ftype="bessel", output="sos")
    iir_params2 = ensure_sos_iir_params(iir_params)
    assert iir_params == iir_params2
    iir_params3 = ensure_sos_iir_params(dict(order=8, ftype="bessel"))
    assert iir_params == iir_params3
    with pytest.warns(RuntimeWarning, match="Only 'sos' output is supported"):
        iir_params4 = ensure_sos_iir_params(dict(order=8, ftype="bessel", output="ba"))
    assert iir_params == iir_params4
    with pytest.warns(RuntimeWarning, match="Only 'sos' output is supported"):
        iir_params5 = ensure_sos_iir_params(
            dict(order=8, ftype="bessel", a=[1, 2], b=[1, 2])
        )
    assert iir_params == iir_params5
