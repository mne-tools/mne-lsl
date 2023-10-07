import numpy as np
import pytest
from mne.io import read_info, read_raw

from mne_lsl.datasets import testing
from mne_lsl.utils._tests import compare_infos, match_stream_and_raw_data

raw = read_raw(testing.data_path() / "sample-eeg-ant-raw.fif", preload=True)
info = read_info(testing.data_path() / "sample_audvis_raw.fif")


def test_match_stream_and_raw_data():
    """Test that the data match works as intended."""
    # test default working match
    data = raw.get_data()  # (n_channels, n_samples)
    match_stream_and_raw_data(data[:, 10:100], raw)
    match_stream_and_raw_data(data[:, 800:1900], raw)
    match_stream_and_raw_data(data, raw)
    match_stream_and_raw_data(data[:, 101].reshape(-1, 1), raw)

    # test wrapping around the end
    match_stream_and_raw_data(np.hstack((data[:, 1700:], data[:, :200])), raw)
    match_stream_and_raw_data(np.hstack((data[:, 1800:], data[:, :10])), raw)

    # test wrapping twice around the end
    match_stream_and_raw_data(np.hstack((data[:, 1800:], data[:, :], data[:, :9])), raw)

    # test edge cases
    match_stream_and_raw_data(data[:, :101], raw)
    match_stream_and_raw_data(data[:, 101:], raw)
    match_stream_and_raw_data(np.hstack((data[:, -1:], data[:, :1])), raw)

    # tolerance error
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        match_stream_and_raw_data(np.hstack((data[:, :200], data[:, 1700:])), raw)

    # shape error
    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        match_stream_and_raw_data(data[:10, 10:100], raw)


def test_compare_infos():
    """Test that the partial info comparison works as intended."""
    with pytest.raises(AssertionError):
        compare_infos(info, raw.info)

    info2 = info.copy()
    compare_infos(info, info2)
    with info2._unlock():
        info2["projs"] = []
    with pytest.raises(AssertionError):
        compare_infos(info, info2)

    info2 = info.copy()
    compare_infos(info, info2)
    with info2._unlock():
        info2["dig"] = None
    with pytest.raises(AssertionError):
        compare_infos(info, info2)

    for param, value in zip(
        ("kind", "coil_type", "loc", "unit", "unit_mul", "ch_name", "coord_frame"),
        (202, 1, np.ones(12), 107, -6, "101", 0),
    ):
        info2 = info.copy()
        compare_infos(info, info2)
        with info2._unlock():
            info2["chs"][0][param] = value
        with pytest.raises(AssertionError):
            compare_infos(info, info2)
