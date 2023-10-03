import numpy as np
import pytest
from mne.io import read_raw

from mne_lsl.datasets import testing
from mne_lsl.utils._tests import match_stream_and_raw_data

fname = testing.data_path() / "sample-eeg-ant-raw.fif"
raw = read_raw(fname, preload=True)


def test_match_stream_and_raw_data():
    """Make sure the match works as intended."""
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
