import time
from collections import Counter

import numpy as np
import pytest
from matplotlib import pyplot as plt
from mne import Info
from mne.channels import DigMontage
from mne.io import read_raw
from numpy.testing import assert_allclose

from bsl import Stream, logger
from bsl.datasets import testing

logger.propagate = True

fname = testing.data_path() / "sample-eeg-ant-raw.fif"
raw = read_raw(fname, preload=True)


def test_stream(mock_lsl_stream):
    """Test a valid Stream."""
    # test connect/disconnect
    stream = Stream(bufsize=2, name="BSL-Player-pytest")
    assert stream.info is None
    assert not stream.connected
    stream.connect()
    assert isinstance(stream.info, Info)
    assert stream.connected
    stream.disconnect()
    assert stream.info is None
    assert not stream.connected
    stream.connect()
    assert isinstance(stream.info, Info)
    assert stream.connected

    # test content
    assert stream.info["ch_names"] == raw.info["ch_names"]
    assert stream.get_channel_types() == raw.get_channel_types()
    assert stream.info["sfreq"] == raw.info["sfreq"]

    # check fs and that the returned data array is in raw a couple of times
    time.sleep(0.1)  # give a bit of time to the stream to acquire the first chunks
    for _ in range(3):
        data, ts = stream.get_data(winsize=0.1)
        assert ts.size == data.shape[1]
        fs = 1 / np.diff(ts)
        assert_allclose(fs, stream.info["sfreq"])
        idx = [np.where(raw[:, :][0] == data[k, 0])[1] for k in range(data.shape[0])]
        idx = np.concatenate(idx)
        counter = Counter(idx)
        idx, n_channels = counter.most_common()[0]
        assert n_channels == data.shape[0]
        assert n_channels == stream.sinfo.n_channels
        start = idx
        stop = start + data.shape[1]
        if stop <= raw.times.size:
            assert_allclose(data, raw[:, start:stop][0])
        else:
            raw_data = np.hstack((raw[:, start:][0], raw[:, :][0]))[:, : stop - start]
            assert_allclose(data, raw_data)
        time.sleep(0.3)

    # montage
    stream.set_montage("standard_1020")
    stream.plot_sensors()
    plt.close("all")
    montage = stream.get_montage()
    assert isinstance(montage, DigMontage)
    assert montage.ch_names == stream.ch_names[1:]  # first channel is

    # dtype
    assert stream.dtype == stream.sinfo.dtype

    # disconnect
    stream.disconnect()


def test_stream_invalid():
    """Test creation and connection to an invalid stream."""
    with pytest.raises(RuntimeError, match="do not uniquely identify an LSL stream"):
        stream = Stream(bufsize=2, name="101")
        stream.connect()
    with pytest.raises(RuntimeError, match="do not uniquely identify an LSL stream"):
        stream = Stream(bufsize=2, stype="EEG")
        stream.connect()
    with pytest.raises(RuntimeError, match="do not uniquely identify an LSL stream"):
        stream = Stream(bufsize=2, source_id="101")
        stream.connect()
    with pytest.raises(TypeError, match="must be an instance of numeric"):
        Stream(bufsize="101")
    with pytest.raises(ValueError, match="must be a strictly positive number"):
        Stream(bufsize=0)
    with pytest.raises(ValueError, match="must be a strictly positive number"):
        Stream(bufsize=-101)
    with pytest.raises(TypeError, match="must be an instance of str"):
        Stream(1, name=101)
    with pytest.raises(TypeError, match="must be an instance of str"):
        Stream(1, stype=101)
    with pytest.raises(TypeError, match="must be an instance of str"):
        Stream(1, source_id=101)


def test_stream_connection_no_args(mock_lsl_stream):
    """Test connection to the only available stream."""
    stream = Stream(bufsize=2)
    assert stream.info is None
    assert not stream.connected
    assert stream.name is None
    assert stream.stype is None
    assert stream.source_id is None
    stream.connect()
    assert isinstance(stream.info, Info)
    assert stream.connected
    assert stream.name == "BSL-Player-pytest"
    assert stream.stype == ""
    assert stream.source_id == "BSL"
    stream.disconnect()


def test_stream_double_connection(mock_lsl_stream, caplog):
    """Test connecting twice to a stream."""
    caplog.set_level(30)  # WARNING
    caplog.clear()
    stream = Stream(bufsize=2, name="BSL-Player-pytest")
    stream.connect()
    assert "stream is already connected" not in caplog.text
    caplog.clear()
    stream.connect()
    assert "stream is already connected" in caplog.text
    stream.disconnect()
