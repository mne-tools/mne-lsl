import re
import time
import uuid

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_lsl.lsl import StreamInfo, StreamInlet, StreamOutlet, local_clock
from mne_lsl.lsl.constants import string2numpy
from mne_lsl.lsl.stream_info import _BaseStreamInfo


@pytest.mark.parametrize(
    ("dtype_str", "dtype"),
    [
        ("float32", np.float32),
        ("float64", np.float64),
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
    ],
)
def test_push_numerical_sample(dtype_str, dtype, close_io):
    """Test push_sample with numerical values."""
    x = np.array([1, 2], dtype=dtype)
    assert x.shape == (2,)
    assert x.dtype == dtype
    # create stream descriptions
    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, uuid.uuid4().hex)
    outlet = StreamOutlet(sinfo, chunk_size=1)
    _test_properties(outlet, dtype_str, 2, "test", 0.0, "")
    inlet = StreamInlet(sinfo)
    inlet.open_stream(timeout=5)
    time.sleep(0.5)  # sleep required because of pylsl inlet
    outlet.push_sample(x)
    data, ts = inlet.pull_sample(timeout=5)
    assert_allclose(data, x)
    with pytest.raises(ValueError, match=re.escape("shape should be (n_channels,)")):
        outlet.push_sample(np.array([1, 2, 3, 4, 5, 6], dtype=dtype).reshape((2, 3)))
    with pytest.raises(ValueError, match="2 elements are expected"):
        outlet.push_sample(np.array([1, 2, 3, 4, 5], dtype=dtype))
    close_io()


def test_push_str_sample(close_io):
    """Test push_sample with strings."""
    x = ["1", "2"]
    # create stream descriptions
    sinfo = StreamInfo("test", "", 2, 0.0, "string", uuid.uuid4().hex)
    outlet = StreamOutlet(sinfo, chunk_size=1)
    _test_properties(outlet, "string", 2, "test", 0.0, "")
    inlet = StreamInlet(sinfo)
    inlet.open_stream(timeout=5)
    time.sleep(0.5)  # sleep required because of pylsl inlet
    outlet.push_sample(x)
    data, ts = inlet.pull_sample(timeout=5)
    assert data == x
    close_io()


@pytest.mark.parametrize(
    ("dtype_str", "dtype"),
    [
        ("float32", np.float32),
        ("float64", np.float64),
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
    ],
)
def test_push_numerical_chunk(dtype_str, dtype):
    """Test the error checking when pushing a numerical chunk."""
    x = np.array([[1, 4], [2, 5], [3, 6]], dtype=dtype)
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, uuid.uuid4().hex)
    outlet = StreamOutlet(sinfo, chunk_size=3)
    _test_properties(outlet, dtype_str, 2, "test", 0.0, "")
    # valid
    outlet.push_chunk(x)

    # invalid
    with pytest.raises(
        AssertionError,
        match="must be an array if numericals are pushed.",
    ):
        outlet.push_chunk(tuple(x))
    with pytest.raises(
        ValueError,
        match=re.escape("the shape should be (n_samples, n_channels)"),
    ):
        outlet.push_chunk(np.array(x, dtype=dtype).T)
    with pytest.raises(
        ValueError,
        match=re.escape("the shape should be (n_samples, n_channels)"),
    ):
        outlet.push_chunk(np.array(x, dtype=dtype).flatten())


def test_push_str_chunk():
    """Test the error checking when pushing a string chunk."""
    sinfo = StreamInfo("test", "", 2, 0.0, "string", uuid.uuid4().hex)
    outlet = StreamOutlet(sinfo, chunk_size=3)
    _test_properties(outlet, "string", 2, "test", 0.0, "")
    # valid
    outlet.push_chunk([["1", "4"], ["2", "5"], ["3", "6"]])

    # invalid
    with pytest.raises(
        AssertionError,
        match="must be a list if strings are pushed.",
    ):
        outlet.push_chunk((["1", "4"], ["2", "5"], ["3", "6"]))
    with pytest.raises(
        ValueError,
        match="must contain one element per channel at each time-point",
    ):
        outlet.push_chunk([["1", "4"], ["2", "5"], ["3", "6"], ["7"]])


def test_wait_for_consumers(close_io):
    """Test wait for client."""
    sinfo = StreamInfo("test", "EEG", 2, 100.0, "float32", uuid.uuid4().hex)
    outlet = StreamOutlet(sinfo, chunk_size=3)
    _test_properties(outlet, "float32", 2, "test", 100.0, "EEG")
    assert not outlet.wait_for_consumers(timeout=0.2)
    assert not outlet.has_consumers
    inlet = StreamInlet(sinfo)
    assert not outlet.wait_for_consumers(timeout=0.2)
    assert not outlet.has_consumers
    inlet.open_stream(timeout=5)
    assert outlet.wait_for_consumers(timeout=0.2)
    assert outlet.has_consumers
    close_io()


def test_invalid_outlet():
    """Test creation of an invalid outlet."""
    sinfo = StreamInfo("test", "EEG", 2, 100.0, "float32", uuid.uuid4().hex)
    with pytest.raises(
        ValueError, match="'chunk_size' must contain a positive integer"
    ):
        StreamOutlet(sinfo, chunk_size=-101)
    with pytest.raises(
        ValueError, match="'max_buffered' must contain a positive number"
    ):
        StreamOutlet(sinfo, max_buffered=-101)


@pytest.mark.parametrize(
    ("dtype_str", "dtype"),
    [
        ("float32", np.float32),
        ("float64", np.float64),
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
        ("string", None),
    ],
)
def test_push_chunk_timestamps(dtype_str, dtype, close_io):
    """Test push_chunk with timestamps."""
    if dtype_str == "string":
        x = [["1", "4"], ["2", "5"], ["3", "6"]]
    else:
        x = np.array([[1, 4], [2, 5], [3, 6]], dtype=dtype)
    # create stream description
    sinfo = StreamInfo("test", "", 2, 1.0, dtype_str, uuid.uuid4().hex)
    outlet = StreamOutlet(sinfo, chunk_size=3)
    _test_properties(outlet, dtype_str, 2, "test", 1.0, "")
    inlet = StreamInlet(sinfo)
    inlet.open_stream(timeout=5)
    time.sleep(0.5)  # sleep required because of pylsl inlet
    # float
    now = np.ceil(local_clock())
    outlet.push_chunk(x, timestamp=now)
    data, ts = inlet.pull_chunk(timeout=2)
    if dtype_str == "string":
        assert x == data
    else:
        assert_allclose(x, data)
    assert_allclose(ts, np.arange(now - len(x) + 1, now + 1))
    # array
    now = np.ceil(local_clock())
    timestamps = np.array([now, now + 1.1, now + 2.2])
    outlet.push_chunk(x, timestamp=timestamps)
    data, ts = inlet.pull_chunk(timeout=2)
    if dtype_str == "string":
        assert x == data
    else:
        assert_allclose(x, data)
    assert_allclose(ts, timestamps)
    # invalid
    with pytest.raises(
        TypeError,
        match="must be a float, an array or None",
    ):
        outlet.push_chunk(x, timestamp=[1, 2, 3])
    with pytest.raises(
        ValueError,
        match="must contain one element per sample",
    ):
        outlet.push_chunk(x, timestamp=np.arange(6).reshape(2, 3))
    with pytest.raises(
        ValueError,
        match="must contain one element per sample",
    ):
        outlet.push_chunk(x, timestamp=np.arange(4))
    close_io()


def test_push_chunk_irregularly_sampled_stream(close_io):
    """Test pushing a chunk on an irregularly sampled stream."""
    x = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, np.float32, uuid.uuid4().hex)
    outlet = StreamOutlet(sinfo, chunk_size=3)
    _test_properties(outlet, "float32", 2, "test", 0.0, "")
    inlet = StreamInlet(sinfo)
    inlet.open_stream(timeout=5)
    time.sleep(0.5)  # sleep required because of pylsl inlet
    # push with timestamp = None
    now = local_clock()
    outlet.push_chunk(x, timestamp=None)
    data, ts = inlet.pull_chunk(timeout=2)
    assert_allclose(x, data)
    assert_allclose(np.diff(ts), np.zeros(ts.size - 1))
    assert_allclose(ts, now, atol=1e-1)
    # push with timestamp = 0
    now = local_clock()
    outlet.push_chunk(x, timestamp=0)
    data, ts = inlet.pull_chunk(timeout=2)
    assert_allclose(x, data)
    assert_allclose(np.diff(ts), np.zeros(ts.size - 1))
    assert_allclose(ts, now, atol=1e-1)
    # push with timestamp = value
    now = local_clock()
    with pytest.warns(RuntimeWarning, match="will be applied to all samples"):
        outlet.push_chunk(x, timestamp=local_clock())
    data, ts = inlet.pull_chunk(timeout=2)
    assert_allclose(x, data)
    assert_allclose(np.diff(ts), np.zeros(ts.size - 1))
    assert_allclose(ts, now, atol=1e-1)
    # push with timestamp = np.array(...)
    timestamp = np.arange(x.shape[0]) + local_clock()
    outlet.push_chunk(x, timestamp=timestamp)
    data, ts = inlet.pull_chunk(timeout=2)
    assert_allclose(x, data)
    assert_allclose(ts, timestamp)
    # push with timestamp = np.zeros(...)
    with pytest.raises(RuntimeError, match="was supplied as an array of zeros"):
        outlet.push_chunk(x, timestamp=np.zeros(x.shape[0]))
    close_io()


def _test_properties(outlet, dtype_str, n_channels, name, sfreq, stype):
    """Test the properties of an outlet against expected values."""
    assert outlet.dtype == string2numpy.get(dtype_str, dtype_str)
    assert outlet.n_channels == n_channels
    assert outlet.name == name
    assert outlet.sfreq == sfreq
    assert outlet.stype == stype
    sinfo = outlet.get_sinfo()
    assert isinstance(sinfo, _BaseStreamInfo)
