import time
import uuid
from itertools import product

import numpy as np
import pytest

from bsl.lsl import StreamInfo, StreamInlet, StreamOutlet
from bsl.lsl.stream_info import _BaseStreamInfo


@pytest.mark.parametrize(
    "dtype_str, dtype",
    [
        ("float32", np.float32),
        ("float64", np.float64),
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
    ],
)
def test_pull_numerical_sample(dtype_str, dtype):
    """Test pull_sample with numerical values."""
    x = np.array([1, 2], dtype=dtype)
    assert x.shape == (2,) and x.dtype == dtype
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo, chunk_size=1)
        inlet = StreamInlet(sinfo)
        inlet.open_stream(timeout=5)
        _test_properties(inlet, dtype_str, 2, "test", 0.0, "")
        outlet.push_sample(x)
        data, ts = inlet.pull_sample(timeout=5)
        _test_numerical_data(data, x, dtype, ts)
        data, ts = inlet.pull_sample(timeout=0)
        assert ts is None
        assert data.size == 0
        # test push/pull with wrong dtype
        outlet.push_sample(x.astype(np.float64 if dtype != np.float64 else np.float32))
        data, ts = inlet.pull_sample(timeout=5)
        _test_numerical_data(data, x, dtype, ts)
    except Exception as error:
        raise error
    finally:
        try:
            del inlet
        except Exception:
            pass
        try:
            del outlet
        except Exception:
            pass


def test_pull_str_sample():
    """Test pull_sample with strings."""
    x = ["1", "2"]
    # create stream description
    sinfo = StreamInfo("test", "Gaze", 2, 10.0, "string", uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo, chunk_size=1)
        inlet = StreamInlet(sinfo)
        inlet.open_stream(timeout=5)
        _test_properties(inlet, "string", 2, "test", 10.0, "Gaze")
        outlet.push_sample(x)
        data, ts = inlet.pull_sample(timeout=5)
        assert isinstance(data, list)
        assert isinstance(ts, float)
        assert data == x
        data, ts = inlet.pull_sample(timeout=0)
        assert ts is None
        assert isinstance(data, list) and len(data) == 0
    except Exception as error:
        raise error
    finally:
        try:
            del inlet
        except Exception:
            pass
        try:
            del outlet
        except Exception:
            pass


@pytest.mark.parametrize(
    "dtype_str, dtype",
    [
        ("float32", np.float32),
        ("float64", np.float64),
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
    ],
)
def test_pull_numerical_chunk(dtype_str, dtype):
    """Test pull_chunk on a numerical chunk."""
    x = np.array([[1, 4], [2, 5], [3, 6]], dtype=dtype)
    assert x.shape == (3, 2) and x.dtype == dtype
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo)
        inlet.open_stream(timeout=5)
        _test_properties(inlet, dtype_str, 2, "test", 0.0, "")
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=3, timeout=5)
        _test_numerical_data(data, x, dtype, ts, 3)
        assert inlet.samples_available == 0
        data, ts = inlet.pull_chunk(max_samples=1, timeout=0)
        assert data.size == ts.size == 0
        # request more samples than available
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=5, timeout=1)
        _test_numerical_data(data, x, dtype, ts, 3)
        # pull sample by sample
        outlet.push_chunk(x)
        data, ts = inlet.pull_sample(timeout=5)
        _test_numerical_data(data, x[0, :], dtype, ts)
        data, ts = inlet.pull_sample(timeout=5)
        _test_numerical_data(data, x[1, :], dtype, ts)
        data, ts = inlet.pull_chunk(max_samples=5, timeout=1)
        _test_numerical_data(data, x[2, :], dtype, ts, 1)
        # test push/pull with wrong dtype
        outlet.push_chunk(x.astype(np.float64 if dtype != np.float64 else np.float32))
        data, ts = inlet.pull_chunk(max_samples=3, timeout=5)
        _test_numerical_data(data, x, dtype, ts, 3)
    except Exception as error:
        raise error
    finally:
        try:
            del inlet
        except Exception:
            pass
        try:
            del outlet
        except Exception:
            pass


def test_pull_str_chunk():
    """Test pull_chunk on a string chunk."""
    x = [["1", "4"], ["2", "5"], ["3", "6"]]
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, "string", uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo)
        inlet.open_stream(timeout=5)
        _test_properties(inlet, "string", 2, "test", 0.0, "")
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=3, timeout=5)
        assert isinstance(data, list)
        assert all(isinstance(elt, list) for elt in data)
        assert x == data
        data, ts = inlet.pull_chunk(max_samples=1, timeout=0)
        assert len(data) == len(ts) == 0
        # request more samples than available
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=5, timeout=1)
        assert isinstance(data, list)
        assert all(isinstance(elt, list) for elt in data)
        assert x == data
        # pull sample by sample
        outlet.push_chunk(x)
        data, ts = inlet.pull_sample(timeout=5)
        assert isinstance(ts, float)
        assert data == x[0]
        data, ts = inlet.pull_sample(timeout=5)
        assert isinstance(ts, float)
        assert data == x[1]
        data, ts = inlet.pull_chunk(max_samples=5, timeout=1)
        assert data == [x[2]]  # chunk is nested
    except Exception as error:
        raise error
    finally:
        try:
            del outlet
        except Exception:
            pass
        try:
            del inlet
        except Exception:
            pass


@pytest.mark.xfail(
    reason="liblsl bug with 20.04 and 22.04 LTS, "
    + "https://github.com/sccn/liblsl/issues/179",
    raises=TimeoutError,
    run=False,
)
def test_get_sinfo():
    """Test getting a StreamInfo from an Inlet."""
    sinfo = StreamInfo("test", "", 2, 0.0, "string", uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo)
        inlet = StreamInlet(sinfo)
        with pytest.raises(TimeoutError):
            inlet.get_sinfo(timeout=0.5)
        inlet.open_stream(timeout=5)
        sinfo = inlet.get_sinfo(timeout=5)
        assert isinstance(sinfo, _BaseStreamInfo)
    except Exception as error:
        raise error
    finally:
        try:
            del outlet
        except Exception:
            pass
        try:
            del inlet
        except Exception:
            pass


@pytest.mark.xfail(
    reason="liblsl bug, https://github.com/sccn/liblsl/issues/180",
    raises=AssertionError,
    run=False,
)
@pytest.mark.parametrize(
    "dtype_str, dtype",
    [
        ("float32", np.float32),
        ("float64", np.float64),
        ("int8", np.int8),
        ("int16", np.int16),
        ("int32", np.int32),
    ],
)
def test_inlet_methods(dtype_str, dtype):
    """Test the methods from an Inlet."""
    x = np.array([[1, 4], [2, 5], [3, 6]], dtype=dtype)
    assert x.shape == (3, 2) and x.dtype == dtype
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo)
        inlet.open_stream(timeout=5)
        outlet.push_chunk(x)
        time.sleep(0.1)  # sleep since samples_available does not have timeout
        assert inlet.samples_available == 3
        n_flush = inlet.flush()
        assert n_flush == 3
        assert inlet.samples_available == 0
        data, ts = inlet.pull_chunk(max_samples=1, timeout=0)
        assert data.size == ts.size == 0
        # close and re-open -- at the moment this is not well supported
        inlet.close_stream()
        inlet.open_stream(timeout=10)
        assert inlet.samples_available == 0
        outlet.push_chunk(x)
        assert inlet.samples_available == 3
    except Exception as error:
        raise error
    finally:
        try:
            del outlet
        except Exception:
            pass
        try:
            del inlet
        except Exception:
            pass


@pytest.mark.parametrize(
    "dtype_str, flags",
    product(
        ("float32", "int16"),
        (
            ["clocksync"],
            ["clocksync", "dejitter"],
            ["clocksync", "dejitter", "monotize"],
            ["clocksync", "dejitter", "monotize", "threadsafe"],
            "all",
        ),
    ),
)
def test_processing_flags(dtype_str, flags):
    """Test that the processing flags are working."""
    x = np.array([[1, 4], [2, 5], [3, 6]])
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, uuid.uuid4().hex[:6])
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo, processing_flags=flags)
        inlet.open_stream(timeout=5)
        _test_properties(inlet, dtype_str, 2, "test", 0.0, "")
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=3, timeout=5)
        assert inlet.samples_available == 0
        data, ts = inlet.pull_chunk(max_samples=1, timeout=0)
        assert data.size == ts.size == 0
    except Exception as error:
        raise error
    finally:
        try:
            del inlet
        except Exception:
            pass
        try:
            del outlet
        except Exception:
            pass


def _test_properties(inlet, dtype_str, n_channels, name, sfreq, stype):
    """Test the properties of an inlet against expected values."""
    assert inlet.dtype == dtype_str
    assert inlet.n_channels == n_channels
    assert inlet.name == name
    assert inlet.sfreq == sfreq
    assert inlet.stype == stype
    sinfo = inlet.get_sinfo(timeout=5)
    assert isinstance(sinfo, _BaseStreamInfo)


def _test_numerical_data(data, expected, dtype, ts, n_samples_expected=None):
    """Check that the pull data match the expected data."""
    assert isinstance(data, np.ndarray)
    assert data.dtype == dtype
    assert np.allclose(data, expected)
    assert data.ndim in (1, 2)
    if data.ndim == 1:  # pull_sample
        assert isinstance(ts, float)
        assert n_samples_expected is None
    elif data.ndim == 2:  # pull_chunk
        assert isinstance(ts, np.ndarray)
        assert ts.size == data.shape[0]
        assert ts.size == n_samples_expected
