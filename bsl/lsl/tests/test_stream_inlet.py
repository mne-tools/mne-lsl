import numpy as np
import pytest

from bsl.lsl import StreamInfo, StreamInlet, StreamOutlet


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
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, dtype_str, "")
    try:
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = StreamInlet(sinfo_bsl)
        inlet.open_stream()
        outlet.push_sample(x)
        data, ts = inlet.pull_sample(timeout=1)
        assert isinstance(data, np.ndarray)
        assert isinstance(ts, float)
        assert np.allclose(data, x)
        data, ts = inlet.pull_sample(timeout=0)
        assert ts is None
        assert data is None
        # test push/pull with wrong dtype
        outlet.push_sample(
            x.astype(np.float64 if dtype != np.float64 else np.float32)
        )
        data, ts = inlet.pull_sample(timeout=1)
        assert isinstance(data, np.ndarray)
        assert isinstance(ts, float)
        assert np.allclose(data, x)
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet


def test_pull_str_sample():
    """Test pull_sample with strings."""
    x = ["1", "2"]
    # create stream description
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, "string", "")
    try:
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = StreamInlet(sinfo_bsl)
        inlet.open_stream()
        outlet.push_sample(x)
        data, ts = inlet.pull_sample(timeout=1)
        assert isinstance(data, list)
        assert isinstance(ts, float)
        assert data == x
        data, ts = inlet.pull_sample(timeout=0)
        assert ts is None
        assert data is None
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet


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
    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, "")
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo)
        inlet.open_stream()
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=3, timeout=1)
        assert isinstance(data, np.ndarray)
        assert np.allclose(x, data)
        assert ts.size == 3
        assert inlet.samples_available == 0
        data, ts = inlet.pull_chunk(max_samples=1, timeout=0)
        assert data.size == ts.size == 0
        # request more samples than available
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=5, timeout=1)
        assert isinstance(data, np.ndarray)
        assert np.allclose(x, data)
        assert ts.size == 3
        # pull sample by sample
        outlet.push_chunk(x)
        data, ts = inlet.pull_sample(timeout=1)
        assert isinstance(data, np.ndarray)
        assert isinstance(ts, float)
        assert np.allclose(x[0, :], data)
        data, ts = inlet.pull_sample(timeout=1)
        assert isinstance(data, np.ndarray)
        assert isinstance(ts, float)
        assert np.allclose(x[1, :], data)
        data, ts = inlet.pull_chunk(max_samples=5, timeout=1)
        assert isinstance(data, np.ndarray)
        assert np.allclose(x[2, :], data)
        # test push/pull with wrong dtype
        outlet.push_chunk(
            x.astype(np.float64 if dtype != np.float64 else np.float32)
        )
        data, ts = inlet.pull_chunk(max_samples=3, timeout=1)
        assert isinstance(data, np.ndarray)
        assert np.allclose(x, data)
        assert ts.size == 3
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet


def test_pull_str_chunk():
    """Test pull_chunk on a string chunk."""
    x = [["1", "4"], ["2", "5"], ["3", "6"]]
    # create stream description
    sinfo = StreamInfo("test", "", 2, 0.0, "string", "")
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo)
        inlet.open_stream()
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(max_samples=3, timeout=1)
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
        data, ts = inlet.pull_sample(timeout=1)
        assert isinstance(ts, float)
        assert data == x[0]
        data, ts = inlet.pull_sample(timeout=1)
        assert isinstance(ts, float)
        assert data == x[1]
        data, ts = inlet.pull_chunk(max_samples=5, timeout=1)
        assert data == [x[2]]  # chunk is nested
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet
