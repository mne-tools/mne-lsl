import time

import numpy as np
import pytest
from pylsl import StreamInfo as pylslStreamInfo
from pylsl import StreamInlet as pylslStreamInlet

from bsl.lsl import StreamInfo, StreamInlet, StreamOutlet


@pytest.mark.parametrize(
    "dtype_str_bsl, dtype_str_pylsl, dtype",
    [
        ("float32", "float32", np.float32),
        ("float64", "double64", np.float64),
        ("int8", "int8", np.int8),
        ("int16", "int16", np.int16),
        ("int32", "int32", np.int32),
    ],
)
def test_pull_numerical_sample(dtype_str_bsl, dtype_str_pylsl, dtype):
    """Test pull_sample against the pylsl version with numerical values."""
    x = [1, 2] if "int" in dtype_str_bsl else [1.0, 2.0]
    x_arr = np.array(x).astype(dtype)
    assert x_arr.shape == (2,) and x_arr.dtype == dtype

    # create stream descriptions
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, dtype_str_bsl, "")
    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0.0, dtype_str_pylsl, "")
    try:
        # test push/pull of single sample with pylsl
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = pylslStreamInlet(sinfo_pylsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert isinstance(data, list)
        assert data == x
        outlet.push_sample(x_arr)
        data, ts = inlet.pull_sample()
        assert isinstance(data, list)
        assert data == x
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del outlet
        del inlet

    try:
        # test push/pull of single sample with bsl.lsl
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = StreamInlet(sinfo_bsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert isinstance(data, np.ndarray)
        assert np.allclose(data, x_arr)
        outlet.push_sample(x_arr)
        data, ts = inlet.pull_sample()
        assert isinstance(data, np.ndarray)
        assert np.allclose(data, x_arr)
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet


def test_pull_str_sample():
    """Test pull_sample against the pylsl version with strings."""
    x = ["1", "2"]

    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0.0, "string", "")
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, "string", "")
    try:
        # test push/pull of single sample with pylsl
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = pylslStreamInlet(sinfo_pylsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert isinstance(data, list)
        assert data == x
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet

    try:
        # test push/pull of single sample with bsl.lsl
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = StreamInlet(sinfo_bsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert isinstance(data, list)
        assert data == x
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
    x = (
        [[1, 2, 3], [4, 5, 6]]
        if "int" in dtype_str
        else [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    x_arr = np.array(x).astype(dtype)
    assert x_arr.shape == (2, 3) and x_arr.dtype == dtype

    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, "")
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_chunk(x_arr)
        data, ts = inlet.pull_chunk(n_samples=3)
        assert isinstance(data, np.ndarray)
        assert np.allclose(x_arr, data)
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(n_samples=3)
        assert isinstance(data, np.ndarray)
        assert np.allclose(x_arr, data)
        assert inlet.samples_available == 0
        data, ts = inlet.pull_chunk(n_samples=1)
        assert data.size == ts.size == 0
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet


def test_pull_str_chunk():
    """Test pull_chunk on a string chunk."""
    x = [["1", "2", "3"], ["4", "5", "6"]]

    sinfo = StreamInfo("test", "", 2, 0.0, "string", "")
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        inlet = StreamInlet(sinfo)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_chunk(x)
        data, ts = inlet.pull_chunk(n_samples=3)
        assert isinstance(data, list)
        assert all(isinstance(elt, list) for elt in data)
        assert x == data
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet
