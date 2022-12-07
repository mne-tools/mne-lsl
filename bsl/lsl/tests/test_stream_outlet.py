import time

import numpy as np
import pytest
from pylsl import StreamInfo as pylslStreamInfo
from pylsl import StreamInlet as pylslStreamInlet
from pylsl import StreamOutlet as pylslStreamOutlet

from bsl.lsl import StreamInfo, StreamOutlet


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
def test_push_numerical_sample(dtype_str_bsl, dtype_str_pylsl, dtype):
    """Test push_sample against the pylsl version with numerical values."""
    x = [1, 2] if "int" in dtype_str_bsl else [1.0, 2.0]
    x_arr = np.array(x).astype(dtype)
    assert x_arr.shape == (2,) and x_arr.dtype == dtype

    # create stream descriptions
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, dtype_str_bsl, "")
    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0.0, dtype_str_pylsl, "")

    # test push/pull of single sample with bsl.lsl
    outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
    inlet = pylslStreamInlet(sinfo_pylsl)
    inlet.open_stream()
    time.sleep(0.1)
    outlet.push_sample(x)
    time.sleep(0.1)
    data, ts = inlet.pull_sample()
    assert data == x
    outlet.push_sample(x_arr)
    time.sleep(0.1)
    data, ts = inlet.pull_sample()
    assert data == x
    inlet.close_stream()
    del inlet
    del outlet

    # test push/pull of single sample with pylsl
    outlet = pylslStreamOutlet(sinfo_pylsl, chunk_size=1)
    inlet = pylslStreamInlet(sinfo_pylsl)
    inlet.open_stream()
    time.sleep(0.1)
    outlet.push_sample(x)
    time.sleep(0.1)
    data, ts = inlet.pull_sample()
    assert data == x
    outlet.push_sample(x_arr)
    time.sleep(0.1)
    data, ts = inlet.pull_sample()
    assert data == x
    inlet.close_stream()
    del inlet
    del outlet


def test_push_str_sample():
    """Test push_sample against the pylsl version with strings."""
    x = ["1", "2"]

    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0.0, "string", "")
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, "string", "")

    # test push/pull of single sample with bsl.lsl
    outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
    inlet = pylslStreamInlet(sinfo_pylsl)
    inlet.open_stream()
    time.sleep(0.1)
    outlet.push_sample(x)
    time.sleep(0.01)
    data, ts = inlet.pull_sample()
    inlet.close_stream()
    del inlet
    del outlet
    assert data == x

    # test push/pull of single sample with pylsl
    outlet = pylslStreamOutlet(sinfo_pylsl, chunk_size=1)
    inlet = pylslStreamInlet(sinfo_pylsl)
    inlet.open_stream()
    time.sleep(0.1)
    outlet.push_sample(x)
    time.sleep(0.1)
    data, ts = inlet.pull_sample()
    inlet.close_stream()
    del inlet
    del outlet
    assert data == x


def _test_push_numerical_chunk():
    """Test push_chunk against the pylsl version with numerical values."""
    # create stream descriptions
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, "float32", "")
    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0.0, "float32", "")

    # test (n_channels, n_samples)
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 3 samples
    x_arr = np.array(x).astype(np.float32)  # (n_channels, n_samples)
    assert x_arr.shape == (2, 3) and x_arr.dtype == np.float32

    # # test (n_samples, n_channels)
    # x = [[1., 4.], [2., 5.], [3., 6.]]  # 3 samples
    # x_arr = np.array(x).astype(np.float32)  # (n_samples, n_channels)
    # assert x_arr.shape == (3, 2) and x_arr.dtype == np.float32

    # test push/pull with pylsl
    outlet = StreamOutlet(sinfo_bsl, chunk_size=3)
    inlet = pylslStreamInlet(sinfo_pylsl)
    inlet.open_stream()
    time.sleep(0.1)
    outlet.push_chunk(x)
    time.sleep(0.1)
    data, ts = inlet.pull_chunk()
    inlet.close_stream()
    del inlet
    del outlet
