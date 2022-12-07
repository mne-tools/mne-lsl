import time

import numpy as np
import pytest
from pylsl import StreamInfo as pylslStreamInfo
from pylsl import StreamOutlet as pylslStreamOutlet
from pylsl import StreamInlet as pylslStreamInlet

from bsl.lsl import StreamInfo, StreamOutlet


@pytest.mark.parametrize("dtype_str_bsl, dtype_str_pylsl, dtype", [
    ("float32", "float32", np.float32),
    ("float64", "double64", np.float64),
])
def test_push_float_sample(dtype_str_bsl, dtype_str_pylsl, dtype):
    """Test push_sample against the pylsl version with floats."""
    x = [1., 2.]
    x_arr = np.array(x).astype(dtype)
    assert x_arr.shape == (2,) and x_arr.dtype == dtype

    sinfo_bsl = StreamInfo("test", "", 2, 0., dtype_str_bsl, "")
    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0., dtype_str_pylsl, "")

    # test push/pull of single sample with bsl.lsl
    outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
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
    assert data == x  # confirms the previous result


@pytest.mark.parametrize("dtype_str, dtype", [
    ("int8", np.int8),
    ("int16", np.int16),
    ("int32", np.int32),
])
def test_push_int_sample(dtype_str, dtype):
    """Test push_sample against the pylsl version with integers."""
    x = [1, 2]
    x_arr = np.array(x).astype(dtype)
    assert x_arr.shape == (2,) and x_arr.dtype == dtype

    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0., dtype_str, "")
    sinfo_bsl = StreamInfo("test", "", 2, 0., dtype_str, "")

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

    # test push/pull of single sample with bsl.lsl
    outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
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


def test_push_str_sample():
    """Test push_sample against the pylsl version with strings."""
    x = ["1", "2"]

    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0., "string", "")
    sinfo_bsl = StreamInfo("test", "", 2, 0., "string", "")

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
