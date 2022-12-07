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
    outlet = StreamOutlet(sinfo_bsl, chunk_size=1)

    # test push/pull of single sample with pylsl
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
    del inlet

    # test push/pull of single sample with bsl.lsl
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
    del inlet
    del outlet


def test_pull_str_sample():
    """Test pull_sample against the pylsl version with strings."""
    x = ["1", "2"]

    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0.0, "string", "")
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, "string", "")
    outlet = StreamOutlet(sinfo_bsl, chunk_size=1)

    # test push/pull of single sample with pylsl
    inlet = pylslStreamInlet(sinfo_pylsl)
    inlet.open_stream()
    time.sleep(0.1)
    outlet.push_sample(x)
    data, ts = inlet.pull_sample()
    assert isinstance(data, list)
    assert data == x
    inlet.close_stream()
    del inlet

    # test push/pull of single sample with bsl.lsl
    inlet = StreamInlet(sinfo_bsl)
    inlet.open_stream()
    time.sleep(0.1)
    outlet.push_sample(x)
    data, ts = inlet.pull_sample()
    assert isinstance(data, list)
    assert data == x
    inlet.close_stream()
    del inlet
    del outlet
