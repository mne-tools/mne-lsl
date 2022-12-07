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
    try:
        # test push/pull of single sample with bsl.lsl
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = pylslStreamInlet(sinfo_pylsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert data == x
        outlet.push_sample(x_arr)
        data, ts = inlet.pull_sample()
        assert data == x
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet

    try:
        # test push/pull of single sample with pylsl
        outlet = pylslStreamOutlet(sinfo_pylsl, chunk_size=1)
        inlet = pylslStreamInlet(sinfo_pylsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert data == x
        outlet.push_sample(x_arr)
        data, ts = inlet.pull_sample()
        assert data == x
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet


def test_push_str_sample():
    """Test push_sample against the pylsl version with strings."""
    x = ["1", "2"]

    sinfo_pylsl = pylslStreamInfo("test", "", 2, 0.0, "string", "")
    sinfo_bsl = StreamInfo("test", "", 2, 0.0, "string", "")

    try:
        # test push/pull of single sample with bsl.lsl
        outlet = StreamOutlet(sinfo_bsl, chunk_size=1)
        inlet = pylslStreamInlet(sinfo_pylsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert data == x
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet

    try:
        # test push/pull of single sample with pylsl
        outlet = pylslStreamOutlet(sinfo_pylsl, chunk_size=1)
        inlet = pylslStreamInlet(sinfo_pylsl)
        inlet.open_stream()
        time.sleep(0.1)
        outlet.push_sample(x)
        data, ts = inlet.pull_sample()
        assert data == x
        inlet.close_stream()
    except Exception as error:
        raise error
    finally:
        del inlet
        del outlet
