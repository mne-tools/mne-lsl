import re
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
def test_push_numerical_chunk(dtype_str, dtype):
    """Test the error checking when pushing a numerical chunk."""
    x = (
        [[1, 2, 3], [4, 5, 6]]
        if "int" in dtype_str
        else [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )

    sinfo = StreamInfo("test", "", 2, 0.0, dtype_str, "")
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        # valid
        outlet.push_chunk(x)

        # invalid
        with pytest.raises(
            AssertionError,
            match="must be a list of list or an array",
        ):
            outlet.push_chunk(tuple(x))
        with pytest.raises(
            ValueError,
            match="must contain one element per channel at each time-point",
        ):
            outlet.push_chunk(x + [[7]])

        # numpy valid
        outlet.push_chunk(np.array(x, dtype=dtype))

        # numpy invalid
        with pytest.raises(
            ValueError,
            match=re.escape("the shape should be (n_channels, n_samples)"),
        ):
            outlet.push_chunk(np.array(x, dtype=dtype).T)
        with pytest.raises(
            ValueError,
            match=re.escape("the shape should be (n_channels, n_samples)"),
        ):
            outlet.push_chunk(np.array(x, dtype=dtype).flatten())
    except Exception as error:
        raise error
    finally:
        del outlet


def test_push_str_chunk():
    """Test the error checking when pushing a string chunk."""
    sinfo = StreamInfo("test", "", 2, 0.0, "string", "")
    try:
        outlet = StreamOutlet(sinfo, chunk_size=3)
        # valid
        outlet.push_chunk([["1", "2", "3"], ["4", "5", "6"]])

        # invalid
        with pytest.raises(
            AssertionError,
            match="must be a list of list or an array",
        ):
            outlet.push_chunk((["1", "2", "3"], ["4", "5", "6"]))
        with pytest.raises(
            ValueError,
            match="must contain one element per channel at each time-point",
        ):
            outlet.push_chunk([["1", "2", "3"], ["4", "5", "6", "7"]])
    except Exception as error:
        raise error
    finally:
        del outlet
