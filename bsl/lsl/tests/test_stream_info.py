from time import strftime

import numpy as np
import pytest

from bsl import logger
from bsl.lsl import StreamInfo

logger.propagate = True


def test_stream_info_desc(caplog):
    """Test setters and getters for StreamInfo."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", strftime("%H%M%S"))
    assert sinfo.get_channel_names() is None
    assert sinfo.get_channel_types() is None
    assert sinfo.get_channel_units() is None

    ch_names = ["1", "2", "3"]
    sinfo.set_channel_names(ch_names)
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() is None
    assert sinfo.get_channel_units() is None

    sinfo.set_channel_types("eeg")
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() == ["eeg"] * 3
    assert sinfo.get_channel_units() is None

    ch_units = ["uV", "microvolt", "something"]
    sinfo.set_channel_units(ch_units)
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() == ["eeg"] * 3
    assert sinfo.get_channel_units() == ch_units

    ch_names = ["101", "201", "301"]
    sinfo.set_channel_names(ch_names)
    ch_types = ["eeg", "eog", "ecg"]
    sinfo.set_channel_types(ch_types)
    sinfo.set_channel_units("101")
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() == ch_types
    assert sinfo.get_channel_units() == ["101"] * 3

    # temper on purpose with the description XML tree
    channels = sinfo.desc.child("channels")
    ch = channels.append_child("channel")
    ch.append_child_value("label", "tempered-label")

    caplog.set_level(30)  # WARNING level
    caplog.clear()
    assert sinfo.get_channel_names() == ch_names + ["tempered-label"]
    assert "description contains 4 elements for 3 channels" in caplog.text
    assert sinfo.get_channel_types() == ch_types + [None]
    assert sinfo.get_channel_units() == ["101"] * 3 + [None]
    ch_names = ["101", "201", "301"]
    sinfo.set_channel_names(ch_names)
    caplog.clear()
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() == ch_types
    assert sinfo.get_channel_units() == ["101"] * 3
    assert "elements for 3 channels" not in caplog.text

    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", strftime("%H%M%S"))
    channels = sinfo.desc.append_child("channels")
    ch = channels.append_child("channel")
    ch.append_child_value("label", "tempered-label")
    caplog.clear()
    assert sinfo.get_channel_names() == ["tempered-label"]
    assert "description contains 1 elements for 3 channels"
    ch_names = ["101", "201", "301"]
    sinfo.set_channel_names(ch_names)
    caplog.clear()
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() is None
    assert sinfo.get_channel_units() is None
    assert "elements for 3 channels" not in caplog.text


def test_stream_info_invalid_desc():
    """Test invalid arguments for the channel description setters."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", strftime("%H%M%S"))
    assert sinfo.get_channel_names() is None
    assert sinfo.get_channel_types() is None
    assert sinfo.get_channel_units() is None

    with pytest.raises(TypeError, match="instance of list or tuple"):
        sinfo.set_channel_names(101)
    with pytest.raises(TypeError, match="an instance of str"):
        sinfo.set_channel_names([101, 101, 101])
    with pytest.raises(ValueError, match="number of provided channel"):
        sinfo.set_channel_names(["101"])

    with pytest.raises(TypeError, match="instance of list or tuple"):
        sinfo.set_channel_types(101)
    with pytest.raises(TypeError, match="an instance of str"):
        sinfo.set_channel_types([101, 101, 101])
    with pytest.raises(ValueError, match="number of provided channel"):
        sinfo.set_channel_types(["101"])

    with pytest.raises(TypeError, match="instance of list or tuple"):
        sinfo.set_channel_units(101)
    with pytest.raises(TypeError, match="an instance of str"):
        sinfo.set_channel_units([101, 101, 101])
    with pytest.raises(ValueError, match="number of provided channel"):
        sinfo.set_channel_units(["101"])


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
def test_create_stream_info_with_numpy_dtype(dtype, dtype_str):
    """Test creation of a StreamInfo with a numpy dtype instead of a string."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, dtype_str, strftime("%H%M%S"))
    assert sinfo.dtype == dtype
    del sinfo
    sinfo = StreamInfo("pytest", "eeg", 3, 101, dtype, strftime("%H%M%S"))
    assert sinfo.dtype == dtype
    del sinfo


def test_create_stream_info_with_invalid_numpy_dtype():
    """Test creation of a StreamInfo with an invalid numpy dtype."""
    with pytest.raises(ValueError, match="provided dtype could not be interpreted as"):
        StreamInfo("pytest", "eeg", 3, 101, np.uint8, strftime("%H%M%S"))
