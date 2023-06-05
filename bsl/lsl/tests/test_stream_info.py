from time import strftime

import pytest

from bsl.lsl import StreamInfo


def test_stream_info_desc(caplog):
    """Test setters and getters for StreamInfo."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", strftime("%H%m%s"))
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
    assert sinfo.get_channel_types() == ch_types
    assert sinfo.get_channel_units() == ["101"] * 3
    ch_names = ["101", "201", "301"]
    sinfo.set_channel_names(ch_names)
    caplog.clear()
    assert sinfo.get_channel_names() == ch_names
    assert "elements for 3 channels" not in caplog.text

    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", strftime("%H%m%s"))
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
    assert "elements for 3 channels" not in caplog.text


def test_stream_info_invalid_desc():
    """Test invalid arguments for the channel description setters."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", strftime("%H%m%s"))
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
