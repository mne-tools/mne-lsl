"""Test meas_info.py"""

import uuid

import pytest

from mne_lsl import logger
from mne_lsl.lsl import StreamInfo
from mne_lsl.utils.meas_info import create_info

logger.propagate = True


def test_valid_info():
    """Test creation of valid info."""
    ch_names = ["F7", "Fp2", "Trigger", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    ch_units = ["uv", "uv", "V", "uV"]
    # nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=[ch_unit], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert info["lowpass"] == 512.0
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, -6]

    # non-nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units):
        desc["channels"][0]["channel"].append(
            dict(label=ch_name, unit=ch_unit, type=ch_type)
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, -6]

    # marker stream
    info = create_info(4, 0, "eeg", desc)
    assert info["sfreq"] == 0.0
    assert info["lowpass"] == 0.0
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, -6]

    with pytest.warns(
        RuntimeWarning,
        match="Something went wrong while reading the channel description.",
    ):
        info = create_info(2, 1024, "eeg", desc)
    assert info["sfreq"] == 1024
    assert len(info.ch_names) == 2
    assert info.ch_names == ["0", "1"]
    assert info.get_channel_types() == ["eeg", "eeg"]
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0]

    # units as integers
    ch_names = ["F7", "Fp2", "Trigger", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    ch_units = ["-6", "-6", "0", "uV"]
    # nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=[ch_unit], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, -6]


def test_invalid_info():
    """Test creation of invalid info."""
    ch_names = ["F7", "Fp2", "Trigger", "EOG"]
    ch_types = ["wrong_type", "eeg", "stim", "eog"]
    ch_units = ["uv", "uv", "V", "uV"]

    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=[ch_unit], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ["eeg"] + ch_types[1:]
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, -6]

    # wrong name
    ch_names = [101, "Fp2", "Trigger", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    ch_units = ["uv", "uv", "V", "uV"]

    # nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=[ch_unit], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024
    assert len(info.ch_names) == 4
    assert info.ch_names == ["0", "1", "2", "3"]
    assert info.get_channel_types() == ["eeg", "eeg", "eeg", "eeg"]
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0, 0, 0]

    # invalid that should raise
    with pytest.raises(TypeError, match="'n_channels' must be an integer"):
        create_info(5.0, 1024, "eeg", desc)
    with pytest.raises(TypeError, match="'sfreq' must be an instance of"):
        create_info(5, [101], "eeg", desc)
    with pytest.raises(ValueError, match="The sampling frequency"):
        create_info(5, -101, "eeg", desc)
    with pytest.raises(TypeError, match="'stype' must be an instance of str"):
        create_info(5, 101, 101, desc)
    with pytest.raises(TypeError, match="'desc' must be an instance of"):
        create_info(5, 101, "eeg", [101])


def test_manufacturer():
    """Test creation of a valid info with a manufacturer entry."""
    ch_names = ["F7", "Fp2", "STI101", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    # nested
    desc = dict(channels=list(), manufacturer=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in zip(ch_names, ch_types):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )
    desc["manufacturer"].append("101")

    info = create_info(4, 1024, "eeg", desc)
    assert info["device_info"]["model"] == "101"

    # not nested
    desc = dict(channels=list(), manufacturer="101")
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in zip(ch_names, ch_types):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["device_info"]["model"] == "101"


def test_valid_info_from_sinfo():
    """Test creation of a valid info from a SreamInfo."""
    sinfo = StreamInfo("pytest", "eeg", 4, 101, "float32", uuid.uuid4().hex)
    ch_names = ["F7", "Fp2", "STI101", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    sinfo.set_channel_names(ch_names)
    sinfo.set_channel_types(ch_types)
    info = create_info(4, 101, "eeg", sinfo)
    assert info["sfreq"] == 101
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0, 0, 0]

    # set channel units
    sinfo.set_channel_units(["uv", "uv", "none", "uv"])
    info = create_info(4, 101, "eeg", sinfo)
    assert info["sfreq"] == 101
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, -6]

    # stream info without a main channel type
    sinfo = StreamInfo("pytest", "", 4, 101, "float32", uuid.uuid4().hex)
    ch_names = ["F7", "Fp2", "STI101", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    sinfo.set_channel_names(ch_names)
    sinfo.set_channel_types(ch_types)
    info = create_info(4, 101, "eeg", sinfo)
    assert info["sfreq"] == 101
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0, 0, 0]

    # set channel units
    sinfo.set_channel_units([-6, "-6", "none", "0"])
    info = create_info(4, 101, "eeg", sinfo)
    assert info["sfreq"] == 101
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, 0]


def test_without_description():
    """Test creation of a valid info without description."""
    info = create_info(2, 1024, "eeg", None)
    assert info["sfreq"] == 1024
    assert len(info.ch_names) == 2
    assert info.ch_names == ["0", "1"]
    assert info.get_channel_types() == ["eeg", "eeg"]
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0]

    sinfo = StreamInfo("pytest", "eeg", 2, 101, "float32", uuid.uuid4().hex)
    info = create_info(2, 101, "eeg", sinfo)
    assert info["sfreq"] == 101
    assert len(info.ch_names) == 2
    assert info.ch_names == ["0", "1"]
    assert info.get_channel_types() == ["eeg", "eeg"]
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0]
