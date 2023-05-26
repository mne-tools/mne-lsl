"""Test meas_info.py"""

import pytest

from bsl.utils.meas_info import create_info


def test_valid_info():
    """Test creation of valid info."""
    channels = {
        "Fp1": "eeg",
        "Fp2": "eeg",
        "Trigger": "stim",
        "EOG": "eog",
    }
    # nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in channels.items():
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert len(info.ch_names) == 4
    assert sorted(info.ch_names) == sorted(channels)
    assert info.get_channel_types() == [channels[ch] for ch in info.ch_names]
    assert all(
        ch["unit_mul"] == (-6 if k in (0, 1) else 0) for k, ch in enumerate(info["chs"])
    )

    # non-nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in channels.items():
        desc["channels"][0]["channel"].append(
            dict(label=ch_name, unit="uv", type=ch_type)
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert len(info.ch_names) == 4
    assert sorted(info.ch_names) == sorted(channels)
    assert info.get_channel_types() == [channels[ch] for ch in info.ch_names]
    assert all(
        ch["unit_mul"] == (-6 if k in (0, 1) else 0) for k, ch in enumerate(info["chs"])
    )

    # marker stream
    info = create_info(4, 0, "eeg", desc)
    assert info["sfreq"] == 0.0
    assert len(info.ch_names) == 4
    assert sorted(info.ch_names) == sorted(channels)
    assert info.get_channel_types() == [channels[ch] for ch in info.ch_names]
    assert all(
        ch["unit_mul"] == (-6 if k in (0, 1) else 0) for k, ch in enumerate(info["chs"])
    )

    info = create_info(2, 1024, "eeg", desc)
    assert info["sfreq"] == 1024
    assert len(info.ch_names) == 2
    assert info.ch_names == ["0", "1"]
    assert info.get_channel_types() == ["eeg", "eeg"]
    assert all(ch["unit_mul"] == -0 for ch in info["chs"])


def test_invalid_info():
    """Test creation of invalid info."""
    # wrong type
    channels = {
        "Fp1": "wrong_type",
        "Fp2": "eeg",
        "Trigger": "stim",
        "EOG": "eog",
    }

    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in channels.items():
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert len(info.ch_names) == 4
    assert sorted(info.ch_names) == sorted(channels)
    assert info.get_channel_types() == [
        channels[ch] if k != 0 else "eeg" for k, ch in enumerate(info.ch_names)
    ]
    assert all(
        ch["unit_mul"] == (-6 if k in (0, 1) else 0) for k, ch in enumerate(info["chs"])
    )

    # wrong name
    channels = {
        101: "eeg",
        "Fp2": "eeg",
        "Trigger": "stim",
        "EOG": "eog",
    }
    # nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in channels.items():
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024
    assert len(info.ch_names) == 4
    assert info.ch_names == ["0", "1", "2", "3"]
    assert info.get_channel_types() == ["eeg", "eeg", "eeg", "eeg"]
    assert all(ch["unit_mul"] == -0 for ch in info["chs"])

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
    channels = {
        "Fp1": "eeg",
        "Fp2": "eeg",
        "Trigger": "stim",
        "EOG": "eog",
    }
    # nested
    desc = dict(channels=list(), manufacturer=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in channels.items():
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )
    desc["manufacturer"].append("101")

    info = create_info(4, 1024, "eeg", desc)
    assert info["device_info"]["model"] == "101"

    # not nested
    desc = dict(channels=list(), manufacturer="101")
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in channels.items():
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["device_info"]["model"] == "101"


def test_without_description():
    """Test creation of a valid info without description."""
    info = create_info(2, 1024, "eeg", None)
    assert info["sfreq"] == 1024
    assert len(info.ch_names) == 2
    assert info.ch_names == ["0", "1"]
    assert info.get_channel_types() == ["eeg", "eeg"]
    assert all(ch["unit_mul"] == -0 for ch in info["chs"])
