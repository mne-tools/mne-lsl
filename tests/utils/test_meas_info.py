from __future__ import annotations

import uuid
from xml.etree import ElementTree

import numpy as np
import pytest
from mne import create_info as mne_create_info
from mne._fiff.constants import FIFF
from mne.channels import make_dig_montage

from mne_lsl.lsl import StreamInfo
from mne_lsl.utils.logs import _use_log_level
from mne_lsl.utils.meas_info import create_info


def test_valid_info() -> None:
    """Test creation of valid info."""
    ch_names = ["F7", "Fp2", "Trigger", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    ch_units = ["uv", "uv", "V", "uV"]
    # nested
    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units, strict=True):
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
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units, strict=True):
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
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units, strict=True):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=[ch_unit], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["sfreq"] == 1024.0
    assert len(info.ch_names) == 4
    assert info.ch_names == ch_names
    assert info.get_channel_types() == ch_types
    assert [ch["unit_mul"] for ch in info["chs"]] == [-6, -6, 0, -6]


def test_invalid_info() -> None:
    """Test creation of invalid info."""
    ch_names = ["F7", "Fp2", "Trigger", "EOG"]
    ch_types = ["wrong_type", "eeg", "stim", "eog"]
    ch_units = ["uv", "uv", "V", "uV"]

    desc = dict(channels=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units, strict=True):
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
    for ch_name, ch_type, ch_unit in zip(ch_names, ch_types, ch_units, strict=True):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=[ch_unit], type=[ch_type])
        )

    with pytest.warns(
        RuntimeWarning,
        match="Something went wrong while reading the channel description.",
    ):
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


def test_manufacturer() -> None:
    """Test creation of a valid info with a manufacturer entry."""
    ch_names = ["F7", "Fp2", "STI101", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    # nested
    desc = dict(channels=list(), manufacturer=list())
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in zip(ch_names, ch_types, strict=True):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )
    desc["manufacturer"].append("101")

    info = create_info(4, 1024, "eeg", desc)
    assert info["device_info"]["model"] == "101"

    # not nested
    desc = dict(channels=list(), manufacturer="101")
    desc["channels"].append(dict(channel=list()))
    for ch_name, ch_type in zip(ch_names, ch_types, strict=True):
        desc["channels"][0]["channel"].append(
            dict(label=[ch_name], unit=["uv"], type=[ch_type])
        )

    info = create_info(4, 1024, "eeg", desc)
    assert info["device_info"]["model"] == "101"


def test_manufacturer_from_sinfo() -> None:
    """Test reading manufacturer information from a StreamInfo."""
    ch_names = ["F7", "Fp2", "STI101", "EOG"]
    ch_types = ["eeg", "eeg", "stim", "eog"]
    sinfo = StreamInfo("pytest", "eeg", 4, 1024, "float32", uuid.uuid4().hex)
    sinfo.set_channel_names(ch_names)
    sinfo.set_channel_types(ch_types)
    sinfo.desc.append_child_value("manufacturer", "101")

    info = create_info(4, 1024, "eeg", sinfo)
    assert info["device_info"]["model"] == "101"


def test_valid_info_from_sinfo() -> None:
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


def test_without_description(caplog: pytest.LogCaptureFixture) -> None:
    """Test creation of a valid info without description."""
    with _use_log_level("INFO"):
        caplog.clear()
        info = create_info(2, 1024, "eeg", None)
        assert "No description provided. Creating a default Info object." in caplog.text
    assert info["sfreq"] == 1024
    assert len(info.ch_names) == 2
    assert info.ch_names == ["0", "1"]
    assert info.get_channel_types() == ["eeg", "eeg"]
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0]

    sinfo = StreamInfo("pytest", "eeg", 2, 101, "float32", uuid.uuid4().hex)
    with pytest.warns(
        RuntimeWarning,
        match="Something went wrong while reading the channel description.",
    ):
        info = create_info(2, 101, "eeg", sinfo)
    assert info["sfreq"] == 101
    assert len(info.ch_names) == 2
    assert info.ch_names == ["0", "1"]
    assert info.get_channel_types() == ["eeg", "eeg"]
    assert [ch["unit_mul"] for ch in info["chs"]] == [0, 0]


def _xml_to_dict(elem):
    """Convert an XML element to the nested dict representation used by pyxdf.

    Each element maps every child tag to the list of that tag's occurrences; a leaf is
    represented by its text content. This mirrors how ``pyxdf.load_xdf`` returns a
    stream description, so the dict path can be exercised from the same source as the
    StreamInfo path.
    """
    children = list(elem)
    if not children:
        return elem.text or ""
    out: dict = dict()
    for child in children:
        out.setdefault(child.tag, list()).append(_xml_to_dict(child))
    return out


def test_dig_and_locations_parity() -> None:
    """The dig, locations and filters are read identically from a sinfo and a dict.

    A description holding digitization points, channel locations and filters (as
    produced by ``neuromag2lsl``) must yield the same :class:`~mne.Info` whether it is
    read from a live :class:`~mne_lsl.lsl.StreamInfo` or from the dictionary returned
    when loading an XDF file.
    """
    ch_names = ["Fz", "Cz", "Pz", "Oz"]
    info_ref = mne_create_info(ch_names, 1000.0, "eeg")
    montage = make_dig_montage(
        nasion=[0.0, 0.1, 0.0],
        lpa=[-0.08, 0.0, 0.0],
        rpa=[0.08, 0.0, 0.0],
        hsp=[[0.01, 0.11, 0.02], [-0.01, 0.11, 0.02]],
        ch_pos={
            "Fz": [0.0, 0.07, 0.07],
            "Cz": [0.0, 0.0, 0.1],
            "Pz": [0.0, -0.07, 0.07],
            "Oz": [0.0, -0.1, 0.03],
        },
        coord_frame="head",
    )
    info_ref.set_montage(montage)
    with info_ref._unlock():
        info_ref["highpass"] = 0.5
        info_ref["lowpass"] = 120.0

    # push the reference info into a StreamInfo description, then read it back (sinfo
    # path) and from the equivalent dict (XDF path)
    sinfo = StreamInfo(
        "pytest", "eeg", len(ch_names), 1000.0, "float32", uuid.uuid4().hex
    )
    sinfo.set_channel_info(info_ref)
    info_sinfo = sinfo.get_channel_info()

    desc = ElementTree.fromstring(sinfo.as_xml).find("desc")
    info_dict = create_info(len(ch_names), 1000.0, "eeg", _xml_to_dict(desc))

    # both paths must recover the digitization, identical to the reference
    for info in (info_sinfo, info_dict):
        assert info["dig"] is not None
        assert len(info["dig"]) == len(info_ref["dig"])
        for point, ref in zip(info["dig"], info_ref["dig"], strict=True):
            assert int(point["kind"]) == int(ref["kind"])
            assert int(point["ident"]) == int(ref["ident"])
            assert point["coord_frame"] == FIFF.FIFFV_COORD_HEAD
            assert np.allclose(point["r"], ref["r"], atol=1e-6)

    # the two paths must agree with each other on dig, location, kind and filters
    assert len(info_sinfo["dig"]) == len(info_dict["dig"])
    for point_sinfo, point_dict in zip(
        info_sinfo["dig"], info_dict["dig"], strict=True
    ):
        assert int(point_sinfo["kind"]) == int(point_dict["kind"])
        assert int(point_sinfo["ident"]) == int(point_dict["ident"])
        assert np.allclose(point_sinfo["r"], point_dict["r"], atol=1e-6)
    for ch_sinfo, ch_dict, ch_ref in zip(
        info_sinfo["chs"], info_dict["chs"], info_ref["chs"], strict=True
    ):
        assert np.allclose(ch_sinfo["loc"], ch_dict["loc"], atol=1e-6, equal_nan=True)
        assert np.allclose(ch_sinfo["loc"], ch_ref["loc"], atol=1e-6, equal_nan=True)
        assert int(ch_sinfo["kind"]) == int(ch_dict["kind"]) == int(ch_ref["kind"])
    for key in ("highpass", "lowpass"):
        assert info_sinfo[key] == info_dict[key] == info_ref[key]
