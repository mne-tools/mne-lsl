import uuid

import numpy as np
import pytest
from mne import create_info
from mne.io import read_raw_fif

from mne_lsl import logger
from mne_lsl.datasets import testing
from mne_lsl.lsl import StreamInfo, StreamInlet, StreamOutlet
from mne_lsl.utils._tests import compare_infos

logger.propagate = True


def test_stream_info_desc():
    """Test setters and getters for StreamInfo."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", uuid.uuid4().hex)
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

    ch_units = np.array([-6, -6, -6], dtype=np.int8)
    sinfo.set_channel_units(ch_units)
    assert sinfo.get_channel_units() == ["-6", "-6", "-6"]

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

    with pytest.warns(
        RuntimeWarning, match="description contains 4 elements for 3 channels"
    ):
        assert sinfo.get_channel_names() == ch_names + ["tempered-label"]
    with pytest.warns(
        RuntimeWarning, match="description contains 4 elements for 3 channels"
    ):
        assert sinfo.get_channel_types() == ch_types + [None]
    with pytest.warns(
        RuntimeWarning, match="description contains 4 elements for 3 channels"
    ):
        assert sinfo.get_channel_units() == ["101"] * 3 + [None]
    ch_names = ["101", "201", "301"]
    sinfo.set_channel_names(ch_names)
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() == ch_types
    assert sinfo.get_channel_units() == ["101"] * 3

    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", uuid.uuid4().hex)
    channels = sinfo.desc.append_child("channels")
    ch = channels.append_child("channel")
    ch.append_child_value("label", "tempered-label")
    with pytest.warns(
        RuntimeWarning, match="description contains 1 elements for 3 channels"
    ):
        assert sinfo.get_channel_names() == ["tempered-label"]
    ch_names = ["101", "201", "301"]
    sinfo.set_channel_names(ch_names)
    assert sinfo.get_channel_names() == ch_names
    assert sinfo.get_channel_types() is None
    assert sinfo.get_channel_units() is None


def test_stream_info_invalid_desc():
    """Test invalid arguments for the channel description setters."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", uuid.uuid4().hex)
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

    with pytest.raises(TypeError, match="list, tuple, ndarray, str, or int-like"):
        sinfo.set_channel_units(101.2)
    with pytest.raises(TypeError, match="an instance of str"):
        sinfo.set_channel_units([5.2, 5.2, 5.2])
    with pytest.raises(ValueError, match="number of provided channel"):
        sinfo.set_channel_units(["101"])
    with pytest.raises(ValueError, match="as a 1D array of integers"):
        sinfo.set_channel_units(np.array([-6, -6, -6, -6]).reshape(2, 2))


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
    sinfo = StreamInfo("pytest", "eeg", 3, 101, dtype_str, uuid.uuid4().hex)
    assert sinfo.dtype == dtype
    del sinfo
    sinfo = StreamInfo("pytest", "eeg", 3, 101, dtype, uuid.uuid4().hex)
    assert sinfo.dtype == dtype
    del sinfo


def test_create_stream_info_with_invalid_numpy_dtype():
    """Test creation of a StreamInfo with an invalid numpy dtype."""
    with pytest.raises(ValueError, match="provided dtype could not be interpreted as"):
        StreamInfo("pytest", "eeg", 3, 101, np.uint8, uuid.uuid4().hex)


def test_stream_info_equality():
    """Test == method."""
    sinfo1 = StreamInfo("pytest", "eeg", 3, 101, "float32", uuid.uuid4().hex)
    assert sinfo1 != 101
    sinfo2 = StreamInfo("pytest-2", "eeg", 3, 101, "float32", uuid.uuid4().hex)
    assert sinfo1 != sinfo2
    sinfo2 = StreamInfo("pytest", "gaze", 3, 101, "float32", uuid.uuid4().hex)
    assert sinfo1 != sinfo2
    sinfo2 = StreamInfo("pytest", "eeg", 3, 10101, "float32", uuid.uuid4().hex)
    assert sinfo1 != sinfo2
    sinfo2 = StreamInfo("pytest", "eeg", 101, 101, "float32", uuid.uuid4().hex)
    assert sinfo1 != sinfo2
    sinfo2 = StreamInfo("pytest", "eeg", 3, 101, np.float64, uuid.uuid4().hex)
    assert sinfo1 != sinfo2
    sinfo2 = StreamInfo("pytest", "eeg", 3, 101, np.float32, "pytest")
    assert sinfo1 != sinfo2
    sinfo2 = StreamInfo("pytest", "eeg", 3, 101, np.float32, sinfo1.source_id)
    assert sinfo1 == sinfo2


def test_stream_info_representation():
    """Test the str() representation of an Info."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", uuid.uuid4().hex)
    repr_ = str(sinfo)
    assert "'pytest'" in repr_
    assert "eeg" in repr_
    assert "101" in repr_
    assert "3" in repr_
    assert "float32" in repr_


def test_stream_info_properties(close_io):
    """Test properties."""
    sinfo = StreamInfo("pytest", "eeg", 3, 101, "float32", uuid.uuid4().hex)
    assert isinstance(sinfo.created_at, float)
    assert sinfo.created_at == 0.0
    assert isinstance(sinfo.hostname, str)
    assert len(sinfo.hostname) == 0
    assert isinstance(sinfo.session_id, str)
    assert len(sinfo.session_id) == 0
    assert isinstance(sinfo.uid, str)
    assert len(sinfo.uid) == 0
    assert isinstance(sinfo.protocol_version, int)
    assert isinstance(sinfo.as_xml, str)

    outlet = StreamOutlet(sinfo)
    sinfo_ = outlet.get_sinfo()
    assert isinstance(sinfo_.created_at, float)
    assert 0 < sinfo_.created_at
    assert isinstance(sinfo_.hostname, str)
    assert len(sinfo_.hostname) != 0
    assert isinstance(sinfo_.session_id, str)
    assert len(sinfo_.session_id) != 0
    assert isinstance(sinfo_.uid, str)
    assert len(sinfo_.uid) != 0

    inlet = StreamInlet(sinfo)
    with pytest.raises(RuntimeError, match="StreamInlet is not open"):
        inlet.get_sinfo()
    inlet.open_stream()
    sinfo_ = inlet.get_sinfo()
    assert isinstance(sinfo_.created_at, float)
    assert 0 < sinfo_.created_at
    assert isinstance(sinfo_.hostname, str)
    assert len(sinfo_.hostname) != 0
    assert isinstance(sinfo_.session_id, str)
    assert len(sinfo_.session_id) != 0
    assert isinstance(sinfo_.uid, str)
    assert len(sinfo_.uid) != 0

    close_io()


def test_invalid_stream_info():
    """Test creation of an invalid StreamInfo."""
    with pytest.raises(ValueError, match="'n_channels' must be a strictly positive"):
        StreamInfo("pytest", "eeg", -101, 101, "float32", uuid.uuid4().hex)
    with pytest.raises(ValueError, match="'sfreq' must be a positive"):
        StreamInfo("pytest", "eeg", 101, -101, "float32", uuid.uuid4().hex)


def test_stream_info_desc_from_info(close_io):
    """Test filling a description from an Info object."""
    info = create_info(5, 1000, "eeg")
    sinfo = StreamInfo("test", "eeg", 5, 1000, np.float32, uuid.uuid4().hex)
    sinfo.set_channel_info(info)
    info_retrieved = sinfo.get_channel_info()
    compare_infos(info, info_retrieved)

    # test with FIFF file from the MNE sample dataset
    fname = testing.data_path() / "sample_audvis_raw.fif"
    raw = read_raw_fif(fname, preload=False)
    sinfo = StreamInfo(
        "test", "", len(raw.ch_names), raw.info["sfreq"], np.float32, uuid.uuid4().hex
    )
    sinfo.set_channel_info(raw.info)
    info_retrieved = sinfo.get_channel_info()
    compare_infos(raw.info, info_retrieved)

    # test from Outlet/Inlet
    outlet = StreamOutlet(sinfo)
    info_retrieved = outlet.get_sinfo().get_channel_info()
    compare_infos(raw.info, info_retrieved)
    inlet = StreamInlet(sinfo)
    inlet.open_stream()
    info_retrieved = inlet.get_sinfo().get_channel_info()
    compare_infos(raw.info, info_retrieved)

    close_io()
