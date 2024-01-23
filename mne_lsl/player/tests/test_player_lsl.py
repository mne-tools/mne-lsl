import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from mne.utils import check_version
from numpy.testing import assert_allclose

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF
else:
    from mne.io.constants import FIFF

from mne_lsl.lsl import StreamInlet, local_clock, resolve_streams
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.utils._tests import match_stream_and_raw_data


def _create_inlet(name: str) -> StreamInlet:
    """Create an inlet to the open-stream."""
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream(timeout=10)
    return inlet


def test_player(fname, raw, close_io):
    """Test a working and valid player."""
    name = "Player-test_player"
    player = Player(fname, name=name)
    assert "OFF" in player.__repr__()
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    player.start()
    assert "ON" in player.__repr__()
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name

    # try double start
    with pytest.warns(RuntimeWarning, match="player is already started"):
        player.start()

    # connect an inlet to the player
    inlet = StreamInlet(streams[0])
    inlet.open_stream(timeout=10)
    data, ts = inlet.pull_chunk()
    now = local_clock()
    assert_allclose(now, ts[-1], rtol=0, atol=0.1)  # 100 ms of freedom for slower CIs
    assert data.shape[1] == len(player.info["ch_names"])

    # check sampling rate
    fs = 1 / np.diff(ts)
    assert_allclose(fs, player.info["sfreq"])

    # check inlet information
    sinfo = inlet.get_sinfo()
    assert sinfo.n_channels == len(player.info["ch_names"])
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_types() == player.get_channel_types()
    assert sinfo.get_channel_units() == [
        str(int(ch["unit_mul"])) for ch in player.info["chs"]
    ]

    # check player vs raw
    assert player.info["sfreq"] == raw.info["sfreq"]
    assert player.info["ch_names"] == raw.info["ch_names"]
    assert player.get_channel_types() == raw.get_channel_types()
    assert [ch["unit_mul"] for ch in player.info["chs"]] == [
        ch["unit_mul"] for ch in raw.info["chs"]
    ]

    # check that the returned data array is in raw
    match_stream_and_raw_data(data.T, raw)
    close_io()
    player.stop()


def test_player_context_manager(fname):
    """Test a working and valid player as context manager."""
    name = "Player-test_player_context_manager"
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    with Player(fname, name=name):
        streams = resolve_streams()
        assert len(streams) == 1
        assert streams[0].name == name
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0


def test_player_context_manager_raw(raw):
    """Test a working and valid player as context manager from a raw object."""
    name = "Player-test_player_context_manager_raw"
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    with Player(raw, name=name) as player:
        streams = resolve_streams()
        assert len(streams) == 1
        assert streams[0].name == name
        assert player.info["ch_names"] == raw.info["ch_names"]
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0

    with pytest.warns(RuntimeWarning, match="raw file has no annotations"):
        with Player(raw, name=name, annotations=True) as player:
            streams = resolve_streams()
            assert len(streams) == 1
            assert streams[0].name == name
            assert player.info["ch_names"] == raw.info["ch_names"]


def test_player_context_manager_raw_annotations(raw_annotations):
    """Test a working player as context manager from a raw object with annotations."""
    name = "Player-test_player_context_manager_raw"
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    with Player(raw_annotations, name=name, annotations=False) as player:
        streams = resolve_streams()
        assert len(streams) == 1
        assert streams[0].name == name
        assert player.info["ch_names"] == raw_annotations.info["ch_names"]
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0

    with Player(raw_annotations, name=name) as player:
        streams = resolve_streams()
        assert len(streams) == 2
        assert any(stream.name == name for stream in streams)
        assert any(stream.name == f"{name}-annotations" for stream in streams)
        assert player.info["ch_names"] == raw_annotations.info["ch_names"]
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0


def test_player_invalid_arguments(fname):
    """Test creation of a player with invalid arguments."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Player("invalid-fname.something")
    with pytest.raises(ValueError, match="Unsupported file type"):
        Player(Path(__file__))
    with pytest.raises(TypeError, match="'name' must be an instance of str or None"):
        Player(fname, name=101)
    with pytest.raises(TypeError, match="'chunk_size' must be an integer"):
        Player(fname, name="101", chunk_size=101.0)
    with pytest.raises(ValueError, match="strictly positive integer"):
        Player(fname, name="101", chunk_size=-101)


def test_player_stop_invalid(fname):
    """Test stopping a player that is not started."""
    player = Player(fname, name="Player-test_stop_player_invalid")
    with pytest.raises(RuntimeError, match="The player is not started"):
        player.stop()
    player.start()
    player.stop()


def test_player_unit(mock_lsl_stream, raw, close_io):
    """Test getting and setting the player channel units."""
    player = mock_lsl_stream
    name = player.name
    assert player.get_channel_types() == raw.get_channel_types()
    assert player.get_channel_types(unique=True) == raw.get_channel_types(unique=True)
    ch_units = player.get_channel_units()
    assert ch_units == [(FIFF.FIFF_UNIT_NONE, FIFF.FIFF_UNITM_NONE)] + [
        (FIFF.FIFF_UNIT_V, FIFF.FIFF_UNITM_NONE)
    ] * (len(player.ch_names) - 1)

    # try setting channel units on a started player
    with pytest.raises(RuntimeError, match="player is already started"):
        player.set_channel_units({"F7": -6, "Fpz": "uv", "Fp2": "microvolts"})
    inlet = _create_inlet(name)
    data, _ = inlet.pull_chunk()
    match_stream_and_raw_data(data.T, raw)
    close_io()
    player.stop()

    # try setting channel units after stopping the player
    player.set_channel_units({"F7": -6, "Fpz": "uv", "Fp2": "microvolts"})
    player.start()
    inlet = _create_inlet(name)
    data, _ = inlet.pull_chunk()
    raw_ = raw.copy().apply_function(lambda x: x * 1e6, picks=["F7", "Fpz", "Fp2"])
    match_stream_and_raw_data(data.T, raw_)
    close_io()
    player.stop()

    # try re-setting the channel unit
    player.set_channel_units({"F7": -3})
    player.start()
    inlet = _create_inlet(name)
    data, _ = inlet.pull_chunk()
    raw_ = raw.copy()
    raw_.apply_function(lambda x: x * 1e3, picks="F7")
    raw_.apply_function(lambda x: x * 1e6, picks=["Fpz", "Fp2"])
    match_stream_and_raw_data(data.T, raw_)
    close_io()
    player.stop()


def test_player_rename_channels(mock_lsl_stream, raw, close_io):
    """Test channel renaming."""
    player = mock_lsl_stream
    name = player.name
    assert player._sinfo.get_channel_names() == player.info["ch_names"]
    with pytest.raises(RuntimeError, match="player is already started"):
        player.rename_channels(mapping={"F7": "EEG1"})
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    close_io()
    player.stop()

    # test changing channel names
    player.rename_channels({"F7": "EEG1", "Fp2": "EEG2"})
    raw_ = raw.copy().rename_channels({"F7": "EEG1", "Fp2": "EEG2"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    close_io()
    player.stop()

    # test re-changing the channel names
    player.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    raw_.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    close_io()
    player.stop()


def test_player_set_channel_types(mock_lsl_stream, raw, close_io):
    """Test channel type setting."""
    player = mock_lsl_stream
    name = player.name
    assert player._sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert player.get_channel_types(unique=False) == raw.get_channel_types(unique=False)
    with pytest.raises(RuntimeError, match="player is already started"):
        player.set_channel_types(mapping={"F7": "misc"})
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    close_io()
    player.stop()

    # test changing types
    player.set_channel_types(mapping={"F7": "eog", "Fp2": "eog"})
    raw_ = raw.copy().set_channel_types(mapping={"F7": "eog", "Fp2": "eog"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert sinfo.get_channel_types() == raw_.get_channel_types(unique=False)
    close_io()
    player.stop()

    # test rechanging types
    player.set_channel_types(mapping={"F7": "eeg", "Fp2": "ecg"})
    raw_ = raw.copy().set_channel_types(mapping={"Fp2": "ecg"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert sinfo.get_channel_types() == raw_.get_channel_types(unique=False)
    close_io()
    player.stop()

    # test unique
    assert sorted(player.get_channel_types(unique=True)) == sorted(
        raw.get_channel_types(unique=True)
    )


def test_player_anonymize(fname):
    """Test anonymization."""
    name = "Player-test_player_anonymize"
    player = Player(fname, name=name)
    assert player.name == name
    assert player.fname == fname
    player.info["subject_info"] = dict(
        id=101,
        first_name="Mathieu",
        sex=1,
    )
    with pytest.warns(RuntimeWarning, match="partially implemented"):
        player.anonymize()
    assert player.info["subject_info"] == dict(id=0, first_name="mne_anonymize", sex=0)
    player.start()
    assert "ON" in player.__repr__()
    with pytest.raises(RuntimeError, match="player is already started"):
        player.anonymize()
    player.stop()


def test_player_set_meas_date(fname):
    """Test player measurement date."""
    name = "Player-test_player_set_meas_date"
    player = Player(fname, name=name)
    assert player.name == name
    assert player.fname == fname
    assert player.info["meas_date"] is None
    with pytest.warns(RuntimeWarning, match="partially implemented"):
        meas_date = datetime(2023, 1, 25, tzinfo=timezone.utc)
        player.set_meas_date(meas_date)
    assert player.info["meas_date"] == meas_date
    player.start()
    assert "ON" in player.__repr__()
    with pytest.raises(RuntimeError, match="player is already started"):
        player.set_meas_date(datetime(2020, 1, 25, tzinfo=timezone.utc))
    assert player.info["meas_date"] == meas_date
    player.stop()


def test_player_annotations(raw_annotations, close_io):
    """Test player with annotations."""
    name = "Player-test_player_annotations"
    annotations = sorted(set(raw_annotations.annotations.description))
    player = Player(raw_annotations, name=name)
    assert player.name == name
    assert player.fname == Path(raw_annotations.filenames[0])
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    player.start()
    streams = resolve_streams()
    assert len(streams) == 2
    assert any(stream.name == name for stream in streams)
    assert any(stream.name == f"{name}-annotations" for stream in streams)

    # find annotation stream and open an inlet
    streams = sorted(streams, key=lambda stream: stream.name)
    inlet = StreamInlet(streams[0])
    inlet.open_stream(timeout=10)
    inlet_annotations = StreamInlet(streams[1])
    inlet_annotations.open_stream(timeout=10)

    # compare inlet stream info and annotations
    sinfo = inlet_annotations.get_sinfo()
    assert sinfo.n_channels == len(annotations)
    assert sinfo.get_channel_names() == annotations
    assert sinfo.get_channel_types() == ["annotations"] * sinfo.n_channels
    assert sinfo.get_channel_units() == ["none"] * sinfo.n_channels

    # compare content
    data, ts = inlet_annotations.pull_chunk(timeout=1)
    assert data.size != 0
    assert ts.size == data.shape[0]
    # compare with a Stream object for simplicity
    stream = Stream(bufsize=40, stype="annotations")
    stream.connect(processing_flags=["clocksync"])
    assert stream.info["ch_names"] == annotations
    assert stream.get_channel_types() == ["misc"] * sinfo.n_channels
    time.sleep(3)  # acquire some annotations
    for single, duration in zip(("bad_test", "test2", "test3"), (0.4, 0.1, 0.05)):
        data, ts = stream.get_data(picks=single)
        data = data.squeeze()
        assert ts.size == data.size
        idx = np.where(data != 0.0)[0]
        assert_allclose(data[idx], [duration] * idx.size)
        assert_allclose(np.diff(ts[idx]), 2, atol=1e-2)
    data, ts = stream.get_data(picks="test1")
    data = data.squeeze()
    idx = np.where(data != 0.0)[0]
    assert_allclose(np.unique(data[idx]), [0.2, 0.55])
    idx = np.where(data == 0.2)[0]
    diff = np.diff(ts[idx])
    expected = np.array([1.6, 0.3, 0.1])
    start = np.where(1 <= diff)[0][0]
    end = diff.size if diff.size <= start + 3 else start + 3
    assert_allclose(diff[start:end], expected[: end - start], atol=1e-2)

    # clean-up
    stream.disconnect()
    close_io()
    player.stop()
