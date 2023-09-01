from pathlib import Path

import numpy as np
import pytest
from mne.io import read_raw
from mne.utils import check_version
from numpy.testing import assert_allclose

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF
else:
    from mne.io.constants import FIFF

from bsl import Player, logger
from bsl.datasets import testing
from bsl.lsl import StreamInlet, local_clock, resolve_streams
from bsl.utils._tests import match_stream_and_raw_data

logger.propagate = True

fname = testing.data_path() / "sample-eeg-ant-raw.fif"
raw = read_raw(fname, preload=True)


def test_player(caplog):
    """Test a working and valid player."""
    name = "BSL-Player-test_player"
    player = Player(fname, name, 16)
    assert "OFF" in player.__repr__()
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    player.start()
    assert "ON" in player.__repr__()
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name

    # try double start
    caplog.set_level(30)  # WARNING
    caplog.clear()
    player.start()
    assert "player is already started" in caplog.text

    # connect an inlet to the player
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
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
        str(ch["unit_mul"]) for ch in player.info["chs"]
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
    del inlet
    player.stop()


def test_player_context_manager():
    """Test a working and valid player as context manager."""
    name = "BSL-Player-test_player_context_manager"
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    with Player(fname, name, 16):
        streams = resolve_streams(timeout=0.1)
        assert len(streams) == 1
        assert streams[0].name == name
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0


def test_player_invalid_arguments():
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


def test_player_stop_invalid():
    """Test stopping a player that is not started."""
    player = player = Player(fname, "BSL-Player-test_stop_player_invalid", 16)
    with pytest.raises(RuntimeError, match="The player is not started"):
        player.stop()
    player.start()
    player.stop()


def test_player_unit():
    """Test getting and setting the player channel units."""
    name = "BSL-Player-test_player_unit"
    player = Player(fname, name, 16)
    assert player.get_channel_types() == raw.get_channel_types()
    assert player.get_channel_types(unique=True) == raw.get_channel_types(unique=True)
    ch_units = player.get_channel_units()
    assert ch_units == [(FIFF.FIFF_UNIT_V, FIFF.FIFF_UNITM_NONE)] * len(player.ch_names)

    # try setting channel units on a started player
    player.start()
    with pytest.raises(RuntimeError, match="player is already started"):
        player.set_channel_units({"Fp1": -6, "Fpz": "uv", "Fp2": "microvolts"})
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
    data, _ = inlet.pull_chunk()
    match_stream_and_raw_data(data.T, raw)
    del inlet
    player.stop()

    # try setting channel units after stopping the player
    player.set_channel_units({"Fp1": -6, "Fpz": "uv", "Fp2": "microvolts"})
    player.start()
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
    data, _ = inlet.pull_chunk()
    raw_ = raw.copy().apply_function(lambda x: x*1e6, picks=["Fp1", "Fpz", "Fp2"])
    match_stream_and_raw_data(data.T, raw_)
    del inlet
    player.stop()

    # try re-setting the channel unit
    player.set_channel_units({"Fp1": -3})
    player.start()
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
    data, _ = inlet.pull_chunk()
    raw_ = raw.copy()
    raw_.apply_function(lambda x: x*1e3, picks="Fp1")
    raw_.apply_function(lambda x: x*1e6, picks=["Fpz", "Fp2"])
    match_stream_and_raw_data(data.T, raw_)
    del inlet
    player.stop()


def test_player_rename_channels():
    """Test channel renaming."""
    name = "BSL-Player-test_player_unit"
    player = Player(fname, name, 16)
    assert player._sinfo.get_channel_names() == player.info["ch_names"]
    player.start()
    with pytest.raises(RuntimeError, match="player is already started"):
        player.rename_channels(mapping={"Fp1": "EEG1"})
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    del inlet
    player.stop()

    # test changing channel names
    player.rename_channels({"Fp1": "EEG1", "Fp2": "EEG2"})
    raw_ = raw.copy().rename_channels({"Fp1": "EEG1", "Fp2": "EEG2"})
    player.start()
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    del inlet
    player.stop()

    # test re-changing the channel names
    player.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    raw_.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    player.start()
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    del inlet
    player.stop()
