from pathlib import Path

import numpy as np
import pytest
from mne.utils import check_version
from numpy.testing import assert_allclose

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF
else:
    from mne.io.constants import FIFF

from mne_lsl import logger
from mne_lsl.lsl import StreamInlet, local_clock, resolve_streams
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.utils._tests import match_stream_and_raw_data

logger.propagate = True


def test_player(caplog, fname, raw, close_io):
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


def test_player_context_manager(fname):
    """Test a working and valid player as context manager."""
    name = "Player-test_player_context_manager"
    streams = resolve_streams(timeout=0.1)
    assert len(streams) == 0
    with Player(fname, name=name):
        streams = resolve_streams(timeout=0.1)
        assert len(streams) == 1
        assert streams[0].name == name
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

    # try setting channel units after stopping the player
    player.set_channel_units({"F7": -6, "Fpz": "uv", "Fp2": "microvolts"})
    player.start()
    inlet = _create_inlet(name)
    data, _ = inlet.pull_chunk()
    raw_ = raw.copy().apply_function(lambda x: x * 1e6, picks=["F7", "Fpz", "Fp2"])
    match_stream_and_raw_data(data.T, raw_)
    close_io()

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

    # test changing channel names
    player.rename_channels({"F7": "EEG1", "Fp2": "EEG2"})
    raw_ = raw.copy().rename_channels({"F7": "EEG1", "Fp2": "EEG2"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    close_io()

    # test re-changing the channel names
    player.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    raw_.rename_channels({"EEG1": "EEG101", "EEG2": "EEG202"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == player.info["ch_names"]
    assert sinfo.get_channel_names() == raw_.info["ch_names"]
    close_io()


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

    # test changing types
    player.set_channel_types(mapping={"F7": "eog", "Fp2": "eog"})
    raw_ = raw.copy().set_channel_types(mapping={"F7": "eog", "Fp2": "eog"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert sinfo.get_channel_types() == raw_.get_channel_types(unique=False)
    close_io()

    # test rechanging types
    player.set_channel_types(mapping={"F7": "eeg", "Fp2": "ecg"})
    raw_ = raw.copy().set_channel_types(mapping={"Fp2": "ecg"})
    player.start()
    inlet = _create_inlet(name)
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_types() == player.get_channel_types(unique=False)
    assert sinfo.get_channel_types() == raw_.get_channel_types(unique=False)
    close_io()

    # test unique
    assert sorted(player.get_channel_types(unique=True)) == sorted(
        raw.get_channel_types(unique=True)
    )


def _create_inlet(name: str) -> StreamInlet:
    """Create an inlet to the open-stream."""
    streams = resolve_streams()
    assert len(streams) == 1
    assert streams[0].name == name
    inlet = StreamInlet(streams[0])
    inlet.open_stream()
    return inlet
