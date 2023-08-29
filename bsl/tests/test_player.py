from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from mne.io import read_raw
from numpy.testing import assert_allclose

from bsl import Player
from bsl.datasets import testing
from bsl.lsl import StreamInlet, local_clock, resolve_streams

fname = testing.data_path() / "sample-eeg-ant-raw.fif"
raw = read_raw(fname, preload=True)


def test_player():
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
    idx = [np.where(raw[:, :][0] == data[0, k])[1] for k in range(data.shape[1])]
    idx = np.concatenate(idx)
    counter = Counter(idx)
    idx, n_channels = counter.most_common()[0]
    assert n_channels == data.shape[1]
    assert n_channels == sinfo.n_channels
    start = idx
    stop = start + data.shape[0]
    if stop <= raw.times.size:
        assert_allclose(data.T, raw[:, start:stop][0])
    else:
        raw_data = np.hstack((raw[:, start:][0], raw[:, :][0]))[:, : stop - start]
        assert_allclose(data.T, raw_data)
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
