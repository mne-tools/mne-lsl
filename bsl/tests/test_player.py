import uuid
from collections import Counter

import numpy as np
from mne.io import read_raw
from numpy.testing import assert_allclose

from bsl import Player
from bsl.datasets import eeg_resting_state_short
from bsl.lsl import StreamInlet, local_clock, resolve_streams

fname = eeg_resting_state_short.data_path()
raw = read_raw(fname, preload=True)


def test_player():
    """Test a working and valid player."""
    name = f"BSL-Player-{uuid.uuid4().hex[:6]}"
    try:
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
        assert_allclose(now, ts[-1], rtol=0, atol=0.1)  # give 100 ms to the slower CIs
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
            ch["unit_mul"] for ch in player.info["chs"]
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
    except Exception as error:
        raise error
    finally:
        del player
        del inlet
