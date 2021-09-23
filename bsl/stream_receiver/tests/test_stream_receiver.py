import mne
import time
import pytest
import numpy as np

from bsl import StreamReceiver
from bsl.datasets import sample
from bsl.utils._testing import Stream, requires_sample_dataset


@requires_sample_dataset
def test_receiver():
    """Test receiver functionalities."""
    dataset = sample
    stream = 'StreamPlayer'
    with Stream(stream, dataset):
        sr = StreamReceiver(bufsize=1, winsize=0.2)
        assert stream in sr.streams

        # test properties
        assert sr.bufsize == 1
        assert sr.winsize == 0.2
        assert sr.stream_name is None
        assert sr.connected

        # test infos
        info = sr.mne_infos[stream]
        assert isinstance(info, mne.io.Info)
        assert info.ch_names == sr.streams[stream].ch_list
        assert info['sfreq'] == sr.streams[stream].sample_rate
        info._check_consistency()

        # test get_xxx methods
        # raises are tested in multi_stream testing
        time.sleep(1)
        sr.acquire()
        window, timestamps_window = sr.get_window()
        buffer, timestamps_buffer = sr.get_buffer()
        assert window.shape[1] == buffer.shape[1] == \
            len(sr.streams[stream].ch_list)
        assert window.shape[0] == len(timestamps_window) == \
            round(sr.streams[stream].sample_rate * 0.2)
        assert buffer.shape[0] == len(timestamps_buffer) == \
            round(sr.streams[stream].sample_rate * 1)

        window2, timestamps_window2 = sr.get_window()
        assert np.all(window == window2)
        assert np.all(timestamps_window == timestamps_window2)

        time.sleep(0.2)
        previous_buffer = buffer
        sr.acquire()
        window2, timestamps_window2 = sr.get_window()
        assert window2[0, :] in previous_buffer

        # test get_xxx MNE raw
        window, _ = sr.get_window(return_raw=True)
        assert isinstance(window, mne.io.BaseRaw)
        assert window.info.ch_names == info.ch_names
        assert window.info['sfreq'] == info['sfreq']
        window.info._check_consistency()

        # test properties setters
        sr.bufsize = 2
        assert sr.bufsize == 2
        assert sr.connected
        assert sr.streams[stream].buffer.bufsize == \
            round(2 * sr.streams[stream].sample_rate)

        sr.winsize = 2
        assert sr.winsize == 2
        assert sr.connected
        assert sr.streams[stream].buffer.winsize == \
            round(2 * sr.streams[stream].sample_rate)

        sr.stream_name = 'random fake stream'
        assert sr.connected
        assert stream in sr.streams


@requires_sample_dataset
def test_receiver_multi_streams():
    """Test StreamReceiver multi-streams functionalities."""
    dataset = sample
    with Stream('StreamPlayer1', dataset), Stream('StreamPlayer2', dataset):
        # test connect to only one
        sr = StreamReceiver(bufsize=1, winsize=0.2,
                            stream_name='StreamPlayer1')
        assert 'StreamPlayer1' in sr.streams
        assert sr.stream_name == ['StreamPlayer1']

        # test connect to both
        sr = StreamReceiver(bufsize=1, winsize=0.2)
        assert 'StreamPlayer1' in sr.streams
        assert 'StreamPlayer2' in sr.streams
        assert sr.stream_name is None

        # test disconnect
        sr.disconnect('StreamPlayer10')
        assert 'StreamPlayer1' in sr.streams
        assert 'StreamPlayer2' in sr.streams
        assert sr.connected
        sr.disconnect('StreamPlayer1')
        assert 'StreamPlayer1' not in sr.streams
        assert 'StreamPlayer2' in sr.streams
        assert sr.connected
        sr.disconnect()
        assert len(sr.streams) == 0
        assert not sr.connected

        # test when no stream is connected
        with pytest.raises(RuntimeError):
            sr.acquire()
            sr.get_window()
            sr.get_buffer()

        # test reconnect
        sr.connect()
        assert 'StreamPlayer1' in sr.streams
        assert 'StreamPlayer2' in sr.streams
        assert sr.connected
        sr.connect(force=True)
        assert 'StreamPlayer1' in sr.streams
        assert 'StreamPlayer2' in sr.streams
        assert sr.connected

        # test get_xxx methods
        sr.acquire()
        with pytest.raises(KeyError):
            sr.get_window(stream_name='random fake stream')
            sr.get_buffer(stream_name='random fake stream')
        with pytest.raises(RuntimeError):
            sr.get_window(stream_name=None)
            sr.get_buffer(stream_name=None)

        window, timestamps = sr.get_window(stream_name='StreamPlayer1')
        assert len(timestamps) == window.shape[0]
        assert window.shape[1] == len(sr.streams['StreamPlayer1'].ch_list)
