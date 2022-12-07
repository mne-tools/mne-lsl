import re
import time

import mne
import numpy as np
import pytest

from bsl import StreamPlayer, StreamReceiver, logger, set_log_level
from bsl.datasets import eeg_resting_state
from bsl.utils._tests import requires_eeg_resting_state_dataset

set_log_level("INFO")
logger.propagate = True


@requires_eeg_resting_state_dataset
def test_stream_receiver():
    """Test stream receiver default functionalities."""
    stream = "StreamPlayer"

    with StreamPlayer(stream, eeg_resting_state.data_path()):
        sr = StreamReceiver(bufsize=1, winsize=0.2, stream_name=stream)
        assert stream in sr.streams
        assert sr._connected

        # test infos
        info = sr.mne_infos[stream]
        assert isinstance(info, mne.io.Info)
        assert info.ch_names == sr.streams[stream].ch_list
        assert info["sfreq"] == sr.streams[stream].sample_rate
        info._check_consistency()

        # test get_xxx numpy
        time.sleep(1)
        sr.acquire()
        window, timestamps_window = sr.get_window()
        buffer, timestamps_buffer = sr.get_buffer()
        assert (
            window.shape[0]
            == buffer.shape[0]
            == len(sr.streams[stream].ch_list)
        )
        assert (
            window.shape[1]
            == len(timestamps_window)
            == round(sr.streams[stream].sample_rate * 0.2)
        )
        assert (
            buffer.shape[1]
            == len(timestamps_buffer)
            == round(sr.streams[stream].sample_rate * 1)
        )

        window2, timestamps_window2 = sr.get_window()
        assert np.all(window == window2)
        assert np.all(timestamps_window == timestamps_window2)

        time.sleep(0.2)
        previous_buffer = buffer
        sr.acquire()
        window3, _ = sr.get_window()
        buffer3, _ = sr.get_buffer()
        assert window3[:, 0] in previous_buffer

        # test get_window MNE raw
        raw, _ = sr.get_window(return_raw=True)
        assert isinstance(raw, mne.io.BaseRaw)
        assert raw.info.ch_names == info.ch_names
        assert raw.info["sfreq"] == info["sfreq"]
        raw.info._check_consistency()
        data = raw.get_data(picks="all").T
        assert np.all(data == window3)

        # test get_buffer MNE raw
        raw, _ = sr.get_buffer(return_raw=True)
        assert isinstance(raw, mne.io.BaseRaw)
        assert raw.info.ch_names == info.ch_names
        assert raw.info["sfreq"] == info["sfreq"]
        raw.info._check_consistency()
        data = raw.get_data(picks="all").T
        assert np.all(data == buffer3)

        del sr


@requires_eeg_resting_state_dataset
def test_receiving_multi_streams():
    """Test StreamReceiver multi-streams functionalities."""
    with StreamPlayer(
        "StreamPlayer1", eeg_resting_state.data_path()
    ), StreamPlayer("StreamPlayer2", eeg_resting_state.data_path()):

        # test connect to only one
        sr = StreamReceiver(
            bufsize=1, winsize=0.2, stream_name="StreamPlayer1"
        )
        assert "StreamPlayer1" in sr.streams
        assert sr.stream_name == ["StreamPlayer1"]
        del sr

        # test connect to both
        sr = StreamReceiver(bufsize=1, winsize=0.2)
        assert "StreamPlayer1" in sr.streams
        assert "StreamPlayer2" in sr.streams
        assert sr.stream_name is None

        # test get_xxx numpy
        time.sleep(1)
        sr.acquire()
        window1, timestamps_window1 = sr.get_window("StreamPlayer1")
        buffer1, timestamps_buffer1 = sr.get_buffer("StreamPlayer1")
        window2, timestamps_window2 = sr.get_window("StreamPlayer2")
        buffer2, timestamps_buffer2 = sr.get_buffer("StreamPlayer2")
        assert (
            window1.shape[1]
            == window2.shape[1]
            == buffer1.shape[1]
            == buffer2.shape[1]
            == len(sr.streams["StreamPlayer1"].ch_list)
            == len(sr.streams["StreamPlayer2"].ch_list)
        )
        assert (
            window1.shape[0]
            == window2.shape[0]
            == len(timestamps_window1)
            == len(timestamps_window2)
            == round(sr.streams["StreamPlayer1"].sample_rate * 0.2)
            == round(sr.streams["StreamPlayer2"].sample_rate * 0.2)
        )
        assert (
            buffer1.shape[0]
            == buffer2.shape[0]
            == len(timestamps_buffer1)
            == len(timestamps_buffer2)
            == round(sr.streams["StreamPlayer1"].sample_rate * 1)
            == round(sr.streams["StreamPlayer2"].sample_rate * 1)
        )

        # test get_window MNE raw
        raw1, _ = sr.get_window("StreamPlayer1", return_raw=True)
        raw2, _ = sr.get_window("StreamPlayer2", return_raw=True)
        assert isinstance(raw1, mne.io.BaseRaw)
        assert isinstance(raw2, mne.io.BaseRaw)
        info1 = sr.mne_infos["StreamPlayer1"]
        info2 = sr.mne_infos["StreamPlayer2"]
        assert (
            raw1.info.ch_names
            == raw2.info.ch_names
            == info1.ch_names
            == info2.ch_names
        )
        assert (
            raw1.info["sfreq"]
            == raw2.info["sfreq"]
            == info1["sfreq"]
            == info2["sfreq"]
        )
        raw1.info._check_consistency()
        raw2.info._check_consistency()
        data1 = raw1.get_data(picks="all").T
        data2 = raw2.get_data(picks="all").T
        assert np.all(data1 == window1)
        assert np.all(data2 == window2)

        # test get_buffer MNE raw
        raw1, _ = sr.get_buffer("StreamPlayer1", return_raw=True)
        raw2, _ = sr.get_buffer("StreamPlayer2", return_raw=True)
        assert isinstance(raw1, mne.io.BaseRaw)
        assert isinstance(raw2, mne.io.BaseRaw)
        info1 = sr.mne_infos["StreamPlayer1"]
        info2 = sr.mne_infos["StreamPlayer2"]
        assert (
            raw1.info.ch_names
            == raw2.info.ch_names
            == info1.ch_names
            == info2.ch_names
        )
        assert (
            raw1.info["sfreq"]
            == raw2.info["sfreq"]
            == info1["sfreq"]
            == info2["sfreq"]
        )
        raw1.info._check_consistency()
        raw2.info._check_consistency()
        data1 = raw1.get_data(picks="all").T
        data2 = raw2.get_data(picks="all").T
        assert np.all(data1 == buffer1)
        assert np.all(data2 == buffer2)

        del sr


@requires_eeg_resting_state_dataset
def test_properties():
    """Test the StreamReceiver properties."""
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        sr = StreamReceiver(bufsize=1, winsize=0.2)
        assert sr.winsize == sr._winsize == 0.2
        assert sr.bufsize == sr._bufsize == 1
        assert sr.stream_name is None
        assert sr.connected == sr._connected
        assert sr.connected
        assert sr.mne_infos == sr._mne_infos
        assert isinstance(sr.mne_infos, dict)
        assert len(sr.mne_infos) == 1
        assert isinstance(sr.mne_infos["StreamPlayer"], mne.io.Info)
        assert sr.streams == sr._streams
        assert isinstance(sr.streams, dict)
        assert len(sr.streams) == 1
        assert "StreamPlayer" in sr.streams

        with pytest.raises(AttributeError, match="can't set attribute"):
            sr.winsize = 1
        with pytest.raises(AttributeError, match="can't set attribute"):
            sr.bufsize = 2
        with pytest.raises(AttributeError, match="can't set attribute"):
            sr.stream_name = "StreamPlayer"
        with pytest.raises(AttributeError, match="can't set attribute"):
            sr.connected = False
        with pytest.raises(AttributeError, match="can't set attribute"):
            sr.mne_infos = dict()
        with pytest.raises(AttributeError, match="can't set attribute"):
            sr.streams = dict()

        sr.disconnect()
        assert sr.streams == dict()
        assert sr.connected is False

        del sr


@requires_eeg_resting_state_dataset
def test_get_method_warning_and_errors(caplog):
    """Test the checking done in get_xxx methods."""
    with StreamPlayer(
        "StreamPlayer1", eeg_resting_state.data_path()
    ), StreamPlayer("StreamPlayer2", eeg_resting_state.data_path()):
        sr = StreamReceiver(bufsize=1, winsize=1, stream_name=None)

        # Disconnect and acquire
        sr.disconnect()
        with pytest.raises(
            RuntimeError,
            match="StreamReceiver is not connected to any " "streams.",
        ):
            sr.acquire()

        # Acquire, disconnect and get.
        sr.connect()
        time.sleep(1)
        sr.acquire()
        time.sleep(0.05)
        sr.disconnect()
        with pytest.raises(
            RuntimeError,
            match="StreamReceiver is not connected to any " "streams.",
        ):
            sr.get_window()
        with pytest.raises(
            RuntimeError,
            match="StreamReceiver is not connected to any " "streams.",
        ):
            sr.get_buffer()

        # multiple streams
        sr.connect()
        time.sleep(1)
        sr.acquire()
        with pytest.raises(
            RuntimeError,
            match="StreamReceiver is connected to multiple "
            "streams. Please provide the stream_name "
            "argument.",
        ):
            sr.get_window()
        with pytest.raises(
            RuntimeError,
            match="StreamReceiver is connected to multiple "
            "streams. Please provide the stream_name "
            "argument.",
        ):
            sr.get_buffer()

        # stream_name not in streams
        with pytest.raises(
            KeyError, match="StreamReceiver is not connected to '101'."
        ):
            sr.get_window(stream_name="101")
        with pytest.raises(
            KeyError, match="StreamReceiver is not connected to '101'."
        ):
            sr.get_buffer(stream_name="101")

        # forgot to call .acquire()
        sr.disconnect()
        sr.connect()
        with pytest.raises(
            AttributeError,
            match=re.escape(
                ".acquire() must be called before " ".get_window()."
            ),
        ):
            sr.get_window(stream_name="StreamPlayer1")
        with pytest.raises(
            AttributeError,
            match=re.escape(
                ".acquire() must be called before " ".get_buffer()."
            ),
        ):
            sr.get_buffer(stream_name="StreamPlayer1")

        # not enough samples
        del sr
        sr = StreamReceiver(bufsize=1, winsize=1, stream_name=None)
        # caplog.clear()
        sr.acquire()
        sr.get_window(stream_name="StreamPlayer1")
        assert (
            "The buffer of StreamPlayer1 does not contain enough samples. "
            "Returning the available samples."
        ) in caplog.text

        # could not convert to MNE raw
        time.sleep(1)
        sr.acquire()
        caplog.clear()
        sr._mne_infos["StreamPlayer1"] = None
        data, _ = sr.get_window(stream_name="StreamPlayer1", return_raw=True)
        assert (
            "The LSL stream StreamPlayer1 can not be converted to MNE raw "
            "instance. Returning numpy arrays."
        ) in caplog.text
        assert isinstance(data, np.ndarray)
        caplog.clear()
        data, _ = sr.get_buffer(stream_name="StreamPlayer1", return_raw=True)
        assert (
            "The LSL stream StreamPlayer1 can not be converted to MNE raw "
            "instance. Returning numpy arrays."
        ) in caplog.text
        assert isinstance(data, np.ndarray)

        del sr


@requires_eeg_resting_state_dataset
def test_connect_disconnect():
    """Test connect and disconnect methods."""
    with StreamPlayer(
        "StreamPlayer1", eeg_resting_state.data_path()
    ), StreamPlayer("StreamPlayer2", eeg_resting_state.data_path()):
        sr = StreamReceiver(bufsize=1, winsize=1, stream_name="StreamPlayer1")
        assert sr.connected
        assert sr.stream_name == ["StreamPlayer1"]
        assert "StreamPlayer1" in sr.streams
        assert "StreamPlayer2" not in sr.streams

        sr.connect("StreamPlayer2", force=False)
        assert sr.connected
        assert sr.stream_name == ["StreamPlayer1"]
        assert "StreamPlayer1" in sr.streams
        assert "StreamPlayer2" not in sr.streams

        sr.connect("StreamPlayer2", force=True)
        assert sr.connected
        assert sr.stream_name == ["StreamPlayer1", "StreamPlayer2"]
        assert "StreamPlayer1" in sr.streams
        assert "StreamPlayer2" in sr.streams

        sr.disconnect("StreamPlayer1")
        assert sr.connected
        assert sr.stream_name == ["StreamPlayer2"]
        assert "StreamPlayer1" not in sr.streams
        assert "StreamPlayer2" in sr.streams

        sr.connect("StreamPlayer1", force=True)
        assert sr.connected
        assert sr.stream_name == ["StreamPlayer2", "StreamPlayer1"]
        assert "StreamPlayer1" in sr.streams
        assert "StreamPlayer2" in sr.streams

        sr.disconnect()
        assert not sr.connected
        assert sr.stream_name == list()
        assert "StreamPlayer1" not in sr.streams
        assert "StreamPlayer2" not in sr.streams

        sr.connect(stream_name=None, force=True)
        assert sr.connected
        assert sr.stream_name is None
        assert "StreamPlayer1" in sr.streams
        assert "StreamPlayer2" in sr.streams

        del sr


@requires_eeg_resting_state_dataset
def test_checker_bufsize(caplog):
    """Test the checker for argument bufsize."""
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        # Valid
        sr = StreamReceiver(bufsize=1, winsize=0.2)
        assert isinstance(sr.bufsize, float)
        assert sr.bufsize == 1.0
        del sr

        # Invalid value
        with pytest.raises(
            ValueError,
            match="Argument bufsize must be a strictly"
            " positive int or a float. "
            "Provided: %s" % -101,
        ):
            StreamReceiver(bufsize=-101, winsize=0.2)

        # Invalid type
        with pytest.raises(TypeError, match="'bufsize' must be an instance"):
            StreamReceiver(bufsize=[101], winsize=0.2)

        # Smaller than winsize
        caplog.clear()
        sr = StreamReceiver(bufsize=1, winsize=2)
        assert isinstance(sr.bufsize, float)
        assert isinstance(sr.winsize, float)
        assert sr.bufsize == 2.0
        assert sr.winsize == 2.0
        assert (
            "Buffer size %.2f is smaller than window size. " % 1.0
            + "Setting to %.2f." % 2.0
        ) in caplog.text

        del sr


@requires_eeg_resting_state_dataset
def test_checker_winsize():
    """Test the checker for argument winsize."""
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        # Valid
        sr = StreamReceiver(bufsize=1, winsize=0.2)
        assert isinstance(sr.winsize, float)
        assert sr.winsize == 0.2
        del sr

        # Invalid value
        with pytest.raises(
            ValueError,
            match="Argument winsize must be a strictly positive"
            " int or a float. Provided: %s" % -101,
        ):
            StreamReceiver(bufsize=1, winsize=-101)

        # Invalid type
        with pytest.raises(TypeError, match="'winsize' must be an instance"):
            StreamReceiver(bufsize=1, winsize=[101])


@requires_eeg_resting_state_dataset
def test_checker_stream_name(caplog):
    """Test the checker for argument stream_name."""
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        # Valid - None
        sr = StreamReceiver(bufsize=1, winsize=0.2, stream_name=None)
        assert sr.stream_name is None
        del sr

        # Valid - str
        sr = StreamReceiver(bufsize=1, winsize=0.2, stream_name="StreamPlayer")
        assert sr.stream_name == ["StreamPlayer"]
        del sr

        # Valid - list of str
        sr = StreamReceiver(
            bufsize=1, winsize=0.2, stream_name=["StreamPlayer"]
        )
        assert sr.stream_name == ["StreamPlayer"]
        del sr

        # Valid - tuple of str
        sr = StreamReceiver(
            bufsize=1, winsize=0.2, stream_name=("StreamPlayer",)
        )
        assert sr.stream_name == ["StreamPlayer"]
        del sr

        # Invalid - list of 'not str'
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument stream_name must be a string or a "
                "list of strings. Provided: [1, 0, 1]"
            ),
        ):
            StreamReceiver(bufsize=1, winsize=0.2, stream_name=[1, 0, 1])

        # Invalid type
        with pytest.raises(
            TypeError, match="'stream_name' must be an instance"
        ):
            StreamReceiver(bufsize=1, winsize=0.2, stream_name=101)


@requires_eeg_resting_state_dataset
def test_representation():
    """Test the representation method."""
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        sr = StreamReceiver(bufsize=1, winsize=0.2)
        expected = "<Receiver: ('StreamPlayer',) | ON | buf: 1.0s - win: 0.2s>"
        assert sr.__repr__() == expected

        sr.disconnect()
        expected = "<Receiver: () | OFF | buf: 1.0s - win: 0.2s>"
        assert sr.__repr__() == expected

        # Explicit stream name.
        sr.connect(stream_name="StreamPlayer")
        expected = "<Receiver: ('StreamPlayer',) | ON | buf: 1.0s - win: 0.2s>"
        assert sr.__repr__() == expected

        sr.disconnect()
        expected = "<Receiver: () | OFF | buf: 1.0s - win: 0.2s>"
        assert sr.__repr__() == expected

        del sr
