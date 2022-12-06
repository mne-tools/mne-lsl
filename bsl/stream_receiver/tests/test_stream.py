import time

import mne
import pytest

from bsl import StreamPlayer, logger, set_log_level
from bsl.datasets import eeg_resting_state
from bsl.lsl import resolve_streams
from bsl.stream_receiver._buffer import Buffer
from bsl.stream_receiver._stream import StreamEEG, StreamMarker  # noqa: F401
from bsl.utils._tests import requires_eeg_resting_state_dataset

set_log_level("INFO")
logger.propagate = True


def _check_stream_properties(
    stream,
    streamInfo,
    stream_name,
    serial,
    is_slave,
    lsl_time_offset,
    blocking,
    blocking_time,
    dataset,
):
    """Check all the _Stream properties from a given stream."""
    raw = mne.io.read_raw_fif(fname=dataset.data_path(), preload=True)

    # test streamInfo
    assert stream.streamInfo == stream._streamInfo == streamInfo
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.streamInfo = "101"

    # test sample rate
    assert stream.sample_rate == stream._sample_rate == raw.info["sfreq"]
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.sample_rate = "2048"

    # test name
    assert stream.name == stream._name == stream_name
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.name = "random-name"

    # test serial
    assert stream.serial == stream._serial == serial
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.serial = "101"

    # test is_slave
    assert stream.is_slave == stream._is_slave == bool(is_slave)
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.is_slave = not stream.is_slave

    # test channel list
    assert stream.ch_list == stream._ch_list == raw.ch_names
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.ch_list = [1, 0, 1]

    # test lsl_time_offset
    assert stream.lsl_time_offset == stream._lsl_time_offset == lsl_time_offset
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.lsl_time_offset = 101

    # test blocking
    assert stream.blocking == stream._blocking == bool(blocking)
    stream.blocking = not bool(blocking)
    assert stream.blocking == stream._blocking == (not bool(blocking))
    stream.blocking = bool(blocking)
    assert stream.blocking == stream._blocking == bool(blocking)

    # test blocking time
    assert stream.blocking_time == stream._blocking_time == blocking_time
    stream.blocking_time = blocking_time + 1
    assert stream.blocking_time == stream._blocking_time == blocking_time + 1
    stream.blocking_time = blocking_time
    assert stream.blocking_time == stream._blocking_time == blocking_time

    # test buffer
    assert isinstance(stream.buffer, Buffer)
    with pytest.raises(AttributeError, match="can't set attribute"):
        stream.buffer = 101


def test_stream_marker():
    """Test StreamMarker class used by the StreamReceiver."""
    # TODO
    pass


@requires_eeg_resting_state_dataset
def test_stream_eeg(caplog):
    """Test StreamEEG class used by the StreamReceiver."""
    # Default
    with StreamPlayer("StreamPlayer", eeg_resting_state.data_path()):
        streamInfos = resolve_streams()
        for streamInfo in streamInfos:
            if streamInfo.name == "StreamPlayer":
                break

        stream = StreamEEG(streamInfo, bufsize=1, winsize=0.5)
        _check_stream_properties(
            stream=stream,
            streamInfo=streamInfo,
            stream_name="StreamPlayer",
            serial="N/A",
            is_slave=False,
            lsl_time_offset=None,
            blocking=True,
            blocking_time=5,
            dataset=eeg_resting_state,
        )

        # test scaling factor
        assert stream.scaling_factor == stream._scaling_factor == 1
        stream.scaling_factor = 1e-6
        assert stream.scaling_factor == stream._scaling_factor == 1e-6
        stream.scaling_factor = 1
        assert stream.scaling_factor == stream._scaling_factor == 1
        with pytest.raises(
            ValueError,
            match="Property scaling_factor must be a strictly "
            "positive number. Provided: 0",
        ):
            stream.scaling_factor = 0
        with pytest.raises(
            ValueError,
            match="Property scaling_factor must be a strictly "
            "positive number. Provided: -1",
        ):
            stream.scaling_factor = -1

        # test acquire and bufsize/winsize
        time.sleep(1.1)
        stream.acquire()
        assert stream.lsl_time_offset is not None
        assert stream.buffer.bufsize == round(1 * stream.sample_rate) == 512
        assert stream.buffer.winsize == round(0.5 * stream.sample_rate) == 256
        assert len(stream.buffer.data) == len(stream.buffer.timestamps)
        assert len(stream.buffer.data) == stream.buffer.bufsize

    # Test disconnection/reconnection
    sp = StreamPlayer("StreamPlayer", eeg_resting_state.data_path())
    sp.start()

    streamInfos = resolve_streams()
    for streamInfo in streamInfos:
        if streamInfo.name == "StreamPlayer":
            break

    caplog.clear()
    stream = StreamEEG(streamInfo, bufsize=1, winsize=0.5)
    stream.blocking = False
    stream.acquire()
    sp.stop()
    stream.acquire()
    assert "Timeout occurred" not in caplog.text

    # Test disconnection timeout
    caplog.clear()
    sp.start()
    stream.blocking = True
    stream.blocking_time = 1
    sp.stop()
    stream.acquire()
    assert "Timeout occurred [1.0secs] while acquiring data" in caplog.text
