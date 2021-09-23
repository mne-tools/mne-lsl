import mne
import time
import pylsl

from bsl.datasets import sample
from bsl.stream_receiver._stream import StreamMarker, StreamEEG
from bsl.utils._testing import Stream, requires_sample_dataset


def test_stream_marker():
    """Test StreamMarker class used by the StreamReceiver."""
    # TODO
    pass


@requires_sample_dataset
def test_stream_eeg():
    """Test StreamEEG class used by the StreamReceiver."""
    dataset = sample
    raw = mne.io.read_raw_fif(fname=dataset.data_path(), preload=True)
    with Stream('StreamPlayer', dataset):
        streamInfos = pylsl.resolve_streams()
        for streamInfo in streamInfos:
            if streamInfo.name() == 'StreamPlayer':
                break
        stream = StreamEEG(streamInfo, bufsize=1, winsize=0.5)

        # test name
        assert stream.name == 'StreamPlayer'

        # test serial
        assert stream.serial == 'N/A'

        # test streamInfo
        assert stream.streamInfo == streamInfo

        # test sample rate
        assert stream.sample_rate == raw.info['sfreq']

        # test channel list
        assert stream.ch_list == raw.ch_names

        # test blocking
        assert stream.blocking
        stream.blocking = False
        assert not stream.blocking
        stream.blocking = 1
        assert stream.blocking

        # test blocking time
        assert stream.blocking_time == 5
        stream.blocking_time = 3
        assert stream.blocking_time == 3
        stream.blocking_time = 5
        assert stream.blocking_time == 5

        # test lsl_time_offset
        assert stream.lsl_time_offset is None

        # test stream multiplier
        assert stream.multiplier == 1
        stream.multiplier = 1e-6
        assert stream.multiplier == 1e-6
        stream.multiplier = 1
        assert stream.multiplier == 1

        # test find trigger channel index
        assert stream._lsl_tr_channel == 0

        # test acquire and bufsize/winsize
        time.sleep(1.1)
        stream.acquire()
        assert stream.lsl_time_offset is not None
        assert stream.buffer.bufsize == round(1 * stream.sample_rate) == 512
        assert stream.buffer.winsize == round(0.5 * stream.sample_rate) == 256
        assert len(stream.buffer.data) == len(stream.buffer.timestamps)
        assert len(stream.buffer.data) == stream.buffer.bufsize
