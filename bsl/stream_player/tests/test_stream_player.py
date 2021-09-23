import time
import pylsl

from bsl import StreamPlayer, logger, set_log_level
from bsl.datasets import sample, event
from bsl.utils._testing import requires_sample_dataset, requires_event_dataset


set_log_level('INFO')
logger.propagate = True


@requires_event_dataset
@requires_sample_dataset
def test_stream_player(caplog):
    """Test stream player capabilities."""
    sp = StreamPlayer('StreamPlayer', sample.data_path())
    sp.start(repeat=float('inf'), high_resolution=False)
    time.sleep(2)
    streams = pylsl.resolve_streams()
    assert sp.stream_name in [stream.name() for stream in streams]
    sp.stop()
    assert sp.process is None
    assert 'Stop streaming data' in caplog.text

    # Test properties
    sp.stream_name = 'StreamPlayer2'
    assert sp.stream_name == 'StreamPlayer2'

    sp.fif_file = sample.data_path()
    assert sp.fif_file == sample.data_path()

    sp.chunk_size = 32
    assert sp.chunk_size == 32

    sp.trigger_file = event.data_path()
    assert sp.trigger_file == event.data_path()

    # Test property setters
    sp.start(high_resolution=False)
    sp.stream_name = 'StreamPlayer'
    assert 'Stop the stream before changing the name' in caplog.text
    sp.fif_file = sample.data_path()
    assert 'Stop the stream before changing the FIF file' in caplog.text
    sp.chunk_size = 16
    assert 'Stop the stream before changing the chunk size' in caplog.text
    sp.trigger_file = None
    assert 'Stop the stream before changing the trigger file' in caplog.text
    sp.stop()

    # Test high_resolution
    sp.start(high_resolution=True)
    time.sleep(2)
    sp.stop()
