import numpy as np

from bsl import logger, set_log_level
from bsl.lsl import StreamInlet, resolve_streams
from bsl.triggers import LSLTrigger

set_log_level("INFO")
logger.propagate = True


def test_trigger_lsl():
    """Testing for LSL triggers."""
    name = "test-trigger-lsl"
    trigger = LSLTrigger(name)
    streams = resolve_streams(name=name)
    assert len(streams) == 1
    sinfo = streams[0]
    del streams
    inlet = StreamInlet(sinfo)
    inlet.open_stream()
    assert inlet.samples_available == 0
    sinfo = inlet.get_sinfo()
    assert sinfo.get_channel_names() == ["STI"]
    assert sinfo.get_channel_types() == ["stim"]
    assert sinfo.get_channel_units() == ["none"]
    trigger.signal(1)
    data, ts = inlet.pull_sample(timeout=10)
    assert data.dtype == np.int8
    assert data.size == 1
    assert data[0] == 1
    assert ts is not None
    trigger.signal(127)
    data, ts = inlet.pull_sample(timeout=10)
    assert data.dtype == np.int8
    assert data.size == 1
    assert data[0] == 127
    assert ts is not None
