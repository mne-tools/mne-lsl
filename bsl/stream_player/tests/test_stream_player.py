import multiprocessing as mp
import time

import mne
import pytest

from bsl import StreamPlayer, logger, set_log_level
from bsl.datasets import (
    eeg_auditory_stimuli,
    eeg_resting_state,
    eeg_resting_state_short,
    trigger_def,
)
from bsl.lsl import StreamInlet, resolve_streams
from bsl.triggers import TriggerDef
from bsl.utils._tests import (
    requires_eeg_auditory_stimuli_dataset,
    requires_eeg_resting_state_dataset,
    requires_eeg_resting_state_short_dataset,
    requires_trigger_def_dataset,
)

set_log_level("INFO")
logger.propagate = True


@requires_eeg_resting_state_dataset
def test_stream_player(caplog):
    """Test stream player default capabilities."""
    stream_name = "StreamPlayer"
    fif_file = eeg_resting_state.data_path()
    raw = mne.io.read_raw_fif(fif_file, preload=True)

    # Test start
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file)
    sp.start()
    assert "Streaming started." in caplog.text

    # Test stream is in resolved streams
    streams = resolve_streams()
    assert stream_name in [stream.name for stream in streams]

    # Test that data is being streamed
    idx = [stream.name for stream in streams].index(stream_name)
    inlet = StreamInlet(streams[idx], max_buffered=int(raw.info["sfreq"]))
    inlet.open_stream()
    time.sleep(0.1)
    chunk, tslist = inlet.pull_chunk(
        timeout=0.0, max_samples=int(raw.info["sfreq"])
    )
    assert len(chunk) == len(tslist)
    assert 0 < len(chunk) < int(raw.info["sfreq"])
    time.sleep(1)
    chunk, tslist = inlet.pull_chunk(
        timeout=0.0, max_samples=int(raw.info["sfreq"])
    )
    assert len(chunk) == len(tslist) == int(raw.info["sfreq"])

    # Test stop
    caplog.clear()
    sp.stop()
    assert (
        "Waiting for StreamPlayer %s process to finish." % "StreamPlayer"
    ) in caplog.text
    assert "Streaming finished." in caplog.text
    assert sp.process is None

    # Test restart/stop
    caplog.clear()
    sp.start()
    assert "Streaming started." in caplog.text
    streams = resolve_streams()
    assert sp.stream_name in [stream.name for stream in streams]
    sp.stop()
    assert (
        "Waiting for StreamPlayer %s process to finish." % "StreamPlayer"
    ) in caplog.text
    assert sp.process is None

    # Test stop when not started
    caplog.clear()
    sp.stop()
    assert "StreamPlayer was not started. Skipping." in caplog.text

    # Test context manager
    caplog.clear()
    with StreamPlayer(stream_name=stream_name, fif_file=fif_file):
        assert "Streaming started." in caplog.text
        streams = resolve_streams()
        assert stream_name in [stream.name for stream in streams]
        time.sleep(0.5)
    assert (
        "Waiting for StreamPlayer %s process to finish." % "StreamPlayer"
    ) in caplog.text
    assert "Streaming finished." in caplog.text


@requires_eeg_resting_state_short_dataset
def test_arg_repeat():
    """Test stream player replay capabilities."""
    sp = StreamPlayer(
        stream_name="StreamPlayer",
        fif_file=eeg_resting_state_short.data_path(),
        repeat=2,
    )
    sp.start()
    assert sp.state.value == 1
    time.sleep(2.1)
    assert sp.state.value == 1
    time.sleep(2.1)
    assert sp.state.value == 0


@requires_eeg_auditory_stimuli_dataset
def test_arg_trigger_def():
    """Test stream player trigger display capabilities."""
    stream_name = "StreamPlayer"
    fif_file = eeg_auditory_stimuli.data_path()

    # Without TriggerDef
    sp = StreamPlayer(
        stream_name=stream_name, fif_file=fif_file, trigger_def=None
    )
    sp.start()
    time.sleep(2)
    sp.stop()

    # With created TriggerDef
    trigger_def = TriggerDef()
    trigger_def.add("rest", 1)
    trigger_def.add("stim", 4)
    sp = StreamPlayer(
        stream_name=stream_name, fif_file=fif_file, trigger_def=trigger_def
    )
    sp.start()
    time.sleep(2)
    sp.stop()

    # With created TriggerDef missing an event
    trigger_def = TriggerDef()
    trigger_def.add("rest", 1)
    sp = StreamPlayer(
        stream_name=stream_name, fif_file=fif_file, trigger_def=trigger_def
    )
    sp.start()
    time.sleep(2)
    sp.stop()


@requires_eeg_resting_state_dataset
def test_arg_high_resolution():
    """Test stream player high-resolution capabilities."""
    sp = StreamPlayer(
        stream_name="StreamPlayer",
        fif_file=eeg_resting_state.data_path(),
        high_resolution=True,
    )
    assert sp.high_resolution is True
    sp.start()
    assert sp.high_resolution is True
    time.sleep(0.1)
    sp.stop()
    assert sp.high_resolution is True


@requires_eeg_resting_state_dataset
def test_properties():
    """Test the StreamPlayer properties."""
    sp = StreamPlayer(
        stream_name="StreamPlayer",
        fif_file=eeg_resting_state.data_path(),
        repeat=float("inf"),
        trigger_def=None,
        chunk_size=16,
        high_resolution=False,
    )

    # Check getters
    assert sp.stream_name == "StreamPlayer"
    assert sp.fif_file == eeg_resting_state.data_path()
    assert sp.repeat == float("inf")
    assert sp.trigger_def is None
    assert sp.chunk_size == 16
    assert sp.high_resolution is False

    # Check setters
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.stream_name = "new name"
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.fif_file = "new fif file"
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.repeat = 10
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.trigger_def = TriggerDef()
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.chunk_size = 8
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.high_resolution = True

    # Process and state
    assert sp.process is None
    assert isinstance(sp.state, mp.sharedctypes.Synchronized)
    assert sp.state.value == 0

    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.process = 5
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.state = 5

    sp.start()
    assert isinstance(sp.process, mp.context.Process)
    assert sp.state.value == 1

    sp.stop()
    assert sp.process is None
    assert sp.state.value == 0


@requires_eeg_resting_state_dataset
def test_checker_arguments():
    """Test the argument error checking."""
    stream_name = "StreamPlayer"
    fif_file = eeg_resting_state.data_path()

    with pytest.raises(
        TypeError, match="'high_resolution' must be an instance"
    ):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, high_resolution=1
        )


def test_checker_fif_file():
    """Test the checker for argument fif file."""
    stream_name = "StreamPlayer"

    with pytest.raises(TypeError, match="'5' is invalid"):
        StreamPlayer(stream_name=stream_name, fif_file=5)
    with pytest.raises(FileNotFoundError, match="path '101-file' does not"):
        StreamPlayer(stream_name=stream_name, fif_file="101-file")


@requires_eeg_resting_state_dataset
def test_checker_repeat():
    """Test the checker for argument repeat."""
    stream_name = "StreamPlayer"
    fif_file = eeg_resting_state.data_path()

    # Default
    sp = StreamPlayer(
        stream_name=stream_name, fif_file=fif_file, repeat=float("inf")
    )
    assert sp.repeat == float("inf")

    # Positive integer
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file, repeat=101)
    assert sp.repeat == 101

    # Positive float
    with pytest.raises(TypeError, match="'repeat' must be an integer"):
        StreamPlayer(stream_name=stream_name, fif_file=fif_file, repeat=101.0)

    # Negative integer
    with pytest.raises(
        ValueError,
        match="Argument repeat must be a strictly "
        "positive integer. Provided: -101",
    ):
        sp = StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, repeat=-101
        )

    # Not integer
    with pytest.raises(TypeError, match="'repeat' must be an integer"):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, repeat=[1, 0, 1]
        )


@requires_trigger_def_dataset
@requires_eeg_resting_state_dataset
def test_checker_trigger_def():
    """Test the checker for argument trigger_def."""
    stream_name = "StreamPlayer"
    fif_file = eeg_resting_state.data_path()
    trigger_file = trigger_def.data_path()

    # Default
    sp = StreamPlayer(
        stream_name=stream_name, fif_file=fif_file, trigger_def=None
    )
    assert sp.trigger_def is None

    # TriggerDef instance
    sp = StreamPlayer(
        stream_name=stream_name,
        fif_file=fif_file,
        trigger_def=TriggerDef(trigger_file=None),
    )
    assert isinstance(sp.trigger_def, TriggerDef)
    sp = StreamPlayer(
        stream_name=stream_name,
        fif_file=fif_file,
        trigger_def=TriggerDef(trigger_file=trigger_file),
    )
    assert isinstance(sp.trigger_def, TriggerDef)

    # Path to a non-existing file
    with pytest.raises(
        FileNotFoundError,
        match="'101-path' does not exist.",
    ):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, trigger_def="101-path"
        )

    # Invalid type
    with pytest.raises(TypeError, match="'101' is invalid"):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, trigger_def=101
        )


@requires_eeg_resting_state_dataset
def test_checker_chunk_size(caplog):
    """Test the checker for argument chunk_size."""
    stream_name = "StreamPlayer"
    fif_file = eeg_resting_state.data_path()

    # Default
    sp = StreamPlayer(
        stream_name=stream_name, fif_file=fif_file, chunk_size=16
    )
    assert sp.chunk_size == 16

    # Positive integer
    sp = StreamPlayer(
        stream_name=stream_name, fif_file=fif_file, chunk_size=32
    )
    assert sp.chunk_size == 32

    # Positive float
    with pytest.raises(TypeError, match="'chunk_size' must be an integer"):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, chunk_size=32.0
        )

    # Positive non-usual integer
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file, chunk_size=8)
    assert (
        "The chunk size %s is different from the usual " "values 16 or 32." % 8
    ) in caplog.text
    assert sp.chunk_size == 8

    # Negative number
    with pytest.raises(
        ValueError,
        match="Argument chunk_size must be a strictly positive "
        "integer. Provided: -101",
    ):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, chunk_size=-101
        )

    # Invalid type
    with pytest.raises(TypeError, match="'chunk_size' must be an integer"):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, chunk_size=[1, 0, 1]
        )

    # Infinite
    with pytest.raises(TypeError, match="'chunk_size' must be an integer"):
        StreamPlayer(
            stream_name=stream_name, fif_file=fif_file, chunk_size=float("inf")
        )


@requires_eeg_resting_state_dataset
def test_representation():
    """Test the representation method."""
    fif_file = eeg_resting_state.data_path()
    sp = StreamPlayer(stream_name="StreamPlayer", fif_file=fif_file)
    expected = f"<Player: StreamPlayer | OFF | {fif_file}>"
    assert sp.__repr__() == expected
    sp.start()
    expected = f"<Player: StreamPlayer | ON | {fif_file}>"
    assert sp.__repr__() == expected
    sp.stop()
    expected = f"<Player: StreamPlayer | OFF | {fif_file}>"
    assert sp.__repr__() == expected
