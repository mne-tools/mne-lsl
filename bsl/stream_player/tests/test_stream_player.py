import time
from pathlib import Path
import multiprocessing as mp

import mne
import pylsl
import pytest

from bsl import StreamPlayer, logger, set_log_level
from bsl.datasets import (eeg_auditory_stimuli, eeg_resting_state,
                          eeg_resting_state_short, trigger_def)
from bsl.triggers import TriggerDef
from bsl.utils._testing import (requires_eeg_auditory_stimuli_dataset,
                                requires_eeg_resting_state_dataset,
                                requires_eeg_resting_state_short_dataset,
                                requires_trigger_def_dataset)


set_log_level('INFO')
logger.propagate = True


@requires_eeg_resting_state_dataset
def test_stream_player(caplog):
    """Test stream player default capabilities."""
    stream_name = 'StreamPlayer'
    fif_file = eeg_resting_state.data_path()
    raw = mne.io.read_raw_fif(fif_file, preload=True)

    # Test start
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file)
    sp.start()
    assert 'Streaming started.' in caplog.text

    # Test stream is in resolved streams
    streams = pylsl.resolve_streams()
    assert stream_name in [stream.name() for stream in streams]

    # Test that data is being streamed
    idx = [stream.name() for stream in streams].index(stream_name)
    inlet = pylsl.StreamInlet(streams[idx], max_buflen=int(raw.info['sfreq']))
    inlet.open_stream()
    time.sleep(0.05)
    chunk, tslist = inlet.pull_chunk(
        timeout=0.0, max_samples=int(raw.info['sfreq']))
    assert len(chunk) == len(tslist)
    assert 0 < len(chunk) < int(raw.info['sfreq'])
    time.sleep(1)
    chunk, tslist = inlet.pull_chunk(
        timeout=0.0, max_samples=int(raw.info['sfreq']))
    assert len(chunk) == len(tslist) ==  int(raw.info['sfreq'])

    # Test stop
    caplog.clear()
    sp.stop()
    assert ('Waiting for StreamPlayer %s process to finish.'
            % 'StreamPlayer') in caplog.text
    assert 'Streaming finished.' in caplog.text
    assert sp.process is None

    # Test restart/stop
    caplog.clear()
    sp.start()
    assert 'Streaming started.' in caplog.text
    streams = pylsl.resolve_streams()
    assert sp.stream_name in [stream.name() for stream in streams]
    sp.stop()
    assert ('Waiting for StreamPlayer %s process to finish.'
            % 'StreamPlayer') in caplog.text
    assert sp.process is None

    # Test stop when not started
    caplog.clear()
    sp.stop()
    assert 'StreamPlayer was not started. Skipping.' in caplog.text

    # Test context manager
    caplog.clear()
    with StreamPlayer(stream_name=stream_name, fif_file=fif_file):
        assert 'Streaming started.' in caplog.text
        streams = pylsl.resolve_streams()
        assert stream_name in [stream.name() for stream in streams]
        time.sleep(0.5)
    assert ('Waiting for StreamPlayer %s process to finish.'
            % 'StreamPlayer') in caplog.text
    assert 'Streaming finished.' in caplog.text


@requires_eeg_resting_state_short_dataset
def test_arg_repeat():
    """Test stream player replay capabilities."""
    sp = StreamPlayer(stream_name='StreamPlayer',
                      fif_file=eeg_resting_state_short.data_path(),
                      repeat=2)
    sp.start()
    assert sp.state.value == 1
    time.sleep(2.1)
    assert sp.state.value == 1
    time.sleep(2.1)
    assert sp.state.value == 0


@requires_eeg_auditory_stimuli_dataset
def test_arg_trigger_def():
    """Test stream player trigger display capabilities."""
    stream_name = 'StreamPlayer'
    fif_file = eeg_auditory_stimuli.data_path()

    # Without TriggerDef
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=None)
    sp.start()
    time.sleep(2)
    sp.stop()

    # With created TriggerDef
    trigger_def = TriggerDef()
    trigger_def.add_event('rest', 1)
    trigger_def.add_event('stim', 4)
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=trigger_def)
    sp.start()
    time.sleep(2)
    sp.stop()

    # With creatred TriggerDef missing an event
    trigger_def = TriggerDef()
    trigger_def.add_event('rest', 1)
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=trigger_def)
    sp.start()
    time.sleep(2)
    sp.stop()


@requires_eeg_resting_state_dataset
def test_arg_high_resolution():
    """Test stream player high-resolution capabilities."""
    sp = StreamPlayer(stream_name='StreamPlayer',
                      fif_file=eeg_resting_state.data_path(),
                      high_resolution=True)
    assert sp.high_resolution is True
    sp.start()
    assert sp.high_resolution is True
    time.sleep(0.1)
    sp.stop()
    assert sp.high_resolution is True


@requires_eeg_resting_state_dataset
def test_properties():
    """Test the StreamPlayer properties."""
    sp = StreamPlayer(stream_name='StreamPlayer',
                      fif_file=eeg_resting_state.data_path(),
                      repeat=float('inf'),
                      trigger_def=None,
                      chunk_size=16,
                      high_resolution=False)

    # Check getters
    assert sp.stream_name == 'StreamPlayer'
    assert sp.fif_file == eeg_resting_state.data_path()
    assert sp.repeat == float('inf')
    assert sp.trigger_def is None
    assert sp.chunk_size == 16
    assert sp.high_resolution is False

    # Check setters
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.stream_name = 'new name'
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.fif_file = 'new fif file'
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


def test_invalid_fif_file():
    """Test that initialization fails if argument fif_file is invalid."""
    with pytest.raises(ValueError, match='Argument fif_file must be'):
        StreamPlayer(stream_name='StreamPlayer', fif_file='non-existing-path')
    with pytest.raises(ValueError, match='Argument fif_file must be'):
        StreamPlayer(stream_name='StreamPlayer', fif_file=5)


@requires_eeg_resting_state_dataset
def test_checker_repeat(caplog):
    """Test the checker for argument repeat."""
    stream_name = 'StreamPlayer'
    fif_file = eeg_resting_state.data_path()

    # Default
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      repeat=float('inf'))
    assert sp.repeat == float('inf')

    # Positive integer
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      repeat=5)
    assert sp.repeat == 5

    # Positive float
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      repeat=5.)
    assert isinstance(sp.repeat, int) and sp.repeat == 5

    # Negative number
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      repeat=-5)
    assert ('Argument repeat must be a strictly positive integer. '
            'Provided: %s -> Changing to +inf.' % -5) in caplog.text
    assert sp.repeat == float('inf')

    # Not convertible to integer
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      repeat=[1, 2])
    assert ('Argument repeat must be a strictly positive integer. '
            'Provided: %s -> Changing to +inf.' % str([1, 2])) in caplog.text
    assert sp.repeat == float('inf')


@requires_trigger_def_dataset
@requires_eeg_resting_state_dataset
def test_checker_trigger_def(caplog):
    """Test the checker for argument trigger_def."""
    stream_name = 'StreamPlayer'
    fif_file = eeg_resting_state.data_path()
    trigger_file = trigger_def.data_path()

    # Default
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=None)
    assert sp.trigger_def is None

    # TriggerDef instance
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=TriggerDef(trigger_file=None))
    assert isinstance(sp.trigger_def, TriggerDef)
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=TriggerDef(trigger_file=trigger_file))
    assert isinstance(sp.trigger_def, TriggerDef)

    # Path as a string
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=str(trigger_file))
    assert isinstance(sp.trigger_def, TriggerDef)

    # Path as a Path object
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=Path(trigger_file))
    assert isinstance(sp.trigger_def, TriggerDef)

    # Path to a non-existing file
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def='non-existing-path')
    assert ('Argument trigger_def is a path that does not exist. '
            'Provided: %s -> Ignoring.' % 'non-existing-path') in caplog.text
    assert sp.trigger_def is None

    # Invalid type
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      trigger_def=5)
    assert ('Argument trigger_def must be a TriggerDef instance or a path '
            'to a trigger definition ini file. '
            'Provided: %s -> Ignoring.' % type(5)) in caplog.text
    assert sp.trigger_def is None


@requires_eeg_resting_state_dataset
def test_checker_chunk_size(caplog):
    """Test the checker for argument chunk_size."""
    stream_name = 'StreamPlayer'
    fif_file = eeg_resting_state.data_path()

    # Default
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      chunk_size=16)
    assert sp.chunk_size == 16

    # Positive integer
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      chunk_size=32)
    assert sp.chunk_size == 32

    # Positive float
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      chunk_size=32.)
    assert isinstance(sp.chunk_size, int) and sp.chunk_size == 32

    # Positive non-usual integer
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      chunk_size=8)
    assert ('The chunk size %s is different from the usual '
            'values 16 or 32.' % 8) in caplog.text
    assert sp.chunk_size == 8

    # Negative number
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      chunk_size=-8)
    assert ('Argument chunk_size must be a strictly positive integer. '
            'Provided: %s -> Changing to 16.' % -8) in caplog.text
    assert sp.chunk_size == 16

    # Not convertible to integer
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      chunk_size=[1, 2])
    assert ('Argument chunk_size must be a strictly positive integer. '
            'Provided: %s -> Changing to 16.' % str([1, 2])) in caplog.text
    assert sp.chunk_size == 16

    # Infinite
    caplog.clear()
    sp = StreamPlayer(stream_name=stream_name, fif_file=fif_file,
                      chunk_size=float('inf'))
    assert ('Argument chunk_size must be a strictly positive integer. '
            'Provided: %s -> Changing to 16.' % float('inf')) in caplog.text
    assert sp.chunk_size == 16


@requires_eeg_resting_state_dataset
def test_representation():
    """Test the representation method."""
    sp = StreamPlayer(stream_name='StreamPlayer',
                      fif_file=eeg_resting_state.data_path())
    expected = f'<Player: StreamPlayer | OFF | {eeg_resting_state.data_path()}>'
    assert sp.__repr__() == expected
    sp.start()
    expected = f'<Player: StreamPlayer | ON | {eeg_resting_state.data_path()}>'
    assert sp.__repr__() == expected
    sp.stop()
    expected = f'<Player: StreamPlayer | OFF | {eeg_resting_state.data_path()}>'
    assert sp.__repr__() == expected
