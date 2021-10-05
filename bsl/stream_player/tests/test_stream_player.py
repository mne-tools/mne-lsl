from pathlib import Path

import pytest

from bsl import StreamPlayer, logger, set_log_level
from bsl.datasets import sample, event
from bsl.triggers import TriggerDef
from bsl.utils._testing import requires_sample_dataset, requires_event_dataset


set_log_level('INFO')
logger.propagate = True


@requires_sample_dataset
def test_stream_player_default(caplog):
    """Test stream player default capabilities."""
    pass


@requires_event_dataset
@requires_sample_dataset
def test_stream_player_trigger_def():
    """Test stream player capabilities with a trigger_def file provided."""
    pass


@requires_sample_dataset
def test_stream_player_properties():
    """Test the stream_player properties."""
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      repeat=float('inf'), trigger_def=None, chunk_size=16,
                      high_resolution=False)

    # Check getters
    assert sp.stream_name == 'StreamPlayer'
    assert sp.fif_file == sample.data_path()
    assert sp.repeat == float('inf')
    assert sp.trigger_def is None
    assert sp.chunk_size == 16
    assert sp.high_resolution is False

    # Check setters
    with pytest.raises(AttributeError, match="can't set attribute"):
        sp.stream_name = 'new name'
        sp.fif_file = 'new fif file'
        sp.repeat = 10
        sp.trigger_def = TriggerDef()
        sp.chunk_size = 8
        sp.high_resolution = True


def test_stream_player_invalid_fif_file():
    """Test that initialization fails if argument fif_file is invalid."""
    with pytest.raises(ValueError, match='Argument fif_file must be'):
        StreamPlayer(stream_name='StreamPlayer', fif_file='non-existing-path')
        StreamPlayer(stream_name='StreamPlayer', fif_file=5)


@requires_sample_dataset
def test_stream_player_checker_repeat(caplog):
    """Test the checker for argument repeat."""
    # Default
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      repeat=float('inf'))
    assert sp.repeat == float('inf')

    # Positive integer
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      repeat=5)
    assert sp.repeat == 5

    # Positive float
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      repeat=5.)
    assert isinstance(sp.repeat, int) and sp.repeat == 5

    # Negative number
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      repeat=-5)
    assert ('Argument repeat must be a strictly positive integer. '
            'Provided: %s -> Changing to +inf.' % -5) in caplog.text
    assert sp.repeat == float('inf')


@requires_event_dataset
@requires_sample_dataset
def test_stream_player_checker_trigger_def(caplog):
    """Test the checker for argument repeat."""
    # Default
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      trigger_def=None)
    assert sp.trigger_def is None

    # TriggerDef instance
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      trigger_def=TriggerDef(trigger_file=None))
    assert isinstance(sp.trigger_def, TriggerDef)
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      trigger_def=TriggerDef(trigger_file=event.data_path()))
    assert isinstance(sp.trigger_def, TriggerDef)

    # Path as a string
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      trigger_def=str(event.data_path()))
    assert isinstance(sp.trigger_def, TriggerDef)

    # Path as a Path object
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      trigger_def=Path(event.data_path()))
    assert isinstance(sp.trigger_def, TriggerDef)

    # Path to a non-existing file
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      trigger_def='non-existing-path')
    assert ('Argument trigger_def is a path that does not exist. '
            'Provided: %s -> Ignoring.' % 'non-existing-path') in caplog.text
    assert sp.trigger_def is None

    # Invalid type
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      trigger_def=5)
    assert ('Argument trigger_def was not a TriggerDef instance or a path '
            'to a trigger definition ini file. '
            'Provided: %s -> Ignoring.' % type(5)) in caplog.text
    assert sp.trigger_def is None

@requires_sample_dataset
def test_stream_player_checker_chunk_size(caplog):
    """Test the checker for argument repeat."""
    # Default
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      chunk_size=16)
    assert sp.chunk_size == 16

    # Positive integer
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      chunk_size=32)
    assert sp.chunk_size == 32

    # Positive float
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      chunk_size=32.)
    assert isinstance(sp.chunk_size, int) and sp.chunk_size == 32

    # Positive non-usual integer
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      chunk_size=8)
    assert ('The chunk size %s is different from the usual '
            'values 16 or 32.' % 8) in caplog.text
    assert sp.chunk_size == 8

    # Negative number
    sp = StreamPlayer(stream_name='StreamPlayer', fif_file=sample.data_path(),
                      chunk_size=-8)
    assert ('Argument chunk_size must be a strictly positive integer. '
            'Provided: %s -> Changing to 16.' % -8) in caplog.text
    assert sp.chunk_size == 16
