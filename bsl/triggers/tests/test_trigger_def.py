from bsl import logger, set_log_level
from bsl.triggers import TriggerDef


set_log_level('INFO')
logger.propagate = True


def _check_pair(tdef, name, value):
    """Check that the pair name, value is correctly set in tdef."""
    assert name in tdef._by_name
    assert value in tdef._by_value
    assert tdef._by_name[name] == value
    assert tdef._by_value[value] == name
    assert hasattr(tdef, name)
    assert getattr(tdef, name) == value


def test_trigger_def(caplog):
    """Test trigger def class."""
    # Test with trigger_file = None
    tdef = TriggerDef()
    assert tdef._by_name == tdef.by_name == {}
    assert tdef._by_value == tdef.by_value == {}

    # Test add_event
    tdef.add_event('stim', 1, overwrite=False)
    _check_pair(tdef, 'stim', 1)

    # Test add existing event (name)
    tdef.add_event('stim', 2, overwrite=False)
    _check_pair(tdef, 'stim', 1)
    assert 2 not in tdef._by_value
    assert 'Event name stim already exists.' in caplog.text

    # Test add existing event (name - overwrite)
    tdef.add_event('stim', 2, overwrite=True)
    _check_pair(tdef, 'stim', 2)
    assert 1 not in tdef._by_value

    # Test add existing event (value)
    tdef.add_event('rest', 2, overwrite=False)
    _check_pair(tdef, 'stim', 2)
    assert 'rest' not in tdef._by_name
    assert 'Event value 2 already exists.' in caplog.text

    # Test add existing event (value - overwrite)
    tdef.add_event('rest', 2, overwrite=True)
    _check_pair(tdef, 'rest', 2)
    assert 'stim' not in tdef._by_name

    # Test remove non-existing event (name)
    tdef.remove_event('stim')
    assert 'Event name stim not found.' in caplog.text

    # Test remove non-existing event (value)
    tdef.remove_event(5)
    assert 'Event value 5 not found.' in caplog.text

    # Test remove existing event (name)
    tdef.remove_event('rest')
    assert 'rest' not in tdef._by_name
    assert 2 not in tdef._by_value
    assert not hasattr(tdef, 'rest')

    # Test remove existing event (value)
    tdef.add_event('rest', 2)
    tdef.remove_event(2)
    assert 'rest' not in tdef._by_name
    assert 2 not in tdef._by_value
    assert not hasattr(tdef, 'rest')

    # TODO: Test with file and test _extract_from_ini.
    # Add example .ini file to datasets.
    # Add example using trigger def and .ini file.
