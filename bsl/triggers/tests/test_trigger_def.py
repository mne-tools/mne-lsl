from bsl.triggers import TriggerDef


def test_trigger_def():
    """Test trigger def class."""
    # Test with trigger_file = None
    tdef = TriggerDef()
    assert tdef._by_name == tdef.by_name == {}
    assert tdef._by_value == tdef.by_value == {}

    # Test add_evemt
    tdef.add_event('stim', 1, overwrite=False)
    assert 'stim' in tdef._by_name
    assert tdef._by_name['stim'] == 1
    assert 1 in tdef._by_value
    assert tdef.stim == 1
    tdef.add_event('stim', 1, overwrite=False)
    assert 'stim' in tdef._by_name
    assert tdef._by_name['stim'] == 1
    assert 1 in tdef._by_value
    assert tdef.stim == 1
    tdef.add_event('stim', 2, overwrite=False)
    assert 'stim' in tdef._by_name
    assert tdef._by_name['stim'] == 1
    assert 1 in tdef._by_value
    assert tdef.stim == 1
    tdef.add_event('stim', 2, overwrite=True)
    assert 'stim' in tdef._by_name
    assert tdef._by_name['stim'] == 2
    assert 2 in tdef._by_value
    assert tdef.stim == 2

    # TODO: Test with file and test _extract_from_ini.
    # Add example .ini file to datasets.
    # Add example using trigger def and .ini file.
