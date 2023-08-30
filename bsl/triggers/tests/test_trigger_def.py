from configparser import ConfigParser

import pytest

from bsl import logger, set_log_level
from bsl.datasets import eeg_resting_state, trigger_def
from bsl.triggers import TriggerDef

set_log_level("INFO")
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
    """Test default behavior."""
    # Test with trigger_file = None
    tdef = TriggerDef()
    assert tdef._by_name == tdef.by_name == {}
    assert tdef._by_value == tdef.by_value == {}

    # Test add_event
    tdef.add("stim", 1, overwrite=False)
    _check_pair(tdef, "stim", 1)

    # Test add existing event (name)
    caplog.clear()
    tdef.add("stim", 2, overwrite=False)
    _check_pair(tdef, "stim", 1)
    assert 2 not in tdef._by_value
    assert "Event name stim already exists. Skipping." in caplog.text

    # Test add existing event (name - overwrite)
    tdef.add("stim", 2, overwrite=True)
    _check_pair(tdef, "stim", 2)
    assert 1 not in tdef._by_value

    # Test add existing event (value)
    caplog.clear()
    tdef.add("rest", 2, overwrite=False)
    _check_pair(tdef, "stim", 2)
    assert "rest" not in tdef._by_name
    assert "Event value 2 already exists. Skipping." in caplog.text

    # Test add existing event (value - overwrite)
    tdef.add("rest", 2, overwrite=True)
    _check_pair(tdef, "rest", 2)
    assert "stim" not in tdef._by_name

    # Test remove non-existing event (name)
    caplog.clear()
    tdef.remove("stim")
    assert "Event name stim not found." in caplog.text

    # Test remove non-existing event (value)
    caplog.clear()
    tdef.remove(5)
    assert "Event value 5 not found." in caplog.text

    # Test remove existing event (name)
    tdef.remove("rest")
    assert "rest" not in tdef._by_name
    assert 2 not in tdef._by_value
    assert not hasattr(tdef, "rest")

    # Test remove existing event (value)
    tdef.add("rest", 2)
    tdef.remove(2)
    assert "rest" not in tdef._by_name
    assert 2 not in tdef._by_value
    assert not hasattr(tdef, "rest")


def test_read_ini(caplog, tmp_path):
    """Test reading from a .ini file."""
    # Valid file
    tdef = TriggerDef(trigger_def.data_path())
    assert tdef._by_name == {
        "rest": 1,
        "stim_left": 2,
        "stim_right": 3,
        "feedback": 4,
        "start": 5,
        "stop": 6,
    }
    assert tdef._by_value == {
        1: "rest",
        2: "stim_left",
        3: "stim_right",
        4: "feedback",
        5: "start",
        6: "stop",
    }

    # Valid .ini file with duplicate values
    caplog.clear()
    config = ConfigParser()
    config["events"] = {"stim": 1, "rest": 1}
    with open(tmp_path / "test.ini", "w") as configfile:
        config.write(configfile)
    tdef = TriggerDef(tmp_path / "test.ini")
    assert "Event value %s already exists. Skipping." % 1 in caplog.text
    assert tdef._by_name == {"stim": 1}
    assert tdef._by_value == {1: "stim"}

    # Invalid file
    with pytest.raises(TypeError, match="'101' is invalid"):
        tdef = TriggerDef(101)

    caplog.clear()
    tdef = TriggerDef(eeg_resting_state.data_path())
    assert tdef._trigger_file is None
    assert (
        "Argument trigger_file must be a valid Path to a .ini file. " "Provided: .fif"
    ) in caplog.text

    caplog.clear()
    tdef = TriggerDef("non-existing-path")
    assert tdef._trigger_file is None
    assert (
        "Trigger event definition file '%s' not found." % "non-existing-path"
    ) in caplog.text


def test_write_ini(tmp_path):
    """Test write to a .ini file."""
    # Valid file
    tdef = TriggerDef()
    tdef.add("stim", 1, overwrite=False)
    tdef.add("rest", 2, overwrite=False)
    tdef.write(tmp_path / "test_write.ini")

    # Invalid file
    trigger_file = 101
    with pytest.raises(TypeError, match="'101' is invalid"):
        tdef.write(trigger_file)

    trigger_file = tmp_path / "test_write.txt"
    with pytest.raises(
        ValueError,
        match="Argument trigger_file must end with .ini. "
        "Provided: %s" % trigger_file.suffix,
    ):
        tdef.write(trigger_file)


def test_properties():
    """Test the properties."""
    tdef = TriggerDef(trigger_def.data_path())
    assert (
        tdef._by_name
        == tdef.by_name
        == {
            "rest": 1,
            "stim_left": 2,
            "stim_right": 3,
            "feedback": 4,
            "start": 5,
            "stop": 6,
        }
    )
    assert (
        tdef._by_value
        == tdef.by_value
        == {
            1: "rest",
            2: "stim_left",
            3: "stim_right",
            4: "feedback",
            5: "start",
            6: "stop",
        }
    )

    with pytest.raises(AttributeError):
        tdef.by_name = dict()
    with pytest.raises(AttributeError):
        tdef.by_value = dict()
