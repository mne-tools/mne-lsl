import pytest

from bsl import logger, set_log_level
from bsl.triggers import MockTrigger

set_log_level("INFO")
logger.propagate = True


def test_trigger_mock(caplog):
    """Testing for Mock triggers."""
    trigger = MockTrigger(verbose=True)
    assert trigger.verbose
    trigger.verbose = False
    assert trigger.signal(1)
    assert "MOCK trigger set to 1" in caplog.text
    assert trigger.signal(2)
    assert "MOCK trigger set to 2" in caplog.text

    with pytest.raises(TypeError, match="'value' must be an instance"):
        trigger.signal(3.0)
    assert "MOCK trigger set to 3.0" not in caplog.text
