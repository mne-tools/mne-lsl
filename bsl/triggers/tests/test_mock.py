import pytest

from bsl import logger, set_log_level
from bsl.triggers import MockTrigger

set_log_level("INFO")
logger.propagate = True


def test_trigger_mock(caplog):
    """Testing for Mock triggers."""
    trigger = MockTrigger()
    trigger.signal(1)
    assert "Mock set to 1" in caplog.text
    trigger.signal(2)
    assert "Mock set to 2" in caplog.text

    with pytest.raises(ValueError, match="between 1 and 127"):
        trigger.signal(128)
    assert "Mock set to 3" not in caplog.text
