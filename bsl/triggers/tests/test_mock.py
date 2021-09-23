from bsl import logger, set_log_level
from bsl.triggers.mock import TriggerMock


set_log_level('INFO')
logger.propagate = True


def test_trigger_mock(caplog):
    """Testing for Mock triggers."""
    trigger = TriggerMock(verbose=True)
    assert trigger.verbose
    trigger.verbose = False
    assert trigger.signal(1)
    assert 'MOCK trigger set to 1' in caplog.text
    assert trigger.signal(2)
    assert 'MOCK trigger set to 2' in caplog.text
