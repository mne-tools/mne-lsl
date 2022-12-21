"""Mock trigger."""

from ..utils._docs import copy_doc
from ..utils._logs import logger
from ._base import BaseTrigger


class MockTrigger(BaseTrigger):
    """Mock trigger class.

    Delivered triggers are logged at the 'INFO' level.
    """

    def __init__(self):
        pass

    @copy_doc(BaseTrigger.signal)
    def signal(self, value: int) -> None:
        super().signal(value)
        logger.info("[Trigger] Mock set to %i.", value)
