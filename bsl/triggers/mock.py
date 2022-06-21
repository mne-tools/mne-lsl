"""Mock trigger."""

from ..utils._docs import copy_doc, fill_doc
from ..utils._logs import logger
from ._base import BaseTrigger


@fill_doc
class MockTrigger(BaseTrigger):
    """Mock trigger class.

    Parameters
    ----------
    %(trigger_verbose)s
    """

    def __init__(self, *, verbose: bool = True):
        super().__init__(verbose)

    @copy_doc(BaseTrigger.signal)
    def signal(self, value: int) -> None:
        super().signal(value)
        self._set_data(value)

    def _signal_off(self) -> None:
        """Reset trigger signal to ``0``."""
        self._set_data(0)

    @copy_doc(BaseTrigger._set_data)
    def _set_data(self, value: int) -> None:
        super()._set_data(value)
        logger.info("MOCK trigger set to %i", value)
