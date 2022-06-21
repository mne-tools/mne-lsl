"""Mock trigger."""

from ..utils._checks import _check_type
from ..utils._docs import copy_doc, fill_doc
from ..utils._logs import logger
from ._trigger import _Trigger


@fill_doc
class MockTrigger(_Trigger):
    """Mock trigger class.

    Parameters
    ----------
    %(trigger_verbose)s
    """

    def __init__(self, *, verbose: bool = True):
        super().__init__(verbose)

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> None:
        super().signal(value)
        self._set_data(value)

    def _signal_off(self) -> None:
        """Reset trigger signal to ``0``."""
        self._set_data(0)

    @copy_doc(_Trigger._set_data)
    def _set_data(self, value: int) -> None:
        super()._set_data(value)
        logger.info("MOCK trigger set to %i", value)
