"""
Mock trigger.
"""

from ._trigger import _Trigger
from ..utils._checks import _check_type
from ..utils._docs import fill_doc, copy_doc
from ..utils._logs import logger


@fill_doc
class MockTrigger(_Trigger):
    """
    Mock trigger class.

    Parameters
    ----------
    %(trigger_verbose)s
    """

    def __init__(self, *, verbose: bool = True):
        super().__init__(verbose)

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
        _check_type(value, ("int",), item_name="value")
        self._set_data(value)
        super().signal(value)
        return True

    def _signal_off(self):
        """
        Reset trigger signal to ``0``.
        """
        self._set_data(0)

    @copy_doc(_Trigger._set_data)
    def _set_data(self, value: int):
        super()._set_data(value)
        logger.info("MOCK trigger set to %i", value)
