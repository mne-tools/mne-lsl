"""
Mock trigger.
"""
from ._trigger import _Trigger
from .. import logger
from ..utils.docs import fill_doc


@fill_doc
class TriggerMock(_Trigger):
    """
    Mock trigger class.

    Parameters
    ----------
    %(trigger_verbose)s
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)

    def signal(self, value: int) -> bool:
        """
        Send a trigger value.
        """
        self._set_data(value)
        super().signal(value)
        return True

    def _signal_off(self):
        """
        Reset trigger signal to 0.
        """
        self._set_data(0)

    def _set_data(self, value: int):
        """
        Set the trigger signal to value.
        """
        super()._set_data(value)
        logger.info(f'MOCK trigger set to {value}')
