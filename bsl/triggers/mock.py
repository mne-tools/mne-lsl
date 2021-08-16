"""
Mock trigger.
"""
from ._trigger import _Trigger
from .. import logger
from ..utils._docs import fill_doc, copy_doc


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

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
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
        logger.info(f'MOCK trigger set to {value}')
