"""
Mock trigger.
"""
from ._trigger import _Trigger
from .. import logger


class TriggerMock(_Trigger):
    """
    Mock trigger class.

    Parameters
    ----------
    verbose : bool
        If True, display a logger.info message when a trigger is sent.
    """
    def __init__(self, verbose=True):
        super().__init__(verbose)

    def signal(self, value):
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
        super()._signal_off()

    def _set_data(self, value):
        """
        Set the trigger signal to value.
        """
        logger.info(f'MOCK trigger set to {value}')
