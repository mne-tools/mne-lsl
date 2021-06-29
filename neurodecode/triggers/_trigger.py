"""
Base class for triggers.
"""
from abc import ABC, abstractmethod

from .. import logger


class _Trigger(ABC):
    """
    Base trigger class.

    Parameters
    ----------
    verbose : bool
        If True, display a logger.info message when a trigger is sent.
    """

    @abstractmethod
    def __init__(self, verbose=True):
        self.verbose = bool(verbose)

    @abstractmethod
    def signal(self, value):
        """
        Send a trigger value.
        """
        if self.verbose:
            logger.info(f'Sending trigger {value}.')

    @abstractmethod
    def _signal_off(self):
        """
        Reset trigger signal to 0.
        """
        self._set_data(0)

    @abstractmethod
    def _set_data(self, value):
        """
        Set the trigger signal to value.
        """
        pass
