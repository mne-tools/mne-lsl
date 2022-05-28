"""Base class for triggers."""

from abc import ABC, abstractmethod

from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils._logs import logger


@fill_doc
class _Trigger(ABC):
    """Base trigger class.

    Parameters
    ----------
    %(trigger_verbose)s
    """

    @abstractmethod
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def signal(self, value: int):
        """Send a trigger value.

        Parameters
        ----------
        value : int
            Value of the trigger.
        """
        if self._verbose:
            logger.info("Sending trigger %s.", value)
        else:
            logger.debug("Sending trigger %s.", value)

    @abstractmethod
    def _set_data(self, value: int):
        """Set the trigger signal to value."""
        logger.debug("Setting data to %d.", value)

    # --------------------------------------------------------------------
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        _check_type(verbose, (bool,), item_name="verbose")
        self._verbose = verbose
