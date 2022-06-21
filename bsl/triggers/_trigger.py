"""Base class for triggers."""

from abc import abstractmethod

from ..typing import Trigger
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils._logs import logger


@fill_doc
class _Trigger(Trigger):
    """Base trigger class.

    Parameters
    ----------
    %(trigger_verbose)s
    """

    @abstractmethod
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def signal(self, value: int) -> None:
        """Send a trigger value.

        Parameters
        ----------
        value : int
            Value of the trigger.
        """
        _check_type(value, ("int",), item_name="value")
        logger_func = logger.info if self._verbose else logger.debug
        logger_func("Sending trigger %s.", value)

    @abstractmethod
    def _set_data(self, value: int) -> None:
        """Set the trigger signal to value."""
        logger.debug("Setting data to %d.", value)

    # --------------------------------------------------------------------
    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        _check_type(verbose, (bool,), item_name="verbose")
        self._verbose = verbose
