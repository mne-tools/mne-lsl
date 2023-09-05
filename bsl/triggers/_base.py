"""Base class for triggers."""

from abc import ABC, abstractmethod


class BaseTrigger(ABC):
    """Base trigger class."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def signal(self, value: int) -> int:
        """Send a trigger value.

        Parameters
        ----------
        value : int
            Value of the trigger, between 1 and 255.
        """
        try:
            value = int(value)
        except TypeError:
            raise TypeError(
                "The argument 'value' of a BSL trigger must be an integer "
                "between 1 and 255 included."
            )
        if not (1 <= value <= 255):
            raise ValueError(
                "The argument 'value' of a BSL trigger must be an integer "
                "between 1 and 255 included."
            )
        return value
