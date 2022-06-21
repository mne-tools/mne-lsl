"""Typing with ABC class for BSL."""

from abc import ABC, abstractmethod


class Trigger(ABC):
    """Typing for a trigger instance."""

    @abstractmethod
    def __init__(self):
        pass
