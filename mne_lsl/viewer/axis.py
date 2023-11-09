from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from pyqtgraph import AxisItem

if TYPE_CHECKING:
    from typing import List


class TimeAxis(AxisItem):
    """Time axis, from 1 to 60 seconds."""

    def __init__(self, duration: float) -> None:
        self._duration = duration
        super().__init__(orientation="bottom")
        # tick spacing and formatting
        self._major = 1.0
        self._minor = 0.25
        self.enableAutoSIPrefix(False)
        self.update_ticks()

    def update_ticks(self) -> None:
        super().setTicks(
            [
                [(elt, str(elt)) for elt in np.arange(self.duration, self.major)],
                [(elt, str(elt)) for elt in np.arange(self.duration, self.minor)],
            ]
        )

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        assert 1 <= duration <= 60, "'duration' should be between 1 and 60 seconds."
        self._duration = duration
        self.update_ticks()

    @property
    def major(self) -> float:
        """Spacing between major ticks."""
        return self._major

    @major.setter
    def major(self, major: float) -> float:
        assert 0 < major, "'major' should be a positive number."
        self._major = major
        self.update_ticks()

    @property
    def minor(self) -> float:
        """Spacing between minor ticks."""
        return self._minor

    @minor.setter
    def minor(self, minor: float) -> float:
        assert 0 < minor, "'minor' should be a positive number."
        self._minor = minor
        self.update_ticks()


class ChannelAxis(AxisItem):
    # TODO: The channel name color should depend on the channel type, thus the pen
    # should change depending on the channel type.

    def __init__(self, ch_names: List[str]) -> None:
        super().__init__(orientation="left")
        self._ch_names = ch_names
        self.enableAutoSIPrefix(False)

    def update_ticks(self) -> None:
        super().setTicks(
            [[(idx, ch_name) for idx, ch_name in enumerate(self.ch_names)], []]
        )

    @property
    def ch_names(self) -> List[str]:
        return self._ch_names

    @ch_names.setter
    def ch_names(self, ch_names: List[str]) -> None:
        self._ch_names = ch_names
        self.update_ticks()
