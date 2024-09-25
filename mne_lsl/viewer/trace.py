from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np
from pyqtgraph import PlotCurveItem

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .._typing import ScalarType


class DataTrace(PlotCurveItem):
    def __init__(
        self,
        data: NDArray[+ScalarType],
        duration: float,
        scaling: float,
        ypos: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._data = data  # array of shape (n_samples,)
        self._duration = duration
        self._scaling = scaling
        self._ypos = ypos
        self.update_data(update_times=True)
        self.update_ypos()

    def update_data(self, *, update_times: bool) -> None:
        if update_times:
            self._times = np.linspace(0, self._duration, self._data.size)
        self.setData(self._times, self._data * self._scaling)

    def update_ypos(self) -> None:
        self.setPos(0, self._ypos)

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, duration: float) -> None:
        self._duration = duration
        self.update_data(update_times=True)

    @property
    def scaling(self) -> float:
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: float) -> None:
        self._scaling = scaling
        self.update_data(update_times=False)

    @property
    def ypos(self) -> int:
        return self._ypos

    @ypos.setter
    def ypos(self, ypos: int) -> None:
        self._ypos = ypos
        self.update_ypos()
