# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import numpy as np

from bsl.lsl.constants import fmt2numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bsl.lsl import StreamInlet


class Buffer:
    def __init_(self, inlet: StreamInlet, bufsize: float):
        # the buffer shape is similar to a pull_sample/pull_chunk from an inlet:
        # (n_samples, n_channels).
        self._buffer = np.zeros(
            ceil(bufsize * inlet.sfreq), inlet.n_channels, dtype=fmt2numpy[inlet._dtype]
        )
        # count the number of samples that go through the buffer
        self._samples_retrieved = 0

    def update(self, data: NDArray[float]):
        # data is a pull from an inlet, thus the shape is (n_samples, n_channels)
        self._buffer = np.roll(self._buffer, -data.shape[0], axis=0)
        self._buffer[-data.shape[0]:, :] = data
        self._samples_retrieved += data.shape[0]

    @property
    def samples_retrieved(self) -> int:
        return self._samples_retrieved
