from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


ScalarFloatType: tuple[DTypeLike, ...] = (np.float32, np.float64)
ScalarIntType: tuple[DTypeLike, ...] = (np.int8, np.int16, np.int32, np.int64)
ScalarType: tuple[DTypeLike, ...] = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
)
