from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Generic, TypeVar


ScalarFloatType = TypeVar("ScalarFloatType", np.float32, np.float64)
ScalarIntType = TypeVar("ScalarIntType", np.int8, np.int16, np.int32, np.int64)
ScalarType = TypeVar(
    "ScalarType", np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
)


class ScalarFloatArray(np.ndarray, Generic[ScalarFloatType]):
    pass


class ScalarIntArray(np.ndarray, Generic[ScalarIntType]):
    pass


class ScalarArray(np.ndarray, Generic[ScalarType]):
    pass
