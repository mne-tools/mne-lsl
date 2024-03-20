from typing import Generic, TypeVar

import numpy as np

ScalarFloatType = TypeVar("ScalarFloatType", np.float32, np.float64)
ScalarIntType = TypeVar("ScalarIntType", np.int8, np.int16, np.int32, np.int64)
ScalarType = TypeVar(
    "ScalarType", np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
)

class ScalarFloatArray(np.ndarray, Generic[ScalarFloatType]): ...
class ScalarIntArray(np.ndarray, Generic[ScalarIntType]): ...
class ScalarArray(np.ndarray, Generic[ScalarType]): ...
