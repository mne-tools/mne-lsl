from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

if TYPE_CHECKING:
    from typing import Any


class StreamFilter(dict):
    """Class defining a filter."""

    _ORDER_STR: dict[int, str] = {1: "1st", 2: "2nd"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in ("ftype", "order"):
            if key in self:
                assert key in self["iir_params"]  # sanity-check
                del self[key]

    def __repr__(self):  # noqa: D105
        order = self._ORDER_STR.get(
            self["irr_params"]["order"], f"{self['irr_params']['order']}th"
        )
        return f"<IIR {order} causal filter ({self['l_freq']}, {self['h_freq']}) Hz>"

    def __eq__(self, other: Any):
        """Equality operator."""
        if not isinstance(other, StreamFilter) or set(self) != set(other):
            return False
        for key in self:
            if key == "zi":  # special case since it's either a np.ndarray or None
                if self[key] is None and other[key] is None:
                    continue
                elif ((self[key] is None) ^ (other[key] is None)) or not np.array_equal(
                    self[key], other[key]
                ):
                    return False
                continue
            type_ = type(self[key])
            if not isinstance(other[key], type_):  # sanity-check
                warn(
                    f"The type of the key '{key}' is different between the 2 filters, "
                    "which should not be possible. Please contact the developers.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return False
            if (
                type_ is np.ndarray
                and not np.array_equal(self[key], other[key], equal_nan=True)
            ) or (type_ is not np.ndarray and self[key] != other[key]):
                return False
        return True

    def __ne__(self, other: Any):  # explicit method required to issue warning
        """Inequality operator."""
        return not self.__eq__(other)
