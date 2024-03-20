from __future__ import annotations  # c.f. PEP 563, PEP 649

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from mne.filter import create_filter as create_filter_mne
from scipy.signal import sosfilt_zi

from ..utils._checks import check_type
from ..utils.logs import logger

if TYPE_CHECKING:
    from typing import Any, Optional


class StreamFilter(dict):
    """Class defining a filter."""

    _ORDER_STR: dict[int, str] = {1: "1st", 2: "2nd"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "iir_params" not in self:
            warn(
                "The 'iir_params' key is missing, which is unexpected.",
                RuntimeWarning,
                stacklevel=2,
            )
            self["iir_params"] = dict()
        for key in ("ftype", "order"):
            if key not in self:
                continue
            if key not in self["iir_params"]:
                self["iir_params"][key] = self[key]
            else:
                if self[key] != self["iir_params"][key]:
                    raise RuntimeError(
                        f"The value of '{key}' in the filter dictionary and in the "
                        "filter parameters '{iir_params}' is inconsistent. "
                        f"{self[key]} != {self['iir_params'][key]}."
                    )
            del self[key]

    def __repr__(self):  # noqa: D105
        order = self._ORDER_STR.get(
            self["iir_params"]["order"], f"{self['iir_params']['order']}th"
        )
        if (
            any(elt is None for elt in (self["l_freq"], self["h_freq"]))
            or self["l_freq"] < self["h_freq"]
        ):
            representation = (
                f"<IIR {order} causal filter @ ({self['l_freq']}, {self['h_freq']}) Hz "
                f"({self['iir_params']['ftype']})>"
            )
        else:
            avg = (self["l_freq"] + self["h_freq"]) / 2
            representation = (
                f"<IIR {order} causal notch filter @ {avg} Hz "
                f"({self['l_freq']}, {self['h_freq']}) ({self['iir_params']['ftype']})>"
            )
        return representation

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


def create_filter(
    sfreq: float,
    l_freq: Optional[float],
    h_freq: Optional[float],
    iir_params: dict[str, Any],
) -> dict[str, Any]:
    """Create an IIR causal filter.

    Parameters
    ----------
    sfreq : float
        The sampling frequency in Hz.
    %(l_freq)s
    %(h_freq)s
    %(iir_params)s

    Returns
    -------
    filt : dict
        The filter parameters and initial conditions.
    """
    filt = create_filter_mne(
        data=None,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=iir_params,
        phase="forward",
        verbose=logger.level,
    )
    # store filter parameters and initial conditions
    filt.update(
        zi_unit=sosfilt_zi(filt["sos"])[..., np.newaxis],
        zi=None,
        l_freq=l_freq,
        h_freq=h_freq,
        iir_params=iir_params,
        sfreq=sfreq,
    )
    return filt


def ensure_sos_iir_params(
    iir_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Ensure that the filter parameters include SOS output."""
    if iir_params is None:
        return dict(order=4, ftype="butter", output="sos")
    check_type(iir_params, (dict,), "iir_params")
    if ("output" in iir_params and iir_params["output"] != "sos") or all(
        key in iir_params for key in ("a", "b")
    ):
        warn(
            "Only 'sos' output is supported for real-time filtering. The filter "
            "output will be automatically changed. Please set "
            "iir_params=dict(output='sos', ...) in your call to the filtering method.",
            RuntimeWarning,
            stacklevel=2,
        )
        for key in ("a", "b"):
            if key in iir_params:
                del iir_params[key]
    iir_params["output"] = "sos"
    return iir_params
