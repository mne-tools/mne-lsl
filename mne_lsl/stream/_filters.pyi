from typing import Any

from ..utils._checks import check_type as check_type
from ..utils.logs import logger as logger

class StreamFilter(dict):
    """Class defining a filter."""

    _ORDER_STR: dict[int, str]

    def __init__(self, *args, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any):
        """Equality operator."""

    def __ne__(self, other: Any):
        """Inequality operator."""

def create_filter(
    sfreq: float, l_freq: float | None, h_freq: float | None, iir_params: dict[str, Any]
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

def ensure_sos_iir_params(iir_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ensure that the filter parameters include SOS output."""
