from __future__ import annotations  # c.f. PEP 563, PEP 649

from copy import deepcopy
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from mne.filters import estimate_ringing_samples, create_filter
from scipy.signal import sosfilt_zi

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from .._typing import ScalarIntType


class StreamFilter(dict):
    """Class defining a filter."""

    def __repr__(self):  # noqa: D105
        return f"<IIR causal filter ({self['l_freq']}, {self['h_freq']}) Hz>"

    def __eq__(self, other: Any):
        """Equality operator."""
        if not isinstance(other, StreamFilter) or sorted(self) != sorted(other):
            return False
        for key in self:
            if key == "zi":  # special case since it's either a np.ndarray or None
                if (self[key] is None or other[key] is None) or not np.array_equal(
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


def _combine_filters(
    filter1: StreamFilter,
    filter2: StreamFilter,
    picks: NDArray[+ScalarIntType],
    *,
    copy: bool = True,
) -> StreamFilter:
    """Combine 2 filters applied on the same set of channels."""
    assert filter1["sfreq"] == filter2["sfreq"]
    if copy:
        filter1 = deepcopy(filter1)
        filter2 = deepcopy(filter2)
    system = np.vstack((filter1["sos"], filter2["sos"]))
    # for 'l_freq', 'h_freq', 'iir_params' we store the filter(s) settings in ordered
    # tuples to keep track of the original settings of individual filters.
    for key in ("l_freq", "h_freq", "iir_params"):
        filter1[key] = list(
            (filter1[key],) if not isinstance(filter1[key], tuple) else filter1[key]
        )
        filter2[key] = list(
            (filter2[key],) if not isinstance(filter2[key], tuple) else filter2[key]
        )
    combined_filter = {
        "output": "sos",
        "padlen": estimate_ringing_samples(system),
        "sos": system,
        "zi": None,  # reset initial conditions on channels combined
        "zi_coeff": sosfilt_zi(system)[..., np.newaxis],
        "l_freq": tuple(filter1["l_freq"] + filter2["l_freq"]),
        "h_freq": tuple(filter1["h_freq"] + filter2["h_freq"]),
        "iir_params": tuple(filter1["iir_params"] + filter2["iir_params"]),
        "sfreq": filter1["sfreq"],
        "picks": picks,
    }
    return StreamFilter(combined_filter)


def _uncombine_filters(filt: StreamFilter) -> list[StreamFilter]:
    """Uncombine a combined filter into its individual components."""
    val = (isinstance(filt[key], tuple) for key in ("l_freq", "h_freq", "iir_params"))
    if not all(val) and any(val):
        raise RuntimeError(
            "The combined filter contains keys 'l_freq', 'h_freq' and 'iir_params' as "
            "both tuple and non-tuple, which should not be possible. Please contact "
            "the developers."
        )
    elif not all(val):
        return [filt]
    # instead of trying to un-tangled the 'sos' matrix, we simply create a new filter
    # for each individual component.
    filters = list()
    for lfq, hfq, iir_param in zip(
        filt["l_freq"], filt["h_freq"], filt["iir_params"], strict=True
    ):
        filt = create_filter(
            data=None,
            sfreq=filt["sfreq"],
            l_freq=lfq,
            h_freq=hfq,
            method="iir",
            iir_params=iir_param,
            phase="forward",
            verbose="CRITICAL",  # effectively disable logs
        )
        filt.update(
            zi=None,
            zi_coeff=sosfilt_zi(filt["sos"])[..., np.newaxis],
            l_freq=lfq,
            h_freq=hfq,
            iir_params=iir_param,
            sfreq=filt["sfreq"],
            picks=filt["picks"],
        )
        del filt["order"]
        del filt["ftype"]
        filters.append(StreamFilter(filt))
    return filters


def _sanitize_filters(
    filters: list[StreamFilter], filter_: StreamFilter, *, copy: bool = True
) -> list[dict[str, Any]]:
    """Sanitize the list of filters to ensure non-overlapping channels."""
    filters = deepcopy(filters) if copy else filters
    additional_filters = []
    for filt in filters:
        intersection = np.intersect1d(
            filt["picks"], filter_["picks"], assume_unique=True
        )
        if intersection.size == 0:
            continue  # non-overlapping channels
        additional_filters.append(_combine_filters(filt, filter_, picks=intersection))
        # reset initial conditions for the overlapping filter
        filt["zi"] = None  # TODO: instead of reset, select initial conditions.
        # remove overlapping channels from both filters
        filt["picks"] = np.setdiff1d(filt["picks"], intersection, assume_unique=True)
        filter_["picks"] = np.setdiff1d(
            filter_["picks"], intersection, assume_unique=True
        )
    # prune filters without any channels
    filters = [
        filt
        for filt in filters + additional_filters + [filter_]
        if filt["picks"].size != 0
    ]
    return filters
