"""Temporary bug-fixes awaiting an upstream fix."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from mne.event import _find_events
from mne.utils import check_version

if TYPE_CHECKING:
    from .._typing import ScalarArray


# https://github.com/sphinx-gallery/sphinx-gallery/issues/1112
class WrapStdOut:
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.
    """

    def __getattr__(self, name):  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux).
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError(f"'file' object has not attribute '{name}'")


def find_events(
    data: ScalarArray,
    first_samp: int,
    verbose: bool | str | int | None = None,
    output: str = None,
    consecutive: bool | str = None,
    min_samples: float = None,
    mask: int | None = None,
    uint_cast: bool = None,
    mask_type: str = None,
    initial_event: bool = None,
    ch_name: str = None,
):
    """Compatibility function for older MNE versions.

    To be dropped when MNE 1.6 is the minimum supported version.
    """
    kwargs = dict(
        data=data,
        first_samp=first_samp,
        verbose=verbose,
        output=output,
        consecutive=consecutive,
        min_samples=min_samples,
        mask=mask,
        uint_cast=uint_cast,
        mask_type=mask_type,
        initial_event=initial_event,
    )
    if check_version("mne", "1.6"):
        kwargs["ch_name"] = ch_name
    return _find_events(**kwargs)
