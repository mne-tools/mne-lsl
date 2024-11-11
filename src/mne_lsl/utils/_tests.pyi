from pathlib import Path

from mne import Info
from mne.io import BaseRaw

from .._typing import ScalarArray as ScalarArray

def sha256sum(fname: str | Path) -> str:
    """Efficiently hash a file."""

def match_stream_and_raw_data(data: ScalarArray, raw: BaseRaw) -> None:
    """Check if the data array is part of the provided raw."""

def compare_infos(info1: Info, info2: Info) -> None:
    """Check that 2 infos are similar, even if some minor attribute deviate."""
