from pathlib import Path as Path
from typing import Union

from mne import Info
from mne.io import BaseRaw
from numpy.typing import NDArray

from .._typing import ScalarType as ScalarType

def sha256sum(fname: Union[str, Path]) -> str:
    """Efficiently hash a file."""

def match_stream_and_raw_data(data: NDArray[None], raw: BaseRaw) -> None:
    """Check if the data array is part of the provided raw."""

def compare_infos(info1: Info, info2: Info) -> None:
    """Check that 2 infos are similar, even if some minor attribute deviate."""
