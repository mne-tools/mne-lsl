from pathlib import Path
from typing import Optional, Union

from ..utils._checks import ensure_path as ensure_path
from ._fetch import fetch_dataset as fetch_dataset

_REGISTRY: Path

def _make_registry(
    folder: Union[str, Path], output: Optional[Union[str, Path]] = ...
) -> None:
    """Create the registry file for the sample dataset.

    Parameters
    ----------
    folder : path-like
        Path to the sample dataset.
    output : path-like
        Path to the output registry file.
    """

def data_path() -> Path:
    """Return the path to the sample dataset, downloaded if needed.

    Returns
    -------
    path : Path
        Path to the sample dataset, by default in ``"~/mne_data/mne_lsl"``.
    """
