from pathlib import Path

from ..utils._checks import ensure_path as ensure_path
from ._fetch import fetch_dataset as fetch_dataset

_REGISTRY: Path

def _make_registry(folder: str | Path, output: str | Path | None = None) -> None:
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
        Path to the sample dataset, by default in ``"~/mne_data/MNE-LSL"``.
    """
