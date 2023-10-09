from __future__ import annotations  # c.f. PEP 563 and PEP 649

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pooch
from mne.utils import get_config

from ..utils._checks import ensure_path
from ._fetch import fetch_dataset

if TYPE_CHECKING:
    from typing import Optional, Union


def _make_registry(
    folder: Union[str, Path], output: Optional[Union[str, Path]] = None
) -> None:
    """Create the registry file for the sample dataset.

    Parameters
    ----------
    folder : path-like
        Path to the sample dataset.
    output : path-like
        Path to the output registry file.
    """
    folder = ensure_path(folder, must_exist=True)
    output = (
        files("mne_lsl.datasets") / "sample-registry.txt" if output is None else output
    )
    pooch.make_registry(folder, output=output, recursive=True)


def data_path() -> Path:
    """Return the path to the sample dataset, downloaded if needed.

    Returns
    -------
    path : Path
        Path to the sample dataset, by default in ``"~/mne_data/MNE-LSL"``.
    """
    path = (
        Path(get_config("MNE_DATA", Path.home())).expanduser()
        / "mne_data"
        / "MNE-LSL"
        / "sample"
    )
    base_url = "https://github.com/mscheltienne/mne-lsl-datasets/raw/main/sample"
    registry = files("mne_lsl.datasets") / "sample-registry.txt"
    return fetch_dataset(path, base_url, registry)
