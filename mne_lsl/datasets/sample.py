from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pooch
from mne.utils import get_config

from ..utils._checks import ensure_path
from ._fetch import fetch_dataset

if TYPE_CHECKING:
    from typing import Optional, Union

_REGISTRY: Path = files("mne_lsl.datasets") / "sample-registry.txt"


def _make_registry(
    folder: Union[str, Path], output: Optional[Union[str, Path]] = None
) -> None:  # pragma: no cover
    """Create the registry file for the sample dataset.

    Parameters
    ----------
    folder : path-like
        Path to the sample dataset.
    output : path-like
        Path to the output registry file.
    """
    folder = ensure_path(folder, must_exist=True)
    output = _REGISTRY if output is None else output
    pooch.make_registry(folder, output=output, recursive=True)


def data_path() -> Path:  # pragma: no cover
    """Return the path to the sample dataset, downloaded if needed.

    Returns
    -------
    path : Path
        Path to the sample dataset, by default in ``"~/mne_data/MNE-LSL-data"``.
    """
    path = (
        Path(get_config("MNE_DATA", Path.home() / "mne_data")).expanduser()
        / "MNE-LSL-data"
        / "sample"
    )
    base_url = "https://github.com/mscheltienne/mne-lsl-datasets/raw/main/sample"
    return fetch_dataset(path, base_url, _REGISTRY)
