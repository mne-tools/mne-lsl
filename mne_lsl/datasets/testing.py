from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pooch
from mne.utils import get_config

from ..utils._checks import ensure_path
from ._fetch import fetch_dataset

if TYPE_CHECKING:
    pass

_REGISTRY: Path = files("mne_lsl.datasets") / "testing-registry.txt"


def _make_registry(folder: str | Path, output: str | Path | None = None) -> None:
    """Create the registry file for the testing dataset.

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


def data_path() -> Path:
    """Return the path to the testing dataset, downloaded if needed.

    Returns
    -------
    path : Path
        Path to the testing dataset, by default in ``"~/mne_data/MNE-LSL-data"``.
    """
    path = (
        Path(get_config("MNE_DATA", Path.home() / "mne_data")).expanduser()
        / "MNE-LSL-data"
        / "testing"
    )
    base_url = "https://github.com/mscheltienne/mne-lsl-datasets/raw/main/testing"
    return fetch_dataset(path, base_url, _REGISTRY)
