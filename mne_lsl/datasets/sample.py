from __future__ import annotations  # c.f. PEP 563 and PEP 649

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pooch
from mne.utils import get_config

from ._fetch import fetch_dataset

if TYPE_CHECKING:
    from typing import Optional, Union


def _make_registry(output: Optional[Union[str, Path]] = None) -> None:
    """Create the registry file for the sample dataset.

    Parameters
    ----------
    output : str | Path
        Path to the output registry file.
    """
    folder = files("mne_lsl").parent / "datasets" / "sample"
    if not folder.exists():
        raise RuntimeError(
            "The sample dataset registry can only be created from a clone of the "
            "repository."
        )
    output = (
        files("mne_lsl.datasets") / "sample-registry.txt" if output is None else output
    )
    pooch.make_registry(folder, output=output, recursive=True)


def data_path() -> Path:
    """Return the path to the sample dataset, downloaded if needed.

    Returns
    -------
    path : Path
        Path to the sample dataset, by default in ``"~/mne_data/mne_lsl"``.
    """
    path = Path(get_config("MNE_DATA", Path.home())).expanduser() / "mne_lsl" / "sample"
    base_url = "https://github.com/mne-tools/mne-lsl/raw/main/datasets/sample/"
    registry = files("mne_lsl.datasets") / "sample-registry.txt"
    return fetch_dataset(path, base_url, registry)
