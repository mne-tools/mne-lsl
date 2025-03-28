from __future__ import annotations

from typing import TYPE_CHECKING

import pooch

from ..utils._checks import ensure_path
from ..utils._path import walk
from ..utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path


def fetch_dataset(path: Path, base_url: str, registry: str | Path) -> Path:
    """Fetch a dataset from the remote.

    Parameters
    ----------
    path : str | Path
        Local path where the dataset should be cloned.
    base_url : str
        Base URL for the remote data sources. All requests will be made relative to this
        URL. If the URL does not end in a '/', a trailing '/' will be added
        automatically.
    registry : str | Path
        Path to the txt file containing the registry.

    Returns
    -------
    path : Path
        Absolute path to the local clone of the dataset.
    """
    path = ensure_path(path, must_exist=False)
    registry = ensure_path(registry, must_exist=True)
    fetcher = pooch.create(
        path=path,
        base_url=base_url,
        registry=None,
        retry_if_failed=2,
        allow_updates=True,
    )
    fetcher.load_registry(registry)
    # check the local copy of the dataset
    for file in fetcher.registry:
        fetcher.fetch(file)
    # remove outdated files from the dataset
    for entry in walk(path):
        if str(entry.relative_to(path).as_posix()) not in fetcher.registry:
            logger.info("Removing outdated dataset file '%s'.", entry.relative_to(path))
            entry.unlink(missing_ok=True)
    return fetcher.abspath
