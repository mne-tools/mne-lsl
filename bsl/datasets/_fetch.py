from __future__ import annotations  # c.f. PEP 563 and PEP 649

from typing import TYPE_CHECKING

import pooch

from ..utils._checks import ensure_path
from ..utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Generator, Union


def fetch_dataset(path: Path, base_url: str, registry: Union[str, Path]) -> Path:
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
    for entry in _walk(path):
        if str(entry.relative_to(path).as_posix()) not in fetcher.registry:
            logger.info("Removing outdated dataset file '%s'.", entry.relative_to(path))
            entry.unlink(missing_ok=True)
    return fetcher.abspath


def _walk(path: Path) -> Generator[Path, None, None]:
    """Walk recursively through a directory tree and yield the existing files.

    Parameters
    ----------
    path : Path
        Path to a directory.
    """
    if not path.is_dir():
        raise RuntimeError(
            f"The provided path '{path}' is not a directory. It can not be walked."
        )
    for entry in path.iterdir():
        if entry.is_dir():
            yield from _walk(entry)
        else:
            yield entry
