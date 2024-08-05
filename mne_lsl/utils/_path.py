from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def walk(path: Path) -> Generator[Path, None, None]:
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
            yield from walk(entry)
        else:
            yield entry
