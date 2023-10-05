from __future__ import annotations  # c.f. PEP 563 and PEP 649

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Generator


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
            yield from _walk(entry)
        else:
            yield entry
