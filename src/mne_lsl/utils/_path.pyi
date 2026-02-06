from collections.abc import Generator
from pathlib import Path

def walk(path: Path) -> Generator[Path]:
    """Walk recursively through a directory tree and yield the existing files.

    Parameters
    ----------
    path : Path
        Path to a directory.
    """
