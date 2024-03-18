from typing import IO, Callable

from packaging.requirements import Requirement

from ._checks import check_type as check_type
from .logs import _use_log_level as _use_log_level

def sys_info(fid: IO | None = None, developer: bool = False):
    """Print the system information for debugging.

    Parameters
    ----------
    fid : file-like | None
        The file to write to, passed to :func:`print`.
        Can be None to use :data:`sys.stdout`.
    developer : bool
        If True, display information about optional dependencies.
    """

def _list_dependencies_info(
    out: Callable,
    ljust: int,
    package: str,
    dependencies: list[Requirement],
    unicode: bool,
) -> None:
    """List dependencies names and versions."""
