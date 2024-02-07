from ctypes import CDLL
from pathlib import Path
from typing import Optional, Union

from _typeshed import Incomplete
from pooch import Pooch

from .._version import __version__ as __version__
from ..utils._checks import ensure_path as ensure_path
from ..utils._path import walk as walk
from ..utils.logs import logger as logger

_LIB_FOLDER: Path
_VERSION_MIN: int
_VERSION_PROTOCOL: int
_PLATFORM: str
_PLATFORM_SUFFIXES: dict[str, str]
_SUPPORTED_DISTRO: dict[str, tuple[str, ...]]
_ERROR_MSG: str

def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system.

    1. Search in the environment variables.
    2. Search in the system folder.
    3. Search in _LIB_FOLDER.
    4. Fetch on GitHub.
    """

def _load_liblsl_environment_variables() -> Optional[str]: ...
def _load_liblsl_system() -> Optional[str]: ...
def _load_liblsl_mne_lsl() -> Optional[str]: ...
def _fetch_liblsl(
    *,
    folder: Union[str, Path] = ...,
    url: str = "https://api.github.com/repos/sccn/liblsl/releases/latest",
) -> CDLL:
    """Fetch liblsl from the GitHub release page."""

def _pooch_processor_liblsl(fname: str, action: str, pooch: Pooch) -> str:
    """Processor of the pooch-downloaded liblsl."""

def _attempt_load_liblsl(
    libpath: Union[str, Path], *, issue_warning: bool = True
) -> tuple[str, Optional[int]]:
    """Try loading a binary LSL library."""

def _is_valid_libpath(libpath: str) -> bool:
    """Check if the library path is valid."""

def _is_valid_version(
    libpath: str, version: int, *, issue_warning: bool = True
) -> bool:
    """Check if the version of the library is supported by MNE-LSL."""

def _set_types(lib: CDLL) -> CDLL:
    """Set the argument and return types for the different liblsl functions.

    Parameters
    ----------
    lib : CDLL
        Loaded binary LSL library.

    Returns
    -------
    lib : CDLL
        Loaded binary LSL library with the return types set.
    """

lib: Incomplete
