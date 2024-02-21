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
_ERROR_MSG: str

def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system.

    The library is loaded in the following order:

    1. Search in the environment variables.
    2. Search in the system folder.
    3. Search in _LIB_FOLDER.
    4. Fetch on GitHub.
    """

def _load_liblsl_environment_variables() -> Optional[str]:
    """Load the binary LSL library from the environment variables.

    Returns
    -------
    libpath : str | None
        Path to the binary LSL library or None if it could not be found.
    """

def _load_liblsl_system() -> Optional[str]:
    """Load the binary LSL library from the system path/folders.

    Returns
    -------
    libpath : str | None
        Path to the binary LSL library or None if it could not be found.
    """

def _load_liblsl_mne_lsl(*, folder: Path = ...) -> Optional[str]:
    """Load the binary LSL library from the system path/folders.

    Parameters
    ----------
    folder : Path
        Path to the folder in which to look for the binary LSL library.

    Returns
    -------
    libpath : str | None
        Path to the binary LSL library or None if it could not be found.
    """

def _fetch_liblsl(
    *,
    folder: Union[str, Path] = ...,
    url: str = "https://api.github.com/repos/sccn/liblsl/releases/latest",
) -> str:
    """Fetch liblsl on the release page.

    Parameters
    ----------
    folder : Path
        Path to the folder in which to download the binary LSL library.
    url : str
        URL from which to fetch the release of liblsl.

    Returns
    -------
    libpath : str
        Path to the binary LSL library.

    Notes
    -----
    This function will raise if it was unable to fetch the release of liblsl. Thus, it
    will never return None.
    """

def _pooch_processor_liblsl(fname: str, action: str, pooch: Pooch) -> str:
    """Processor of the pooch-downloaded liblsl.

    Parameters
    ----------
    fname : str
        The full path of the file in the local data storage.
    action : str
        Either:
        * "download" (file doesn't exist and will be downloaded)
        * "update" (file is outdated and will be downloaded)
        * "fetch" (file exists and is updated so no download is necessary)
    pooch : Pooch
        The instance of the Pooch class that is calling this function.

    Returns
    -------
    fname : str
        The full path to the file in the local data storage.
    """

def _is_valid_libpath(libpath: str) -> bool:
    """Check if the library path is valid."""

def _attempt_load_liblsl(
    libpath: Union[str, Path], *, issue_warning: bool = True
) -> tuple[str, Optional[int]]:
    """Try loading a binary LSL library.

    Parameters
    ----------
    libpath : Path
        Path to the binary LSL library.
    issue_warning : bool
        If True, issue a warning if the library could not be loaded.

    Returns
    -------
    libpath : str
        Path to the binary LSL library, converted to string for the given OS.
    version : int
        Version of the binary LSL library.
        The major version is version // 100.
        The minor version is version % 100.
    """

def _is_valid_version(
    libpath: str, version: int, *, issue_warning: bool = True
) -> bool:
    """Check if the version of the library is supported by MNE-LSL.

    Parameters
    ----------
    libpath : str
        Path to the binary LSL library, converted to string for the given OS.
    version : int
        Version of the binary LSL library.
        The major version is version // 100.
        The minor version is version % 100.
    issue_warning : bool
        If True, issue a warning if the version is not supported.

    Returns
    -------
    valid : bool
        True if the version is supported, False otherwise.
    """

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
