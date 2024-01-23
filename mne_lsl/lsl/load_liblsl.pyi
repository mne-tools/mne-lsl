from ctypes import CDLL
from pathlib import Path
from typing import Optional, Union

from _typeshed import Incomplete
from pooch import Pooch

from .._version import __version__ as __version__
from ..utils._path import walk as walk
from ..utils.logs import logger as logger

_VERSION_MIN: int
_VERSION_PROTOCOL: int
_PLATFORM: str
_PLATFORM_SUFFIXES: dict[str, str]
_SUPPORTED_DISTRO: dict[str, tuple[str, ...]]
_ERROR_MSG: str
_LIB_FOLDER: Path

def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system."""

def _find_liblsl() -> Optional[CDLL]:
    """Search for liblsl in the environment variable and in the system folders.

    Returns
    -------
    lib : CDLL | None
        Loaded binary LSL library. None if not found.
    """

def _fetch_liblsl(folder: Path = ...) -> Optional[CDLL]:
    """Fetch liblsl on the release page.

    Parameters
    ----------
    folder : Path
        Folder where the fetched liblsl is stored.

    Returns
    -------
    lib : CDLL | None
        Loaded binary LSL library. None if not found for this platform.
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

def _attempt_load_liblsl(libpath: Union[str, Path]) -> tuple[str, Optional[int]]:
    """Try loading a binary LSL library.

    Parameters
    ----------
    libpath : Path
        Path to the binary LSL library.

    Returns
    -------
    libpath : str
        Path to the binary LSL library, converted to string for the given OS.
    version : int
        Version of the binary LSL library.
        The major version is version // 100.
        The minor version is version % 100.
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
