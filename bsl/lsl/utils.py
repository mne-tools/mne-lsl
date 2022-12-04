import os
from ctypes import CDLL, util
from pathlib import Path
from platform import system
from typing import Dict, Optional, Tuple, Union

from . import minversion
from .. import logger


def find_liblsl():
    """Search for the binary LSL library on the system."""
    # look for the PYLSL_LIB environment variable
    libpath, version = _find_liblsl_env()
    if version is not None:
        assert libpath is not None  # sanity-check
        lib = CDLL(libpath)
        return lib

    else:
        # look in folder


def _find_liblsl_env() -> Tuple[Optional[str], Optional[int]]:
    """Search for the LSL library in the environment variable PYLSL_LIB.

    Returns
    -------
    libpath : str | None
        Path to the binary LSL library, converted to string for the given OS.
        None if the environment variable PYLSL_LIB does not exists or yields
        a non-existent path.
    version : int
        Version of the binary LSL library.
        The major version is version // 100.
        The minor version is version % 100.
    """
    if "PYLSL_LIB" not in os.environ:
        return None, None

    libpath = Path(os.environ["PYLSL_LIB"])
    if libpath.exists():
        libpath, version = _load_liblsl(libpath)
        if version is None:
            logger.warning(
                "The LIBLSL '%s' provided in the environment variable "
                "'PYLSL_LIB' can not be loaded.", libpath
            )
        elif version < minversion:
            logger.warning(
                "The LIBLSL '%s' provided in the environment variable "
                "'PYLSL_LIB' is outdated. The version is %i.%i while the "
                "minimum version required by BSL is %i.%i.",
                libpath,
                version // 100,
                version % 100,
                minversion // 100,
                minversion % 100,
            )
            version = None
    else:
        logger.warning(
            "The LIBLSL path '%s' provided in the environment variable "
            "'PYLSL_LIB' does not exists.", libpath
        )
        libpath = None
        version = None
    return libpath, version


def _load_liblsl(libpath: Union[str, Path]) -> Tuple[str, Optional[int]]:
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
    libpath = str(Path) if isinstance(libpath, Path) else libpath
    try:
        lib = CDLL(libpath)
        version = lib.lsl_library_version()
        del lib
    except OSError:
        version = None
    return libpath, version
