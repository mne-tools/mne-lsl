from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
import platform
from ctypes import CDLL
from ctypes.util import find_library
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from ..utils.logs import logger

if TYPE_CHECKING:
    from typing import Optional, Union


# folder where the library is fetched
_LIB_FOLDER: Path = files("mne_lsl.lsl") / "lib"
# minimum liblsl version. The major version is given by version // 100
# and the minor version is given by version % 100.
_VERSION_MIN: int = 115
# liblsl objects created with the same protocol version are inter-compatible.
_VERSION_PROTOCOL: int = 110  # un-used for now
_PLATFORM: str = platform.system().lower().strip()
_PLATFORM_SUFFIXES: dict[str, str] = {
    "windows": ".dll",
    "darwin": ".dylib",
    "linux": ".so",
}


def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system.

    1. Search in the environment variables.
    2. Search in the system folder.
    3. Search in _LIB_FOLDER.
    4. Fetch on GitHub.
    """
    if _PLATFORM not in _PLATFORM_SUFFIXES:
        raise RuntimeError(
            "The OS could not be determined. Please open an issue on GitHub and "
            "provide the error traceback to the developers."
        )
    libpath = _load_liblsl_environment_variables()
    libpath = _load_liblsl_system() if libpath is None else libpath
    libpath = _load_liblsl_mne_lsl() if libpath is None else libpath


def _load_liblsl_environment_variables() -> Optional[str]:
    for variable in ("MNE_LSL_LIB", "PYLSL_LIB"):
        libpath = os.environ.get(variable, None)
        if libpath is None:
            logger.debug("The environment variable %s is not set.", variable)
            continue
        logger.debug(
            "Attempting to load libpath '%s' stored in the environment variable '%s'.",
            libpath,
            variable,
        )
        _is_valid_libpath(libpath)
        libpath, version = _attempt_load_liblsl(libpath)
        if version is None:
            warn(
                f"The LIBLSL '{libpath}' can not be loaded.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if _is_valid_version(version):
            return libpath
    return None


def _load_liblsl_system() -> Optional[str]:
    libpath = find_library("liblsl")
    if libpath is None:
        logger.debug("The library liblsl is not found in the system folder.")
        return None
    if _PLATFORM != "linux":
        # validate the path only for macOS and windows because find_library does not
        # return an absolute path on linux systems.
        _is_valid_libpath(libpath)
    libpath, version = _attempt_load_liblsl(libpath)
    if version is None:
        return None
    if _is_valid_version(version):
        return libpath
    else:
        return None


def _load_liblsl_mne_lsl() -> Optional[str]:
    for libpath in _LIB_FOLDER.glob(f"*{_PLATFORM_SUFFIXES[_PLATFORM]}"):
        libpath, version = _attempt_load_liblsl(libpath)
        if version is None:
            libpath = Path(libpath)
            warn(
                "The previously downloaded liblsl {libpath.name} could not be loaded. "
                "It will be removed.",
                RuntimeWarning,
                stacklevel=2,
            )
            libpath.unlink(missing_ok=False)
            continue
        _is_valid_version(version)
        return libpath
    return None


def _attempt_load_liblsl(libpath: Union[str, Path]) -> tuple[str, Optional[int]]:
    """Try loading a binary LSL library."""
    libpath = str(libpath) if isinstance(libpath, Path) else libpath
    try:
        lib = CDLL(libpath)
        version = lib.lsl_library_version()
        del lib
    except OSError:
        version = None
    return libpath, version


def _is_valid_libpath(libpath: str) -> bool:
    """Check if the library path is valid."""
    libpath = Path(libpath)
    if libpath.suffix != _PLATFORM_SUFFIXES[_PLATFORM]:
        warn(
            f"The liblsl '{libpath}' ends with '{libpath.suffix}' which is "
            f"different from the expected extension '{_PLATFORM_SUFFIXES[_PLATFORM]}' "
            f"for {_PLATFORM} based OS.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    if not libpath.exists():
        warn(
            f"The LIBLSL '{libpath}' does not exist.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def _is_valid_version(libpath: str, version: int) -> bool:
    """Check if the version of the library is supported by MNE-LSL."""
    if version < _VERSION_MIN:
        warn(
            f"The LIBLSL '{libpath}' is outdated. The version is "
            f"{version // 100}.{version % 100} while the minimum version required by "
            f"MNE-LSL is {_VERSION_MIN // 100}.{_VERSION_MIN % 100}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True
