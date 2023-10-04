from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
import platform
import requests
import tarfile
import zipfile
from ctypes import CDLL, c_char_p, c_double, c_long, c_void_p, sizeof
from ctypes.util import find_library
from importlib.resources import files
from pathlib import Path
from shutil import move, rmtree
from typing import TYPE_CHECKING

import pooch

from ..utils.logs import logger

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union

    from pooch import Pooch

# Minimum liblsl version. The major version is given by version // 100
# and the minor version is given by version % 100.
_VERSION_MIN = 115
# liblsl objects created with the same protocol version are inter-compatible.
_VERSION_PROTOCOL = 110
_PLATFORM = platform.system().lower().strip()
_PLATFORM_SUFFIXES = {
    "windows": ".dll",
    "darwin": ".dylib",
    "linux": ".so",
}
# variables which should be kept in sync with liblsl release
_SUPPORTED_DISTRO = {
    "ubuntu": ("18.04", "20.04", "22.04"),
}


def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system."""
    if _PLATFORM not in _PLATFORM_SUFFIXES:
        raise RuntimeError(
            "The OS could not be determined. Please open an issue on GitHub and "
            "provide the error traceback to the developers."
        )
    lib = _find_liblsl()
    if lib is not None:
        return _set_types(lib)
    lib = _fetch_liblsl()
    if lib is not None:
        return _set_types(lib)
    else:
        raise RuntimeError(
            "The liblsl library could not be found on your system or fetched by "
            "MNE-LSL for your platform. Please visit the liblsl repository "
            "(https://github.com/sccn/liblsl) to find a release for your platform or "
            "instruction to build the library on your platforn."
        )


def _find_liblsl() -> Optional[CDLL]:
    """Search for liblsl in the environment variable and in the system folders.

    Returns
    -------
    lib : CDLL | None
        Loaded binary LSL library. None if not found.
    """
    for libpath in (os.environ.get("MNE_LSL_LIB", None), find_library("lsl")):
        if libpath is None:
            continue
        libpath = Path(libpath)
        if libpath.suffix != _PLATFORM_SUFFIXES[_PLATFORM]:
            logger.warning(
                "The LIBLSL '%s' ends with '%s' which is different from the expected "
                "extension '%s' for this OS.",
                libpath,
                libpath.suffix,
                _PLATFORM_SUFFIXES[_PLATFORM],
            )
            continue
        if not libpath.exists():
            logger.warning("The LIBLSL '%s' does not exist.")
            continue
        libpath, version = _attempt_load_liblsl(libpath)
        if version is None:
            logger.warning("The LIBLSL '%s' can not be loaded.", libpath)
            continue
        if version < _VERSION_MIN:
            logger.warning(
                "The LIBLSL '%s' is outdated. The version is %i.%i while the "
                "minimum version required by MNE-LSL is %i.%i.",
                libpath,
                version // 100,
                version % 100,
                _VERSION_MIN // 100,
                _VERSION_MIN % 100,
            )
            continue
        assert libpath is not None  # sanity-check
        lib = CDLL(libpath)
        break
    else:
        lib = None  # only executed if we did not break the for loop
    return lib


def _attempt_load_liblsl(libpath: Union[str, Path]) -> Tuple[str, Optional[int]]:
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


def _fetch_liblsl() -> Optional[CDLL]:
    """Fetch liblsl on the release page.

    Returns
    -------
    lib : CDLL | None
        Loaded binary LSL library. None if not found for this platform.
    """
    response = requests.get("https://api.github.com/repos/sccn/liblsl/releases/latest")
    assets = [elt for elt in response.json()["assets"] if "liblsl" in elt["name"]]
    # let's try to filter assets for our platform
    if _PLATFORM == "linux":
        import distro

        if distro.name().lower() in _SUPPORTED_DISTRO:
            distro_like = distro.name().lower()
        else:
            for elt in distro.like().split(" "):
                if elt in _SUPPORTED_DISTRO:
                    distro_like = elt
                    break
            else:
                raise RuntimeError(
                    "The liblsl library released on GitHub supports "
                    f"{', '.join(_SUPPORTED_DISTRO)} based distributions. "
                    f"{distro.name()} is not supported. Please build the liblsl "
                    "library from source (https://github.com/sccn/liblsl) and install "
                    "it in the system directories or provide it in the environment "
                    "variable MNE_LSL_LIB."
                )
        if distro.version() not in _SUPPORTED_DISTRO[distro_like]:
            raise RuntimeError(
                "The liblsl library released on GitHub supports "
                f"{', '.join(_SUPPORTED_DISTRO)} based distributions on versions "
                f"{', '.join(_SUPPORTED_DISTRO[distro_like])}. Version "
                f"{distro.version()} is not supported. Please build the liblsl library "
                "from source (https://github.com/sccn/liblsl) and install it in the "
                "system directories or provide it in the environment variable "
                "MNE_LSL_LIB."
            )
        assets = [elt for elt in assets if distro.codename() in elt["name"]]

    elif _PLATFORM == "darwin":
        assets = [elt for elt in assets if "OSX" in elt["name"]]
        if platform.processor() == "arm":
            assets = [elt for elt in assets if "arm" in elt["name"]]
        elif platform.processor() == "i386":
            assets = [elt for elt in assets if "arm" in elt["name"]]
        else:
            raise RuntimeError(
                "The processor architecture could not be determined. Please open an "
                "issue on GitHub and provide the error traceback to the developers."
            )

    elif _PLATFORM == "windows":
        assets = [elt for elt in assets if "Win" in elt["name"]]
        if sizeof(c_void_p) == 4:  # 32 bits
            assets = [elt for elt in assets if "i386" in elt["name"]]
        elif sizeof(c_void_p) == 8:  # 64 bits
            assets = [elt for elt in assets if "amd64" in elt["name"]]
        else:
            raise RuntimeError(
                "The processor architecture could not be determined. Please open an "
                "issue on GitHub and provide the error traceback to the developers."
            )

    if len(assets) != 1:
        return None

    asset = assets[0]
    folder = files("mne_lsl.lsl") / "lib"
    os.makedirs(folder, exist_ok=True)
    libpath = (folder / asset["name"]).with_suffix(_PLATFORM_SUFFIXES[_PLATFORM])
    if libpath.exists():
        _, version = _attempt_load_liblsl(libpath)
        if version is None:
            logger.warning(
                "Previously downloaded liblsl '%s' could not be loaded. It will be "
                "removed."
            )
            libpath.unlink(missing_ok=False)
        else:
            return CDLL(libpath)

    # liblsl was not already present in mne_lsl/lsl/lib, thus we need to download it
    libpath = pooch.retrieve(
        url=asset["browser_download_url"],
        fname=asset["name"],
        path=libpath.parent,
        processor=_pooch_processor_liblsl,
    )
    return CDLL(libpath)


def _pooch_processor_liblsl(fname: str, action: str, pooch: Pooch) -> str:
    """Processor of the pooch-downloaded liblsl.

    Parameters
    ----------
    fname : str
        The full path of the file in the local data storage.
    action : str
        Either: "download" (file doesn't exist and will be downloaded),
        "update" (file is outdated and will be downloaded), or "fetch"
        (file exists and is updated so no download is necessary).
    pooch : Pooch
        The instance of the Pooch class that is calling this function.

    Returns
    -------
    fname : str
        The full path to the file in the local data storage.
    """
    if _PLATFORM == "linux":
        return fname

    folder = files("mne_lsl.lsl") / "lib"
    fname = Path(fname)
    uncompressed = folder / f"{fname.name}.achive"
    if _PLATFORM == "darwin":
        with tarfile.open(fname, "r:bz2") as archive:
            archive.extractall(uncompressed)
        files_ = [
            elt
            for elt in (uncompressed / "lib").iterdir()
            if elt.is_file() and not elt.is_symlink()
        ]
        assert len(files) == 1, "Please contact the developers on GitHub."
        target = (folder / f"{fname.name.split('.tar.bz2')[0]}").with_suffix(
            _PLATFORM_SUFFIXES["darwin"]
        )
        move(files_[0], target)
    elif _PLATFORM == "windows":
        with zipfile.ZipFile(fname, "r") as archive:
            archive.extractall(uncompressed)
        files_ = [
            elt
            for elt in (uncompressed / "bin").iterdir()
            if elt.is_file() and elt.suffix == _PLATFORM_SUFFIXES["windows"]
        ]
        assert len(files) == 1, "Please contact the developers on GitHub."
        target = (folder / fname.name).with_suffix(_PLATFORM_SUFFIXES["windows"])
        move(files_[0], target)

    # clean-up
    fname.unlink()
    rmtree(uncompressed)
    return str(target)


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
    lib.lsl_local_clock.restype = c_double
    lib.lsl_create_streaminfo.restype = c_void_p
    lib.lsl_library_info.restype = c_char_p
    lib.lsl_get_name.restype = c_char_p
    lib.lsl_get_type.restype = c_char_p
    lib.lsl_get_nominal_srate.restype = c_double
    lib.lsl_get_source_id.restype = c_char_p
    lib.lsl_get_created_at.restype = c_double
    lib.lsl_get_uid.restype = c_char_p
    lib.lsl_get_session_id.restype = c_char_p
    lib.lsl_get_hostname.restype = c_char_p
    lib.lsl_get_desc.restype = c_void_p
    lib.lsl_get_xml.restype = c_char_p
    lib.lsl_create_outlet.restype = c_void_p
    lib.lsl_create_inlet.restype = c_void_p
    lib.lsl_get_fullinfo.restype = c_void_p
    lib.lsl_get_info.restype = c_void_p
    lib.lsl_open_stream.restype = c_void_p
    lib.lsl_time_correction.restype = c_double
    lib.lsl_pull_sample_f.restype = c_double
    lib.lsl_pull_sample_d.restype = c_double
    lib.lsl_pull_sample_l.restype = c_double
    lib.lsl_pull_sample_i.restype = c_double
    lib.lsl_pull_sample_s.restype = c_double
    lib.lsl_pull_sample_c.restype = c_double
    lib.lsl_pull_sample_str.restype = c_double
    lib.lsl_pull_sample_buf.restype = c_double
    lib.lsl_first_child.restype = c_void_p
    lib.lsl_first_child.argtypes = [
        c_void_p,
    ]
    lib.lsl_last_child.restype = c_void_p
    lib.lsl_last_child.argtypes = [
        c_void_p,
    ]
    lib.lsl_next_sibling.restype = c_void_p
    lib.lsl_next_sibling.argtypes = [
        c_void_p,
    ]
    lib.lsl_previous_sibling.restype = c_void_p
    lib.lsl_previous_sibling.argtypes = [
        c_void_p,
    ]
    lib.lsl_parent.restype = c_void_p
    lib.lsl_parent.argtypes = [
        c_void_p,
    ]
    lib.lsl_child.restype = c_void_p
    lib.lsl_child.argtypes = [c_void_p, c_char_p]
    lib.lsl_next_sibling_n.restype = c_void_p
    lib.lsl_next_sibling_n.argtypes = [c_void_p, c_char_p]
    lib.lsl_previous_sibling_n.restype = c_void_p
    lib.lsl_previous_sibling_n.argtypes = [c_void_p, c_char_p]
    lib.lsl_name.restype = c_char_p
    lib.lsl_name.argtypes = [
        c_void_p,
    ]
    lib.lsl_value.restype = c_char_p
    lib.lsl_value.argtypes = [
        c_void_p,
    ]
    lib.lsl_child_value.restype = c_char_p
    lib.lsl_child_value.argtypes = [
        c_void_p,
    ]
    lib.lsl_child_value_n.restype = c_char_p
    lib.lsl_child_value_n.argtypes = [c_void_p, c_char_p]
    lib.lsl_append_child_value.restype = c_void_p
    lib.lsl_append_child_value.argtypes = [c_void_p, c_char_p, c_char_p]
    lib.lsl_prepend_child_value.restype = c_void_p
    lib.lsl_prepend_child_value.argtypes = [c_void_p, c_char_p, c_char_p]

    # return type for lsl_set_child_value, lsl_set_name, lsl_set_value is int
    lib.lsl_set_child_value.argtypes = [c_void_p, c_char_p, c_char_p]
    lib.lsl_set_name.argtypes = [c_void_p, c_char_p]
    lib.lsl_set_value.argtypes = [c_void_p, c_char_p]
    lib.lsl_append_child.restype = c_void_p
    lib.lsl_append_child.argtypes = [c_void_p, c_char_p]
    lib.lsl_prepend_child.restype = c_void_p
    lib.lsl_prepend_child.argtypes = [c_void_p, c_char_p]
    lib.lsl_append_copy.restype = c_void_p
    lib.lsl_append_copy.argtypes = [c_void_p, c_void_p]
    lib.lsl_prepend_copy.restype = c_void_p
    lib.lsl_prepend_copy.argtypes = [c_void_p, c_void_p]
    lib.lsl_remove_child_n.argtypes = [c_void_p, c_char_p]
    lib.lsl_remove_child.argtypes = [c_void_p, c_void_p]
    lib.lsl_destroy_string.argtypes = [c_void_p]

    # TODO: Check if the minimum version for MNE-LSL requires those try/except.
    try:
        lib.lsl_pull_chunk_f.restype = c_long
        lib.lsl_pull_chunk_d.restype = c_long
        lib.lsl_pull_chunk_l.restype = c_long
        lib.lsl_pull_chunk_i.restype = c_long
        lib.lsl_pull_chunk_s.restype = c_long
        lib.lsl_pull_chunk_c.restype = c_long
        lib.lsl_pull_chunk_str.restype = c_long
        lib.lsl_pull_chunk_buf.restype = c_long
    except Exception:
        logger.info(
            "[LIBLSL] Chunk transfer functions not available in your liblsl version."
        )
    try:
        lib.lsl_create_continuous_resolver.restype = c_void_p
        lib.lsl_create_continuous_resolver_bypred.restype = c_void_p
        lib.lsl_create_continuous_resolver_byprop.restype = c_void_p
    except Exception:
        logger.info(
            "[LIBLSL] Continuous resolver functions not available in your liblsl "
            "version."
        )

    return lib


# load library
lib = load_liblsl()
