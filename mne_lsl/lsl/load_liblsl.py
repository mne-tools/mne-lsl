from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
import platform
import subprocess
import tarfile
import zipfile
from ctypes import CDLL, c_char_p, c_double, c_long, c_void_p, sizeof
from ctypes.util import find_library
from importlib.resources import files
from pathlib import Path
from shutil import move, rmtree
from typing import TYPE_CHECKING

import pooch
import requests

from .. import __version__
from ..utils._path import walk
from ..utils.logs import logger

if TYPE_CHECKING:
    from typing import Optional, Union

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
# generic error message
_ERROR_MSG = (
    "Please visit liblsl library github page (https://github.com/sccn/liblsl) and "
    "install a release in the system directories or provide its path in the "
    "environment variable MNE_LSL_LIB or PYLSL_LIB."
)


def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system."""
    if _PLATFORM not in _PLATFORM_SUFFIXES:
        raise RuntimeError(
            "The OS could not be determined. Please open an issue on GitHub and "
            "provide the error traceback to the developers."
        )
    lib = _find_liblsl()
    if lib is None:
        lib = _fetch_liblsl()
    return _set_types(lib)


def _find_liblsl() -> Optional[CDLL]:
    """Search for liblsl in the environment variable and in the system folders.

    Returns
    -------
    lib : CDLL | None
        Loaded binary LSL library. None if not found.
    """
    for libpath in (
        os.environ.get("MNE_LSL_LIB", None),
        os.environ.get("PYLSL_LIB", None),
        find_library("lsl"),
    ):
        if libpath is None:
            continue

        logger.debug("Attempting to load libpath %s", libpath)
        if _PLATFORM == "linux":
            # for linux, find_library does not return an absolute path, so we can not
            # try to triage based on the libpath.
            libpath, version = _attempt_load_liblsl(libpath)
        else:
            libpath = Path(libpath)
            if libpath.suffix != _PLATFORM_SUFFIXES[_PLATFORM]:
                logger.warning(
                    "The liblsl '%s' ends with '%s' which is different from the "
                    "expected extension '%s' for this OS.",
                    libpath,
                    libpath.suffix,
                    _PLATFORM_SUFFIXES[_PLATFORM],
                )
                continue
            if not libpath.exists():
                logger.warning("The LIBLSL '%s' does not exist.", libpath)
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


def _fetch_liblsl() -> Optional[CDLL]:
    """Fetch liblsl on the release page.

    Returns
    -------
    lib : CDLL | None
        Loaded binary LSL library. None if not found for this platform.
    """
    try:
        response = requests.get(
            "https://api.github.com/repos/sccn/liblsl/releases/latest",
            timeout=15,
            headers={"user-agent": f"mne-lsl/{__version__}"},
        )
        logger.debug("Response code: %s", response.status_code)
        assets = [elt for elt in response.json()["assets"] if "liblsl" in elt["name"]]
    except Exception as error:
        logger.exception(error)
        raise RuntimeError("The latest release of liblsl could not be fetch.")
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
                    f"{distro.name()} is not supported. " + _ERROR_MSG
                )
        if distro.version() not in _SUPPORTED_DISTRO[distro_like]:
            raise RuntimeError(
                "The liblsl library released on GitHub supports "
                f"{', '.join(_SUPPORTED_DISTRO)} based distributions on versions "
                f"{', '.join(_SUPPORTED_DISTRO[distro_like])}. Version "
                f"{distro.version()} is not supported. " + _ERROR_MSG
            )
        # TODO: check that POP_OS! distro.codename() does match Ubuntu codenames, else
        # we also need a mpping between the version and the ubuntu codename.
        assets = [elt for elt in assets if distro.codename() in elt["name"]]

    elif _PLATFORM == "darwin":
        assets = [elt for elt in assets if "OSX" in elt["name"]]
        if platform.processor() == "arm":
            assets = [elt for elt in assets if "arm" in elt["name"]]
            # TODO: fix for M1-M2 while liblsl doesn't consistently release a version
            # for arm64 architecture with every release.
            if len(assets) == 0:
                assets = [
                    dict(
                        name="liblsl-1.16.0-OSX_arm64.tar.bz2",
                        browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.16.0/liblsl-1.16.0-OSX_arm64.tar.bz2",  # noqa: E501
                    )
                ]
        elif platform.processor() == "i386":
            assets = [elt for elt in assets if "amd64" in elt["name"]]
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

    if len(assets) == 0:
        raise RuntimeError(
            "MNE-LSL could not find a liblsl on the github release page which match "
            "your architecture. " + _ERROR_MSG
        )
    elif len(assets) != 1:
        raise RuntimeError(
            "MNE-LSL found multiple liblsl on the github release page which match "
            "your architecture. " + _ERROR_MSG
        )

    asset = assets[0]
    folder = files("mne_lsl.lsl") / "lib"
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as error:
        logger.exception(error)
        raise RuntimeError(
            "MNE-LSL could not create the directory 'lib' in which to download liblsl "
            "for your platform. " + _ERROR_MSG
        )
    libpath = (folder / asset["name"]).with_suffix(_PLATFORM_SUFFIXES[_PLATFORM])
    if libpath.exists():
        _, version = _attempt_load_liblsl(libpath)
        if version is None:
            logger.warning(
                "Previously downloaded liblsl '%s' could not be loaded. It will be "
                "removed and downloaded again."
            )
            libpath.unlink(missing_ok=False)
        else:
            return CDLL(str(libpath))

    # liblsl was not already present in mne_lsl/lsl/lib, thus we need to download it
    libpath = pooch.retrieve(
        url=asset["browser_download_url"],
        fname=asset["name"],
        path=libpath.parent,
        processor=_pooch_processor_liblsl,
        known_hash=None,
    )
    libpath, version = _attempt_load_liblsl(libpath)
    if version is None:
        Path(libpath).unlink()
        raise RuntimeError(
            f"MNE-LSL could not load the downloaded liblsl '{libpath}'. " + _ERROR_MSG
        )
    lib = CDLL(libpath)
    return lib


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
    folder = files("mne_lsl.lsl") / "lib"
    fname = Path(fname)
    uncompressed = folder / f"{fname.name}.archive"

    if _PLATFORM == "linux" and fname.suffix == ".deb":
        os.makedirs(uncompressed, exist_ok=True)
        result = subprocess.run(["ar", "x", str(fname), "--output", str(uncompressed)])
        if result.returncode != 0:
            logger.warning(
                "Could not run 'ar x' command to unpack debian package. Do you have "
                "binutils installed with 'sudo apt install binutils'?"
            )
            return str(fname)

        # untar control and data
        with tarfile.open(uncompressed / "control.tar.gz") as archive:
            archive.extractall(uncompressed / "control")
        with tarfile.open(uncompressed / "data.tar.gz") as archive:
            archive.extractall(uncompressed / "data")

        # parse dependencies for logging information
        with open(uncompressed / "control" / "control", "r") as file:
            lines = file.readlines()
        lines = [
            line.split("Depends:")[1].strip()
            for line in lines
            if line.startswith("Depends:")
        ]
        if len(lines) != 1:
            logger.warning(
                "Dependencies from debian liblsl package could not be parsed."
            )
        else:
            logger.info(
                "Attempting to retrieve liblsl from the release page. It requires %s.",
                lines[0],
            )

        for file in walk(uncompressed / "data"):
            if file.is_symlink() or file.parent.name != "lib":
                continue
            break
        target = (folder / fname.name).with_suffix(_PLATFORM_SUFFIXES["linux"])
        move(file, target)

    elif _PLATFORM == "linux":
        return str(fname)  # let's try to load it and hope for the best

    elif _PLATFORM == "darwin":
        with tarfile.open(fname, "r:bz2") as archive:
            archive.extractall(uncompressed)
        for file in walk(uncompressed):
            if file.is_symlink() or file.parent.name != "lib":
                continue
            break
        target = (
            folder / f"{fname.name.split('.tar.bz2')[0]}{_PLATFORM_SUFFIXES['darwin']}"
        )
        move(file, target)

    elif _PLATFORM == "windows":
        with zipfile.ZipFile(fname, "r") as archive:
            archive.extractall(uncompressed)
        for file in walk(uncompressed):
            if (
                file.suffix != _PLATFORM_SUFFIXES["windows"]
                or file.parent.name != "bin"
            ):
                continue
            break
        target = (folder / fname.name).with_suffix(_PLATFORM_SUFFIXES["windows"])
        move(file, target)

    # clean-up
    fname.unlink()
    rmtree(uncompressed)
    return str(target)


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
    libpath = str(libpath) if isinstance(libpath, Path) else libpath
    try:
        lib = CDLL(libpath)
        version = lib.lsl_library_version()
        del lib
    except OSError:
        version = None
    return libpath, version


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
