from __future__ import annotations

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

from .._version import __version__
from ..utils._checks import ensure_path
from ..utils._path import walk
from ..utils.logs import logger, warn

if TYPE_CHECKING:
    from typing import Optional, Union

    from pooch import Pooch


# folder where the library is fetched
_LIB_FOLDER: Path = files("mne_lsl.lsl") / "lib"
# minimum liblsl version. The major version is given by version // 100
# and the minor version is given by version % 100.
_VERSION_MIN: int = 115
# liblsl objects created with the same protocol version are inter-compatible.
_VERSION_PROTOCOL: int = 110  # noqa: W0612
_PLATFORM: str = platform.system().lower().strip()
_PLATFORM_SUFFIXES: dict[str, str] = {
    "windows": ".dll",
    "darwin": ".dylib",
    "linux": ".so",
}
# generic error message
_ERROR_MSG: str = (
    "Please visit LIBLSL library github page (https://github.com/sccn/liblsl) and "
    "install a release in the system directories or provide its path in the "
    "environment variable MNE_LSL_LIB or PYLSL_LIB."
)


def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system.

    The library is loaded in the following order:

    1. Search in the environment variables.
    2. Search in the system folder.
    3. Search in _LIB_FOLDER.
    4. Fetch on GitHub.
    """
    if _PLATFORM not in _PLATFORM_SUFFIXES:  # pragma: no cover
        raise RuntimeError(
            "The OS could not be determined. Please open an issue on GitHub and "
            "provide the error traceback to the developers."
        )
    libpath = _load_liblsl_environment_variables()
    libpath = _load_liblsl_system() if libpath is None else libpath
    libpath = _load_liblsl_mne_lsl() if libpath is None else libpath
    libpath = _fetch_liblsl() if libpath is None else libpath
    assert isinstance(libpath, str)  # sanity-check
    lib = CDLL(libpath)
    return _set_types(lib)


def _load_liblsl_environment_variables() -> Optional[str]:
    """Load the binary LSL library from the environment variables.

    Returns
    -------
    libpath : str | None
        Path to the binary LSL library or None if it could not be found.
    """
    for variable in ("MNE_LSL_LIB", "PYLSL_LIB"):
        libpath = os.environ.get(variable, None)
        if libpath is None:
            logger.debug("The environment variable '%s' is not set.", variable)
            continue
        logger.debug(
            "Attempting to load libpath '%s' stored in the environment variable '%s'.",
            libpath,
            variable,
        )
        # even if the path is not valid, we still try to load it and issue a second
        # generic warning 'can not be loaded' if it fails.
        _is_valid_libpath(libpath)
        libpath, version = _attempt_load_liblsl(libpath)
        if version is None:
            continue
        # we do not accept outdated versions from the environment variables.
        if _is_valid_version(libpath, version):
            return libpath
    return None


def _load_liblsl_system() -> Optional[str]:
    """Load the binary LSL library from the system path/folders.

    Returns
    -------
    libpath : str | None
        Path to the binary LSL library or None if it could not be found.
    """
    libpath = find_library("lsl")
    if libpath is None:
        logger.debug("The library LIBLSL is not found in the system folder.")
        return None
    logger.debug(
        "Attempting to load libpath '%s' from the system folders.",
        libpath,
    )
    # no need to validate the path as this is returned by the system directly, so we
    # try to load it and issue a generic warning 'can not be loaded' if it fails.
    libpath, version = _attempt_load_liblsl(libpath)
    if version is None:
        return None
    # we do not accept outdated versions from the system folders.
    if _is_valid_version(libpath, version):
        return libpath
    return None


def _load_liblsl_mne_lsl(*, folder: Path = _LIB_FOLDER) -> Optional[str]:
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
    for libpath in folder.glob(f"*{_PLATFORM_SUFFIXES[_PLATFORM]}"):
        # disable the generic warning 'can not be loaded' in favor of a detailed warning
        # mentioning the file deletion.
        logger.debug("Loading previously downloaded liblsl '%s'.", libpath)
        libpath, version = _attempt_load_liblsl(libpath, issue_warning=False)
        if version is None:
            libpath = ensure_path(libpath, must_exist=False)
            warn(
                f"The previously downloaded LIBLSL '{libpath.name}' in "
                f"'{libpath.parent}' could not be loaded. It will be removed.",
            )
            libpath.unlink(missing_ok=False)
            continue
        # we do not accept outdated versions from the mne-lsl folder and we will remove
        # outdated versions.
        # disable the generic version warning in favor of a detailed warning mentioning
        # the file deletion.
        if _is_valid_version(libpath, version, issue_warning=False):
            return libpath
        libpath = ensure_path(libpath, must_exist=False)
        warn(
            f"The previously downloaded LIBLSL '{libpath.name}' in '{libpath.parent}' "
            f"is outdated. The version is {version // 100}.{version % 100} while the "
            "minimum version required by MNE-LSL is "
            f"{_VERSION_MIN // 100}.{_VERSION_MIN % 100}. It will be removed.",
        )
        libpath.unlink(missing_ok=False)
    return None


def _fetch_liblsl(
    *,
    folder: Union[str, Path] = _LIB_FOLDER,
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
    folder = ensure_path(folder, must_exist=False)
    if folder.is_file():
        raise RuntimeError(
            f"The path '{folder}' is a file. Please provide a directory path."
        )
    # the requests.get() call is likely to fail on CIs with a 403 Forbidden error.
    try:
        response = requests.get(
            url, timeout=15, headers={"user-agent": f"mne-lsl/{__version__}"}
        )
        logger.debug("Response code: %s", response.status_code)
        assets = [elt for elt in response.json()["assets"] if "liblsl" in elt["name"]]
    except Exception:  # pragma: no cover
        raise KeyError("The latest release of liblsl could not be fetch.")
    # filter the assets for our platform
    if _PLATFORM == "linux":
        import distro  # attempt to identify the distribution based on the codename

        assets = [elt for elt in assets if distro.codename() in elt["name"]]
    elif _PLATFORM == "darwin":
        assets = [elt for elt in assets if "OSX" in elt["name"]]
        if platform.processor() == "arm":
            assets = [elt for elt in assets if "arm" in elt["name"]]
            # download v1.16.0 for M1-M2 since liblsl doesn't consistently release a
            # version for arm64 architecture with every bugfix release.
            if len(assets) == 0:
                assets = [
                    dict(
                        name="liblsl-1.16.0-OSX_arm64.tar.bz2",
                        browser_download_url="https://github.com/sccn/liblsl/releases/download/v1.16.0/liblsl-1.16.0-OSX_arm64.tar.bz2",  # noqa: E501
                    )
                ]
        elif platform.processor() == "i386":
            assets = [elt for elt in assets if "amd64" in elt["name"]]
        else:  # pragma: no cover
            raise RuntimeError(
                f"The processor architecture {platform.processor()} could not be "
                "identified. Please open an issue on GitHub and provide the error "
                "traceback to the developers."
            )
    elif _PLATFORM == "windows":
        assets = [elt for elt in assets if "Win" in elt["name"]]
        if sizeof(c_void_p) == 4:  # 32 bits
            assets = [elt for elt in assets if "i386" in elt["name"]]
        elif sizeof(c_void_p) == 8:  # 64 bits
            assets = [elt for elt in assets if "amd64" in elt["name"]]
        else:  # pragma: no cover
            raise RuntimeError(
                "The processor architecture could not be determined from 'c_void_p' "
                f"size {sizeof(c_void_p)}. Please open an issue on GitHub and provide "
                "the error traceback to the developers."
            )
    # at this point, we should have identified a unique asset to download.
    if len(assets) == 0:
        raise RuntimeError(
            "MNE-LSL could not find a liblsl on the github release page which match "
            f"your architecture. {_ERROR_MSG}"
        )
    elif len(assets) != 1:  # pragma: no cover
        raise RuntimeError(
            "MNE-LSL found multiple liblsl on the github release page which match "
            f"your architecture. {_ERROR_MSG}"
        )
    asset = assets[0]
    logger.debug("Fetching liblsl into '%s'.", folder)
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as error:  # pragma: no cover
        logger.exception(error)
        raise RuntimeError(
            f"MNE-LSL could not create the directory '{folder}' in which to download "
            f"LIBLSL for your platform. {_ERROR_MSG}"
        )
    libpath = pooch.retrieve(
        url=asset["browser_download_url"],
        fname=asset["name"],
        path=folder,
        processor=_pooch_processor_liblsl,
        known_hash=None,
    )
    libpath, version = _attempt_load_liblsl(libpath)
    if version is None:  # pragma: no cover
        libpath = ensure_path(libpath, must_exist=False)
        libpath.unlink(missing_ok=True)
        raise RuntimeError(
            f"The downloaded LIBLSL '{libpath.name}' in '{libpath.parent}' could not "
            f"be loaded. It will be removed. {_ERROR_MSG}"
        )
    # we do not accept outdated versions from GitHub, which should not be possible
    # anyway since we fetch the latest release.
    if _is_valid_version(libpath, version, issue_warning=False):
        return libpath
    libpath = ensure_path(libpath, must_exist=False)
    libpath.unlink(missing_ok=True)
    raise RuntimeError(
        f"The downloaded LIBLSL '{libpath.name}' in '{libpath.parent}' "
        f"is outdated. The version is {version // 100}.{version % 100} while the "
        "minimum version required by MNE-LSL is "
        f"{_VERSION_MIN // 100}.{_VERSION_MIN % 100}. It will be removed. {_ERROR_MSG}",
    )


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
    fname = Path(fname)
    uncompressed = fname.with_suffix(".archive")
    logger.debug("Processing '%s' with pooch.", fname)

    if _PLATFORM == "linux" and fname.suffix == ".deb":
        os.makedirs(uncompressed, exist_ok=True)
        result = subprocess.run(["ar", "x", str(fname), "--output", str(uncompressed)])
        if result.returncode != 0:  # pragma: no cover
            # we did not manage to open the '.deb' package, there is not point in
            # attempting to load the lib with CDLL, thus let's raise after deleting the
            # downloaded file.
            fname.unlink(missing_ok=False)
            raise RuntimeError(
                "Could not run 'ar x' command to unpack debian package. Do you have "
                "binutils installed with 'sudo apt install binutils'? Alternatively, "
                "p{_ERROR_MSG[1:]} The downloaded file will be removed.",
            )
        # untar control and data
        with tarfile.open(uncompressed / "control.tar.gz") as archive:
            archive.extractall(uncompressed / "control")
        with tarfile.open(uncompressed / "data.tar.gz") as archive:
            archive.extractall(uncompressed / "data")
        # parse dependencies for logging
        with open(uncompressed / "control" / "control") as file:
            lines = file.readlines()
        lines = [
            line.split("Depends:")[1].strip()
            for line in lines
            if line.startswith("Depends:")
        ]
        if len(lines) != 1:  # pragma: no cover
            warn("Dependencies from debian liblsl package could not be parsed.")
        else:
            logger.info(
                "Attempting to retrieve liblsl from the release page. It requires %s.",
                lines[0],
            )
        # find and move the library
        for file in walk(uncompressed / "data"):
            if file.is_symlink() or file.parent.name != "lib":
                continue
            break
        target = fname.with_suffix(_PLATFORM_SUFFIXES["linux"])
        logger.debug("Moving '%s' to '%s'.", file, target)
        move(file, target)

    elif _PLATFORM == "linux":  # pragma: no cover
        return str(fname)  # let's try to load it and hope for the best

    elif _PLATFORM == "darwin":
        with tarfile.open(fname, "r:bz2") as archive:
            archive.extractall(uncompressed)
        # find and move the library
        for file in walk(uncompressed):
            if file.is_symlink() or file.parent.name != "lib":
                continue
            break
        target = (
            fname.parent
            / f"{fname.name.split('.tar.bz2')[0]}{_PLATFORM_SUFFIXES['darwin']}"
        )
        logger.debug("Moving '%s' to '%s'.", file, target)
        move(file, target)

    elif _PLATFORM == "windows":
        with zipfile.ZipFile(fname, "r") as archive:
            archive.extractall(uncompressed)
        # find and move the library
        for file in walk(uncompressed):
            if (
                file.suffix != _PLATFORM_SUFFIXES["windows"]
                or file.parent.name != "bin"
            ):
                continue
            break
        target = fname.with_suffix(_PLATFORM_SUFFIXES["windows"])
        logger.debug("Moving '%s' to '%s'.", file, target)
        move(file, target)

    # clean-up
    fname.unlink()
    rmtree(uncompressed)
    return str(target)


def _is_valid_libpath(libpath: str) -> bool:
    """Check if the library path is valid."""
    assert isinstance(libpath, str)  # sanity-check
    libpath = ensure_path(libpath, must_exist=False)
    if libpath.suffix != _PLATFORM_SUFFIXES[_PLATFORM]:
        warn(
            f"The LIBLSL '{libpath}' ends with '{libpath.suffix}' which is "
            f"different from the expected extension '{_PLATFORM_SUFFIXES[_PLATFORM]}' "
            f"for {_PLATFORM} based OS."
        )
        return False
    if not libpath.exists():
        warn(f"The LIBLSL '{libpath}' does not exist.")
        return False
    return True


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
    libpath = str(libpath) if isinstance(libpath, Path) else libpath
    try:
        lib = CDLL(libpath)
        version = lib.lsl_library_version()
        del lib
    except OSError:
        version = None
        if issue_warning:
            warn(f"The LIBLSL '{libpath}' can not be loaded.")
    return libpath, version


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
    assert isinstance(libpath, str)  # sanity-check
    assert isinstance(version, int)  # sanity-check
    if version < _VERSION_MIN:
        if issue_warning:
            warn(
                f"The LIBLSL '{libpath}' is outdated. The version is "
                f"{version // 100}.{version % 100} while the minimum version required "
                f"by MNE-LSL is {_VERSION_MIN // 100}.{_VERSION_MIN % 100}."
            )
        return False
    return True


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
    except Exception:  # pragma: no cover
        logger.info(
            "[LIBLSL] Chunk transfer functions not available in your LIBLSL version."
        )
    try:
        lib.lsl_create_continuous_resolver.restype = c_void_p
        lib.lsl_create_continuous_resolver_bypred.restype = c_void_p
        lib.lsl_create_continuous_resolver_byprop.restype = c_void_p
    except Exception:  # pragma: no cover
        logger.info(
            "[LIBLSL] Continuous resolver functions not available in your LIBLSL "
            "version."
        )

    return lib


# load library
lib = load_liblsl()
