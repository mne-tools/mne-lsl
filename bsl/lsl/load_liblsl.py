import os
import platform
from ctypes import CDLL, c_char_p, c_double, c_long, c_void_p, sizeof
from pathlib import Path
from typing import Optional, Tuple, Union

from ..utils._exceptions import _GH_ISSUES
from ..utils._logs import logger

# Minimum/Maximum liblsl version. The major version is given by version // 100
# and the minor version is given by version % 100.
_VERSION_MIN = 115
_VERSION_MAX = 116
# Description of the current version packaged with BSL: 1.16.0
_VERSION_BSL = 116
_VERSION_BSL_PATCH = 0
# liblsl objects created with the same protocol version are inter-compatible.
_VERSION_PROTOCOL = 110

_PLATFORM_SUFFIXES = {
    "Windows": ".dll",
    "Darwin": ".dylib",
    "Linux": ".so",
}
_SUPPORTED_DISTRO = {
    "ubuntu": ("20.04", "22.04"),
}


def load_liblsl():
    """Load the binary LSL library on the system."""
    if platform.system() not in _PLATFORM_SUFFIXES:
        raise RuntimeError("The OS could not be determined. " + _GH_ISSUES)

    lib = _find_liblsl_env()
    if lib is not None:
        return _set_return_types(lib)
    lib = _find_liblsl_bsl()
    if lib is not None:
        return _set_return_types(lib)
    else:
        raise RuntimeError(
            "The liblsl library packaged with BSL could not be loaded. " + _GH_ISSUES
        )


def _find_liblsl_env() -> Optional[CDLL]:
    """Search for the LSL library in the environment variable LSL_LIB.

    Returns
    -------
    lib : CDLL | None
        Loaded binary LSL library. None if the value retrieved in the
        environment variable was not valid or yielded an invalid library.
    """
    if "LSL_LIB" not in os.environ:
        return None

    libpath = Path(os.environ["LSL_LIB"])
    if libpath.suffix != _PLATFORM_SUFFIXES[platform.system()]:
        logger.error(
            "The LIBLSL '%s' provided in the environment variable "
            "'LSL_LIB' ends with '%s' which is different from the expected "
            "extension for this OS.",
            libpath,
            libpath.suffix,
        )
        return None
    if libpath.exists():
        libpath, version = _attempt_load_liblsl(libpath)
        if version is None:
            logger.error(
                "The LIBLSL '%s' provided in the environment variable "
                "'LSL_LIB' can not be loaded.",
                libpath,
            )
        elif version < _VERSION_MIN:
            logger.error(
                "The LIBLSL '%s' provided in the environment variable "
                "'LSL_LIB' is outdated. The version is %i.%i while the "
                "minimum version required by BSL is %i.%i.",
                libpath,
                version // 100,
                version % 100,
                _VERSION_MIN // 100,
                _VERSION_MIN % 100,
            )
            version = None
        elif _VERSION_MAX < version:
            logger.warning(
                "The LIBLSL '%s' provided in the environment variable "
                "'LSL_LIB' is not officially supported. The version is %i.%i "
                "while the maximum supported version required by BSL is "
                "%i.%i. Use this version at your own risk.",
                libpath,
                version // 100,
                version % 100,
                _VERSION_MIN // 100,
                _VERSION_MIN % 100,
            )
    else:
        logger.error(
            "The LIBLSL path '%s' provided in the environment variable "
            "'LSL_LIB' does not exists.",
            libpath,
        )
        libpath = None
        version = None
    if version is not None:
        assert libpath is not None  # sanity-check
        lib = CDLL(libpath)
    else:
        lib = None
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


def _find_liblsl_bsl() -> Optional[CDLL]:
    """Search for the LSL library packaged with BSL."""
    libname = (
        f"liblsl-{_VERSION_BSL // 100}"
        + f".{_VERSION_BSL % 100}"
        + f".{_VERSION_BSL_PATCH}-"
    )

    # check linux distribution
    if platform.system() == "Linux":
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
                    "The liblsl library packaged with BSL supports "
                    f"{', '.join(_SUPPORTED_DISTRO)} based distributions. "
                    f"{distro.name()} is not supported. Please build the "
                    "liblsl library from source and provide it in the "
                    "environment variable LSL_LIB."
                )
        if distro.version() not in _SUPPORTED_DISTRO[distro_like]:
            raise RuntimeError(
                "The liblsl library packaged with BSL supports distro_like "
                "based distributions on versions "
                f"{', '.join(_SUPPORTED_DISTRO[distro_like])}. Version "
                f"{distro.version()} is not supported. Please build the "
                "liblsl library from source and provide it in the environment "
                "variable LSL_LIB."
            )
        libname += f"{distro.version()}_amd64.so"

    # check macOS intel vs arm
    elif platform.system() == "Darwin":
        if platform.processor() == "arm":
            libname += "OSX_arm64.dylib"
        elif platform.processor() == "i386":
            libname += "OSX_amd64.dylib"
        else:
            raise RuntimeError(
                "The processor architecture could not be determined. " + _GH_ISSUES
            )

    # check windows 32 vs 64 bits
    elif platform.system() == "Windows":
        if sizeof(c_void_p) == 4:  # 32 bits
            libname += "Win_i386.dll"
        elif sizeof(c_void_p) == 8:  # 64 bits
            libname += "Win_amd64.dll"
        else:
            raise RuntimeError(
                "The processor architecture could not be determined. " + _GH_ISSUES
            )

    # attempt to load the corresponding liblsl
    libpath = Path(__file__).parent / "lib" / libname
    try:
        lib = CDLL(str(libpath))
        assert _VERSION_MIN <= lib.lsl_library_version() <= _VERSION_MAX
    except Exception:
        lib = None
    return lib


def _set_return_types(lib: CDLL) -> CDLL:
    """Set the return types for the different liblsl functions.

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

    # TODO: Check if the minimum version for BSL requires those try/except.
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
            "[LIBLSL] Chunk transfer functions not available in your liblsl " "version."
        )
    try:
        lib.lsl_create_continuous_resolver.restype = c_void_p
        lib.lsl_create_continuous_resolver_bypred.restype = c_void_p
        lib.lsl_create_continuous_resolver_byprop.restype = c_void_p
    except Exception:
        logger.info(
            "[LIBLSL] Continuous resolver functions not available in your "
            "liblsl version."
        )

    return lib


# load library
lib = load_liblsl()
