from __future__ import annotations

import os
import platform
from ctypes import CDLL, c_char_p, c_double, c_long, c_void_p
from importlib.resources import files

from ..utils._checks import ensure_path
from ..utils.logs import logger, warn

# minimum recommended liblsl version. The major version is given by version // 100
# and the minor version is given by version % 100.
_VERSION_MIN: int = 115
# liblsl objects created with the same protocol version are inter-compatible.
_VERSION_PROTOCOL: int = 110  # noqa: W0612


def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system.

    The library is loaded in the following order:

    1. Search in the environment variables.
    2. Search in the system folder.
    3. Search in the defined library folder.
    4. Fetch on GitHub.
    """
    libpath = _load_liblsl_environment_variables()
    libpath = _load_liblsl_wheel_path()
    assert isinstance(libpath, str)  # sanity-check
    lib = CDLL(libpath)
    _set_types(lib)
    return lib


def _load_liblsl_environment_variables(
    *, version_min: int = _VERSION_MIN
) -> str | None:
    """Load the binary LSL library from the environment variables.

    Parameters
    ----------
    version_min : int
        Minimum version of the LSL library.

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
        libpath = ensure_path(libpath, must_exist=False)
        try:
            lib = CDLL(libpath)
            version = lib.lsl_library_version()
        except OSError:
            continue
        # we warn for outdated versions
        if version < version_min:
            warn(
                f"The LIBLSL '{libpath}' (version {version // 100}.{version % 100}) is "
                "outdated, use at your own discretion. MNE-LSL recommends to use "
                f"version {version_min // 100}.{version_min % 100} and above."
            )
        return libpath
    return None


def _load_liblsl_wheel_path() -> str:
    """Load the binary LSL library from the wheel path.

    Returns
    -------
    libpath : str
        Path to the binary LSL library bundled with mne-lsl.
    """
    libpath: str | None = None
    if platform.system() == "Linux":
        # auditwheel will relocate and mangle, e.g.:
        # mne_lsl/../mne_lsl.libs/liblsl-65106c22.so.1.16.2
        libs = files("mne_lsl").parent / "mne_lsl.libs"
        lib_files = list(libs.glob("liblsl*.so*"))
        if len(lib_files) != 1:
            raise RuntimeError(
                f"Could not find the LIBLSL library bundle with mne-lsl in '{libs}'."
            )
        libpath = lib_files[0]
    elif platform.system() == "Windows":
        # delvewheel has similar behavior to auditwheel
        libs = files("mne_lsl").parent / "mne_lsl.libs"
        lib_files = list(libs.glob("lsl*.dll"))
        if len(lib_files) != 1:
            raise RuntimeError(
                f"Could not find the LIBLSL library bundle with mne-lsl in '{libs}'."
            )
        libpath = lib_files[0]
    elif platform.system() == "Darwin":
        libs = files("mne_lsl") / ".dylibs"
        lib_files = list(libs.glob("liblsl*.dylib"))
        if len(lib_files) != 1:
            raise RuntimeError(
                f"Could not find the LIBLSL library bundle with mne-lsl in '{libs}'."
            )
        libpath = lib_files[0]
    else:
        raise RuntimeError(
            f"Unsupported platform {platform.system()}. Please use the environment "
            "variable MNE_LSL_LIB or PYLSL_LIB to provide the path to LIBLSL."
        )
    logger.debug("Found wheel path '%s'.", libpath)
    return str(libpath)


def _set_types(lib: CDLL) -> None:
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


# load library
lib = load_liblsl()
