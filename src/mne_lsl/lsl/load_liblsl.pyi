from ctypes import CDLL

from _typeshed import Incomplete

from ..utils.logs import logger as logger
from ..utils.logs import warn as warn

_VERSION_MIN: int
_VERSION_PROTOCOL: int

def load_liblsl() -> CDLL:
    """Load the binary LSL library on the system.

    The library is loaded in the following order:

    1. Search in the environment variables.
    2. Search in the system folder.
    3. Search in the defined library folder.
    4. Fetch on GitHub.
    """

def _load_liblsl_environment_variables(*, version_min: int = ...) -> str | None:
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

def _load_liblsl_wheel_path() -> str:
    """Load the binary LSL library from the wheel path.

    Returns
    -------
    libpath : str
        Path to the binary LSL library bundled with mne-lsl.
    """

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

lib: Incomplete
