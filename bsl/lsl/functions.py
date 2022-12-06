from ctypes import byref, c_double, c_void_p

from ..utils._checks import _check_type
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo


def library_version() -> int:
    """Version of the binary LSL library.

    Returns
    -------
    version : int
        Version of the binary LSL library.
        The major version is version // 100.
        The minor version is version % 100.
    """
    return lib.lsl_library_version()


def local_clock() -> int:
    """Obtain a local system timestamp in seconds.

    Returns
    -------
    time : int
        Local timestamp in seconds.
    """
    return lib.lsl_local_clock()


def resolve_streams(timeout: float = 1.0):
    """Resolve all streams on the network.

    This function returns all currently available streams from any outlet on
    the network. The network is usually the subnet specified at the local
    router, but may also include a group of machines visible to each other via
    multicast packets (given that the network supports it), or list of
    hostnames. These details may optionally be customized by the experimenter
    in a configuration file (see Network Connectivity in the LSL wiki).

    Parameters
    ----------
    timeout : float
        Timeout (in seconds) of the operation. If this is too short (e.g.
        ``< 0.5 seconds``) only a subset (or none) of the outlets that are
        present on the network may be returned.

    Returns
    -------
    sinfos : list
        List of `~bsl.lsl.StreamInfo` objects found on the network.
        While a `~bsl.lsl.StreamInfo` is not bound to an Inlet, the description
        field remains empty.
    """
    _check_type(timeout, ("numeric",), "timeout")
    if timeout <= 0:
        raise ValueError(
            "The argument 'timeout' must be a strictly positive integer. "
            f"{timeout} is invalid."
        )
    buffer = (c_void_p * 1024)()
    num_found = lib.lsl_resolve_all(byref(buffer), 1024, c_double(timeout))
    return [_BaseStreamInfo(buffer[k]) for k in range(num_found)]
