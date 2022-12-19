from .load_liblsl import lib


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


def protocol_version() -> int:
    """Version of the LSL protocol.

    Returns
    -------
    version : int
        Version of the binary LSL library.
        The major version is version // 100.
        The minor version is version % 100.

    Notes
    -----
    Clients with different minor versions are protocol-compatible with each
    other, while clients with different major versions will refuse to work
    together.
    """
    return lib.lsl_protocol_version()


def local_clock() -> float:
    """Obtain a local system timestamp in seconds.

    Returns
    -------
    time : int
        Local timestamp in seconds.
    """
    return lib.lsl_local_clock()
