from __future__ import annotations

from ctypes import byref, c_char_p, c_double, c_void_p
from typing import TYPE_CHECKING

from ..utils._checks import check_type, ensure_int, ensure_path
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo

if TYPE_CHECKING:
    from pathlib import Path


def library_version() -> int:
    """Version of the binary LSL library.

    Returns
    -------
    version : int
        Version of the binary LSL library.
        The major version is ``version // 100``.
        The minor version is ``version % 100``.
    """
    return lib.lsl_library_version()


def protocol_version() -> int:
    """Version of the LSL protocol.

    Returns
    -------
    version : int
        Version of the binary LSL library.
        The major version is ``version // 100``.
        The minor version is ``version % 100``.

    Notes
    -----
    Clients with different minor versions are protocol-compatible with each other, while
    clients with different major versions will refuse to work together.
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


def set_config_filename(filename: str | Path) -> None:
    """Set a custom configuration file for liblsl.

    Override the file from which liblsl loads its configuration. By default, liblsl
    looks for a configuration file in the current working directory, in the home
    directory, and in the system-wide configuration directory (see Network Connectivity
    in the LSL wiki).

    Parameters
    ----------
    filename : str | Path
        Path to the configuration file to load.

    Notes
    -----
    This function must be called before any other liblsl operation, e.g. before creating
    a :class:`~mne_lsl.lsl.StreamInfo`, :class:`~mne_lsl.lsl.StreamInlet` or
    :class:`~mne_lsl.lsl.StreamOutlet`, as the configuration is loaded once on the first
    call requiring it. It requires ``liblsl >= 1.17.7``.
    """
    filename = ensure_path(filename, must_exist=True)
    if not hasattr(lib, "lsl_set_config_filename"):
        raise NotImplementedError(
            "The function 'set_config_filename' requires liblsl >= 1.17.7, while "
            f"version {library_version()} is loaded."
        )
    lib.lsl_set_config_filename(c_char_p(str.encode(str(filename))))


def set_config_content(content: str) -> None:
    """Set the liblsl configuration from an in-memory string.

    Provide the configuration content directly as a string, e.g. for platforms where
    configuration files or environment variables are not convenient (see Network
    Connectivity in the LSL wiki).

    Parameters
    ----------
    content : str
        Content of the configuration, in the same format as a configuration file.

    Notes
    -----
    This function must be called before any other liblsl operation, e.g. before creating
    a :class:`~mne_lsl.lsl.StreamInfo`, :class:`~mne_lsl.lsl.StreamInlet` or
    :class:`~mne_lsl.lsl.StreamOutlet`, as the configuration is loaded once on the first
    call requiring it. It requires ``liblsl >= 1.17.7``.
    """
    check_type(content, (str,), "content")
    if not hasattr(lib, "lsl_set_config_content"):
        raise NotImplementedError(
            "The function 'set_config_content' requires liblsl >= 1.17.7, while "
            f"version {library_version()} is loaded."
        )
    lib.lsl_set_config_content(c_char_p(str.encode(content)))


def resolve_streams(
    timeout: float = 1.0,
    name: str | None = None,
    stype: str | None = None,
    source_id: str | None = None,
    minimum: int = 1,
) -> list[_BaseStreamInfo]:
    """Resolve streams on the network.

    This function returns all currently available streams from any outlet on the
    network. The network is usually the subnet specified at the local router, but may
    also include a group of machines visible to each other via multicast packets (given
    that the network supports it), or list of hostnames. These details may optionally be
    customized by the experimenter in a configuration file (see Network Connectivity in
    the LSL wiki).

    Parameters
    ----------
    timeout : float
        Timeout (in seconds) of the operation. If this is too short (e.g.
        ``< 0.5 seconds``) only a subset (or none) of the outlets that are present on
        the network may be returned.
    name : str | None
        Restrict the selected streams to this name.
    stype : str | None
        Restrict the selected stream to this type.
    source_id : str | None
        Restrict the selected stream to this source ID.
    minimum : int
        Minimum number of stream to return when restricting the selection. As soon as
        this minimum is hit, the search will end. Only works if at least one of the 3
        identifiers ``name``, ``stype`` or ``source_id`` is not ``None``.

    Returns
    -------
    sinfos : list
        List of :class:`~mne_lsl.lsl.StreamInfo` objects found on the network. While a
        :class:`~mne_lsl.lsl.StreamInfo` is not bound to an Inlet, the description field
        remains empty.
    """
    check_type(timeout, ("numeric",), "timeout")
    if timeout <= 0:
        raise ValueError(
            "The argument 'timeout' must be a strictly positive integer. "
            f"{timeout} is invalid."
        )
    properties = (name, stype, source_id)
    buffer = (c_void_p * 1024)()
    if all(prop is None for prop in properties):
        num_found = lib.lsl_resolve_all(byref(buffer), 1024, c_double(timeout))
        streams = [_BaseStreamInfo(buffer[k]) for k in range(num_found)]
        streams = list(set(streams))  # remove duplicates
        return streams
    minimum = ensure_int(minimum, "minimum")
    if minimum <= 0:
        raise ValueError(
            "The argument 'minimum' must be a strictly positive integer. "
            f"Provided '{minimum}' is invalid."
        )

    properties = [
        # filter out the properties set to None
        (prop, name)
        for prop, name in zip(properties, ("name", "stype", "source_id"), strict=True)
        if prop is not None
    ]
    timeout /= len(properties)

    streams = []
    for prop, name in properties:
        check_type(prop, (str,), name)
        # rename properties for lsl compatibility
        name = "type" if name == "stype" else name
        num_found = lib.lsl_resolve_byprop(
            byref(buffer),
            1024,
            c_char_p(str.encode(name)),
            c_char_p(str.encode(prop)),
            minimum,
            c_double(timeout),
        )
        new_streams = [_BaseStreamInfo(buffer[k]) for k in range(num_found)]
        # now delete the ones that don't have all the correct properties
        stream2delete = list()
        for k, stream in enumerate(new_streams):
            for prop, name in properties:
                if getattr(stream, name) != prop:
                    stream2delete.append(k)
                    break
        for idx in stream2delete[::-1]:
            del new_streams[idx]
        streams.extend(new_streams)
        if minimum <= len(streams):
            break
    return list(set(streams))  # remove duplicates
