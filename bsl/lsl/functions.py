from __future__ import annotations  # c.f. PEP 563, PEP 649

from ctypes import byref, c_char_p, c_double, c_void_p
from typing import TYPE_CHECKING

from ..utils._checks import check_type, ensure_int
from .load_liblsl import lib
from .stream_info import _BaseStreamInfo

if TYPE_CHECKING:
    from typing import List, Optional


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


def resolve_streams(
    timeout: float = 1.0,
    name: Optional[str] = None,
    stype: Optional[str] = None,
    source_id: Optional[str] = None,
    minimum: int = 1,
) -> List[_BaseStreamInfo]:
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
        Minimum number of stream to return where restricting the selection. As soon as
        this minimum is hit, the search will end. Only works if at least one of the 3
        identifiers ``name``, ``stype`` or ``source_id`` is not ``None``.

    Returns
    -------
    sinfos : list
        List of :class:`~bsl.lsl.StreamInfo` objects found on the network. While a
        :class:`~bsl.lsl.StreamInfo` is not bound to an Inlet, the description field
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
        for prop, name in zip(properties, ("name", "stype", "source_id"))
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
        # now delete the ones that dn't have all the correct properties
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
