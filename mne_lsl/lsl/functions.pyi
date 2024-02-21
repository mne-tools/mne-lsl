from typing import Optional

from ..utils._checks import check_type as check_type
from ..utils._checks import ensure_int as ensure_int
from .load_liblsl import lib as lib
from .stream_info import _BaseStreamInfo as _BaseStreamInfo

def library_version() -> int:
    """Version of the binary LSL library.

    Returns
    -------
    version : int
        Version of the binary LSL library.
        The major version is ``version // 100``.
        The minor version is ``version % 100``.
    """

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

def local_clock() -> float:
    """Obtain a local system timestamp in seconds.

    Returns
    -------
    time : int
        Local timestamp in seconds.
    """

def resolve_streams(
    timeout: float = 1.0,
    name: Optional[str] = None,
    stype: Optional[str] = None,
    source_id: Optional[str] = None,
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
        Minimum number of stream to return where restricting the selection. As soon as
        this minimum is hit, the search will end. Only works if at least one of the 3
        identifiers ``name``, ``stype`` or ``source_id`` is not ``None``.

    Returns
    -------
    sinfos : list
        List of :class:`~mne_lsl.lsl.StreamInfo` objects found on the network. While a
        :class:`~mne_lsl.lsl.StreamInfo` is not bound to an Inlet, the description field
        remains empty.
    """
