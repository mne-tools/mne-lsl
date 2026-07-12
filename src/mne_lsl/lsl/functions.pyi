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

def set_config_filename(filename: str) -> None:
    """Set a custom configuration file for liblsl.

    Override the file from which liblsl loads its configuration. By default, liblsl
    looks for a configuration file in the current working directory, in the home
    directory, and in the system-wide configuration directory (see Network Connectivity
    in the LSL wiki).

    Parameters
    ----------
    filename : str
        Path to the configuration file to load.

    Notes
    -----
    This function must be called before any other liblsl operation, e.g. before creating
    a :class:`~mne_lsl.lsl.StreamInfo`, :class:`~mne_lsl.lsl.StreamInlet` or
    :class:`~mne_lsl.lsl.StreamOutlet`, as the configuration is loaded once on the first
    call requiring it. It requires ``liblsl >= 1.17.7``.
    """

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
