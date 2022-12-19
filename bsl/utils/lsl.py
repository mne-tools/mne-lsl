"""LSL wrapper functions for creating a server and a client."""

import xml.etree.ElementTree as ET

from ..lsl import StreamInlet, resolve_streams
from . import Timer
from ._checks import _check_type
from ._logs import logger


def list_lsl_streams(ignore_markers=False):
    """List all the available outlets on LSL network.

    Parameters
    ----------
    ignore_markers : bool
        If ``True``, ignore streams with Marker type.

    Returns
    -------
    stream_list : list
        List of the found stream name.
    streamInfos : list
        List of the corresponding `~bsl.lsl.StreamInfo`.
    """
    _check_type(ignore_markers, (bool,), item_name="ignore_markers")

    stream_list = []
    streamInfos = resolve_streams()

    if len(streamInfos) == 0:
        return stream_list, []

    for index, streamInfo in enumerate(streamInfos):
        if ignore_markers and "Markers" in streamInfo.stype:
            continue
        stream_list.append(streamInfo.name)

    if ignore_markers:
        streamInfos = [
            streamInfo
            for streamInfo in streamInfos
            if "Markers" not in streamInfo.stype
        ]

    return stream_list, streamInfos


def search_lsl(ignore_markers=False, timeout=10):
    """Search and select an available stream on LSL network.

    This function does not open an LSL inlet.

    Parameters
    ----------
    ignore_markers : bool
        If ``True``, ignore streams with Marker type.
    timeout : int
        Timeout duration in seconds after which the search of an LSL stream
        is interrupted.

    Returns
    -------
    stream_name : str
        Selected stream name.
    """
    _check_type(ignore_markers, (bool,), item_name="ignore_markers")
    _check_type(timeout, ("numeric",), item_name="timeout")
    assert 0 < timeout

    watchdog = Timer()
    while watchdog.sec() <= timeout:
        stream_list, streamInfos = list_lsl_streams(ignore_markers)
        if len(stream_list) != 0:
            break
    else:
        logger.error("Timeout. No LSL stream found.")
        return None

    logger.info("-- List of servers --")
    for i, stream_name in enumerate(stream_list):
        logger.info("%i: %s", i, stream_name)

    if len(stream_list) == 0:
        logger.error("No LSL stream found on the network.")
    elif len(stream_list) == 1:
        index = 0
    else:
        index = input(
            "Stream index? "
            "Hit enter without index to select the first server.\n>> "
        )
        if index.strip() == "":
            index = 0
        else:
            index = int(index.strip())

    stream_name = stream_list[index]
    streamInfo = streamInfos[index]
    assert stream_name == streamInfo.name

    logger.info("Selected: %s", stream_name)

    return stream_name


def lsl_channel_list(inlet):
    """Extract the channels name list from the LSL info.

    Parameters
    ----------
    inlet : StreamInlet
        Inlet to extract the channels list from.

    Returns
    -------
    ch_list : list
        List of channels name ``[name1, name2, ... ]``.
    """
    _check_type(inlet, (StreamInlet,), item_name="inlet")
    root = ET.fromstring(inlet.get_sinfo().as_xml)
    ch_list = []
    for elt in root.iter("channel"):
        ch_list.append(elt.find("label").text)
    return ch_list
