"""
LSL wrapper functions for creating a server and a client.
"""
import xml.etree.ElementTree as ET

import pylsl

from . import Timer
from .. import logger


def list_lsl_streams(ignore_markers=False):
    """
    List all the available outlets on LSL network.

    Parameters
    ----------
    ignore_markers : `bool`
        If ``True``, ignore streams with Marker type.

    Returns
    -------
    stream_list : `list`
        List of the found stream name.
    streamInfos : `list`
        List of the corresponding ``pylsl.StreamInfo``.
    """
    stream_list = []
    streamInfos = pylsl.resolve_streams()

    if len(streamInfos) == 0:
        return stream_list, []

    for index, streamInfo in enumerate(streamInfos):
        if ignore_markers and 'Markers' in streamInfo.type():
            continue
        stream_list.append(streamInfo.name())

    if ignore_markers:
        streamInfos = [streamInfo for streamInfo in streamInfos
                       if 'Markers' not in streamInfo.type()]

    return stream_list, streamInfos


def search_lsl(ignore_markers=False, timeout=10):
    """
    Search and select an available stream on LSL network.
    Does not open an LSL inlet.

    Parameters
    ----------
    ignore_markers : `bool`
        If ``True``, ignore streams with Marker type.
    timeout : `int`
        Timeout duration in seconds after which the search of an LSL stream
        is interrupted.

    Returns
    -------
    stream_name : `str`
        Selected stream name.
    """
    watchdog = Timer()
    while watchdog.sec() <= timeout:
        stream_list, streamInfos = list_lsl_streams(ignore_markers)
        if len(stream_list) != 0:
            break
    else:
        logger.error('Timeout. No LSL stream found.')
        return None

    logger.info('-- List of servers --')
    for i, stream_name in enumerate(stream_list):
        logger.info(f'{i}: {stream_name}')

    if len(stream_list) == 0:
        logger.error('No LSL stream found on the network.')
    elif len(stream_list) == 1:
        index = 0
    else:
        index = input(
            'Stream index? '
            'Hit enter without index to select the first server.\n>> ')
        if index.strip() == '':
            index = 0
        else:
            index = int(index.strip())

    stream_name = stream_list[index]
    streamInfo = streamInfos[index]
    assert stream_name == streamInfo.name()

    logger.info(f'Selected: {stream_name}')

    return stream_name


def lsl_channel_list(inlet):
    """
    Extract the channels name list from the LSL info.

    Parameters
    ----------
    inlet : pylsl.StreamInlet
        Inlet to extract the channels list from.

    Returns:
    --------
    ch_list : `list`
        List of channels name [name1, name2, ... ]
    """
    if not isinstance(inlet, pylsl.StreamInlet):
        logger.error(f'Wrong input type {type(inlet)}')
        raise TypeError

    xml_str = inlet.info().as_xml()
    root = ET.fromstring(xml_str)

    ch_list = []
    for elt in root.iter('channel'):
        ch_list.append(elt.find('label').text)

    return ch_list
