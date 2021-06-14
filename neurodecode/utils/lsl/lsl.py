"""
LSL wrapper functions for creating a server and a client.
"""
import time
import multiprocessing as mp
import xml.etree.ElementTree as ET

import pylsl

from ... import logger


def start_server(server_name, n_channels=1, channel_format='string',
                 nominal_srate=pylsl.IRREGULAR_RATE, stype='Markers',
                 source_id=None):
    """
    Start a new LSL server.

    Parameters
    ----------
    server_name : str
        Name of the server.
    n_channels : int
        Number of channels.
    channel_format : str
        The channels' format.
            ('string', 'float32', 'double64', 'int8', 'int16',
             'int32', 'int64')
    nominal_srate : float
        Sampling rate in Hz.
    stype : str
        Signal type.
        (https://github.com/sccn/xdf/wiki/Meta-Data#stream-content-types)
    source_id : str
        Unique identifier of the device or source of the data, if available
        (such as the serial number). If None, set to server name.

    Returns
    -------
    LSL outlet :
        LSL server object
    """
    if source_id is None:
        source_id = server_name

    sinfo = pylsl.StreamInfo(server_name,
                             channel_count=n_channels,
                             channel_format=channel_format,
                             nominal_srate=nominal_srate,
                             type=stype, source_id=source_id)
    return pylsl.StreamOutlet(sinfo)


def start_client(server_name, state=mp.Value('i', 1)):
    """
    Search for an LSL outlet (server) and open an LSL inlet (client).

    Parameters
    ----------
    server_name: str
        Name of the server to search.
    state : Multiprocessing.Value
        Used to stop searching from another process.

    Returns
    -------
    LSL inlet:
        LSL client object.
    """
    while state.value == 1:
        logger.info(f'Searching for LSL server {server_name} ...')
        streamInfos = pylsl.resolve_byprop("name", server_name, timeout=1)

        if not streamInfos:
            continue

        for sinfo in streamInfos:
            logger.info(f'Found {sinfo.name()}')
        sinfo = streamInfos[0]
        break

    return pylsl.StreamInlet(sinfo)


def list_lsl_streams(ignore_markers=False,
                     logger=logger, state=mp.Value('i', 1)):
    """
    List all the available outlets on LSL network.

    Parameters
    ----------
    ignore_markers : bool
        If True, ignore streams with Marker type.
    logger : logging.Logger
        The logger to output info.
    state: mp.Value
        The multiprocess sharing variable, used to stop search from another
        process.
    """

    # look for LSL servers
    stream_list = []
    stream_list_markers = []

    while state.value == 1:

        streamInfos = pylsl.resolve_streams()

        if len(streamInfos) > 0:

            for index, streamInfo in enumerate(streamInfos):
                stream_name = streamInfo.name()
                if 'Markers' in streamInfo.type():
                    stream_list_markers.append((index, stream_name))
                else:
                    stream_list.append((index, stream_name))
            break

        logger.info('No server available yet on the network...')
        time.sleep(1)

    if ignore_markers is False:
        stream_list += stream_list_markers

    logger.info('-- List of servers --')

    for i, (index, stream_name) in enumerate(stream_list):
        logger.info(f'{i}: {stream_name}')

    return stream_list, streamInfos


def search_lsl(ignore_markers=False, logger=logger, state=mp.Value('i', 1)):
    """
    Search and select an available stream on LSL network.

    Does not open a LSL inlet.

    Parameters
    ----------
    ignore_markers : bool
        If True, ignore streams with Marker type.
    logger : logging.Logger
        The logger to output info.
    state: mp.Value
        The multiprocess sharing variable, used to stop search from another
        process.

    Returns:
    --------
    str : The selected stream name.
    """
    stream_list, streamInfos = list_lsl_streams(ignore_markers, logger, state)

    if len(stream_list) == 1:
        index = 0
    else:
        index = input(
            'Stream index? '
            'Hit enter without index to select the first server.\n>> ')
        if index.strip() == '':
            index = 0
        else:
            index = int(index.strip())

    stream_index, stream_name = stream_list[index]
    streamInfo = streamInfos[stream_index]
    assert stream_name == streamInfo.name()

    logger.info(f'Selected: {stream_name}')

    return stream_name


def lsl_channel_list(inlet):
    """
    Extract the channels name list from the LSL info.

    Parameters
    ----------
    inlet : pylsl.StreamInlet
        The inlet to extract channels list from.

    Returns:
    --------
    list : List of channels name [ name1, name2, ... ]
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
