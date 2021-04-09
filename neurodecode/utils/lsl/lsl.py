from __future__ import print_function, division

"""
LSL wrapper functions for creating a server and a client.
"""
import pylsl
import multiprocessing as mp
from neurodecode import logger

#----------------------------------------------------------------------
def start_server(server_name, n_channels=1, channel_format='string', nominal_srate=pylsl.IRREGULAR_RATE, stype='EEG',
                 source_id=None):
    """
    Start a new LSL server.

    Parameters
    ----------
    server_name : str
        Name of the server
    n_channels : int
        Number of channels
    channel_format : str
        The channels' format ('string', 'float32', 'double64', 'int8', 'int16', 'int32', 'int64')
    nominal_srate : float
        Sampling rate in Hz
    stype : str
        Signal type
    source_id : str
        If None, set to server name

    Returns
    -------
    LSL outlet :
        LSL server object
    """
    if source_id is None:
        source_id = server_name
        
    sinfo = pylsl.StreamInfo(server_name, channel_count=n_channels, channel_format=channel_format,\
                           nominal_srate=nominal_srate, type=stype, source_id=source_id)
    return pylsl.StreamOutlet(sinfo)

#----------------------------------------------------------------------
def start_client(server_name, state=mp.Value('i', 1)):
    """
    Search and connect to an LSL server.

    Parameters
    ----------
    server_name: str
        Name of the server to search
    state : Multiprocessing.Value 
        Used to searching stop from another process.

    Returns
    -------
    LSL inlet:
        LSL client object
    """
    while state.value == 1:
        logger.info('Searching for LSL server %s ...' % server_name)
        streamInfos = pylsl.resolve_byprop("name", server_name, timeout=1)
        
        if not streamInfos:
            continue
        
        for sinfo in streamInfos:
            logger.info('Found %s' % sinfo.name())
        sinfo = streamInfos[0]
        break
    
    return pylsl.StreamInlet(sinfo)
