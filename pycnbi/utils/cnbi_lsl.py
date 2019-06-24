from __future__ import print_function, division

"""
LSL wrapper functions for creating a server and a client

Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import sys
import pylsl
import pycnbi.utils.q_common as qc
from pycnbi import logger


def start_server(server_name, n_channels=1, channel_format='string', nominal_srate=pylsl.IRREGULAR_RATE, stype='EEG',
                 source_id=None):
    """
    Start a new LSL server

    Params
    ------
    server_name:
        Name of the server
    n_channels:
        Number of channels
    channel_format:
        pylsl.cf_string (or 'string')
        pylsl.cf_float32 (or 'float32')
        pylsl.cf_double64 (or 'double64')
        pylsl.cf_int8 (or 'int8')
        pylsl.cf_int16 (or 'int16')
        pylsl.cf_int32 (or 'int32')
        pylsl.cf_int64 (or 'int64')
    nominal_srate:
        Sampling rate in Hz. Defaults to irregular sampling rate.
    stype:
        Signal type in string format
    source_id:
        If None, set to server name

    Returns
    -------
    outlet: LSL server object

    """
    if source_id is None:
        source_id = server_name
    sinfo = pylsl.StreamInfo(server_name, channel_count=n_channels, channel_format=channel_format,\
                           nominal_srate=nominal_srate, type=stype, source_id=source_id)
    return pylsl.StreamOutlet(sinfo)


def start_client(server_name):
    """
    Search and connect to an LSL server

    Params
    ------
    server_name:
        Name of the server to search

    Returns
    -------
    inlet:
        LSL client object

    """
    while True:
        logger.info('Searching for LSL server %s ...' % server_name)
        streamInfos = pylsl.resolve_byprop("name", server_name, timeout=1)
        if not streamInfos:
            continue
        for sinfo in streamInfos:
            logger.info('Found %s' % sinfo.name())
        if len(streamInfos) == 0:
            logger.info('No desired LSL server found. Keep searching...')
            time.sleep(1.0)
        else:
            sinfo = streamInfos[0]
            break
    return pylsl.StreamInlet(sinfo)
