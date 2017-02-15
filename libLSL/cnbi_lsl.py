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

import pylsl as lsl
import q_common as qc
import sys


def start_server(server_name, n_channels=1, channel_format='string', nominal_srate=lsl.IRREGULAR_RATE, stype='EEG',
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
        lsl.cf_string (or 'string')
        lsl.cf_float32 (or 'float32')
        lsl.cf_double64 (or 'double64')
        lsl.cf_int8 (or 'int8')
        lsl.cf_int16 (or 'int16')
        lsl.cf_int32 (or 'int32')
        lsl.cf_int64 (or 'int64')
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
    if source_id == None:
        source_id = server_name
    sinfo = lsl.StreamInfo(server_name, channel_count=n_channels, channel_format=channel_format,\
                           nominal_srate=nominal_srate, type=stype, source_id=source_id)
    return lsl.StreamOutlet(sinfo)


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
    print('Searching for LSL server %s' % server_name)
    while True:
        streamInfos = lsl.resolve_byprop("name", server_name)
        for sinfo in streamInfos:
            print('Found %s' % sinfo.name())
        '''
        if len(streamInfos) > 1:
            print('>> Error: More than 1 server with the same name %s found. Stopping.'% server_name)
            sys.exit(-1)
        elif len(streamInfos)==1:
            sinfo= streamInfos[0]
            break
        else:
            print('[cnbi_lsl] No desired LSL server found. Keep searching...')
            time.sleep(1.0)
        '''
        if len(streamInfos) == 0:
            print('[cnbi_lsl] No desired LSL server found. Keep searching...')
            time.sleep(1.0)
        else:
            sinfo = streamInfos[0]
            break
    return lsl.StreamInlet(sinfo)
