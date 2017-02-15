from __future__ import print_function, division

"""
Stream Player

Stream signals from a recorded file on LSL network.

TODO:
	Create XML header.

Kyuhwa Lee, 2015

"""

import pycnbi_config
import pycnbi_utils as pu
import numpy as np
import pylsl as lsl
import q_common as qc
import cnbi_lsl
import time

SERVER_NAME = 'StreamPlayer'
CHUNK_SIZE = 32  # chunk streaming frequency in Hz
RECORD_FILE = r'D:\data\MI\rx1\train\20160601-105104-raw.fif'

raw, events = pu.load_raw(RECORD_FILE)
sfreq = 512  # sampling frequency
n_channels = 16  # number of channels

sdelay = CHUNK_SIZE / sfreq  # in seconds
# outlet= cnbi_lsl.start_server(SERVER_NAME, n_channels=n_channels, channel_format='float32', nominal_srate=sfreq, stype='EEG')
sinfo = lsl.StreamInfo(SERVER_NAME, channel_count=n_channels, channel_format='float32', nominal_srate=sfreq, type='EEG',
                       source_id=SERVER_NAME)
outlet = lsl.StreamOutlet(sinfo)
print(SERVER_NAME, 'server start.')

tm = qc.Timer()
snext = sdelay
index_s = 0
while True:
    while tm.sec() < snext:
        time.sleep(0.001)
    snext += sdelay
    data = raw._data[1:, index_s:(index_s + CHUNK_SIZE)].tolist()
    outlet.push_chunk(data)
    print('[%7.3fs] sent %d samples' % (tm.sec(), len(data)))
    index_s += CHUNK_SIZE

    # replay
    if index_s >= raw._data.shape[1] - CHUNK_SIZE:
        index_s = 0
        snext = sdelay
        tm.reset()
        print('Reached end of file. Restarting.\n')
