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

def stream_player(fif_file):
    raw, events = pu.load_raw(fif_file)
    sfreq = raw.info['sfreq']  # sampling frequency
    n_channels = len(raw.ch_names)  # number of channels
    if raw is not None:
        print('Successfully loaded %s' % fif_file)
        print('Number of channels: %d' % n_channels)
        print('Sampling frequency %.1f Hz' % sfreq)
    else:
        raise RuntimeError('Error while loading %s' % fif_file)

    sdelay = chunk_size / sfreq  # in seconds
    # outlet= cnbi_lsl.start_server(server_name, n_channels=n_channels, channel_format='float32', nominal_srate=sfreq, stype='EEG')
    sinfo = lsl.StreamInfo(server_name, channel_count=n_channels, channel_format='float32', nominal_srate=sfreq, type='EEG',
                           source_id=server_name)
    outlet = lsl.StreamOutlet(sinfo)
    print(server_name, 'server start.')

    idx_chunk = 0
    t_chunk = chunk_size / sfreq
    restart = False
    tm = qc.Timer()
    while True:
        idx_current = idx_chunk * chunk_size
        if idx_current < raw._data.shape[1] - chunk_size:
            data = raw._data[:, idx_current:idx_current + chunk_size].tolist()
        else:
            data = raw._data[:, idx_current:].tolist()
            restart = True
        time.sleep(idx_chunk * t_chunk - tm.sec())
        outlet.push_chunk(data)
        print('[%8.3fs] sent %d samples' % (tm.sec(), len(data[0])))
        idx_chunk += 1

        if restart:
            print('Reached the end of data. Restarting.\n')
            idx_chunk = 0
            restart = False
            tm.reset()

# sample code
if __name__ == '__main__':
    server_name = 'StreamPlayer'
    chunk_size = 32  # chunk streaming frequency in Hz
    fif_file = r'D:\data\CHUV\ECoG17\20171005\20171005-103303-raw_T1.fif'
    stream_player(fif_file)
