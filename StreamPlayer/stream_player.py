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
# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

def stream_player(server_name, fif_file, chunk_size, auto_restart=True):
    raw, events = pu.load_raw(fif_file)
    sfreq = raw.info['sfreq']  # sampling frequency
    n_channels = len(raw.ch_names)  # number of channels
    if raw is not None:
        print('Successfully loaded %s\n' % fif_file)
        print('Server name: %s' % server_name)
        print('Sampling frequency %.1f Hz' % sfreq)
        print('Number of channels : %d' % n_channels)
        for i, ch in enumerate(raw.ch_names):
            print(i, ch)
    else:
        raise RuntimeError('Error while loading %s' % fif_file)

    # set server information
    sinfo = lsl.StreamInfo(server_name, channel_count=n_channels, channel_format='float32',\
        nominal_srate=sfreq, type='EEG', source_id=server_name)
    xmldesc = sinfo.desc().append_child("channels")
    for ch in raw.ch_names:
        xmldesc.append_child('channel').append_child_value('label', str(ch))\
            .append_child_value('type','EEG').append_child_value('unit','microvolts')
    outlet = lsl.StreamOutlet(sinfo)
    sinfo.desc().append_child('amplifier').append_child('settings').append_child_value('is_slave', 'false')
    sinfo.desc().append_child('acquisition').append_child_value('manufacturer', 'PyCNBI')\
        .append_child_value('serial_number', 'None')
    
    input('Press Enter to start streaming.')

    idx_chunk = 0
    t_chunk = chunk_size / sfreq
    finished = False
    tm = qc.Timer()
    while True:
        idx_current = idx_chunk * chunk_size
        if idx_current < raw._data.shape[1] - chunk_size:
            data = raw._data[:, idx_current:idx_current + chunk_size].transpose().tolist()
        else:
            data = raw._data[:, idx_current:].transpose().tolist()
            finished = True
        t_wait = idx_chunk * t_chunk - tm.sec()
        if t_wait > 0:
            time.sleep(t_wait)
        outlet.push_chunk(data)
        print('[%8.3fs] sent %d samples' % (tm.sec(), len(data[0])))
        idx_chunk += 1

        if finished:
            if auto_restart is False:
                input('Reached the end of data. Press Enter to restart or Ctrl+C to stop.')
            else:
                print('Reached the end of data. Restarting.')
            idx_chunk = 0
            finished = False
            tm.reset()

# sample code
if __name__ == '__main__':
    server_name = 'StreamPlayer'
    chunk_size = 32  # chunk streaming frequency in Hz
    fif_file = r'D:\data\CHUV\ECoG17\20171005\20171005-103303-raw_T1.fif'
    stream_player(server_name, fif_file, chunk_size)
