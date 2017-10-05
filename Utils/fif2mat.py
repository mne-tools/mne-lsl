from __future__ import print_function, division

"""
Export fif data into mat files.

"""

import pycnbi_config
import pycnbi_utils as pu
import scipy.io, mne
import q_common as qc

def fif2mat(data_dir):
    for rawfile in qc.get_file_list(data_dir, fullpath=True):
        if rawfile[-4:] != '.fif': continue
        raw, events = pu.load_raw(rawfile)
        sfreq = raw.info['sfreq']
        data = dict(signals=raw._data, events=events, sfreq=sfreq, ch_names=raw.ch_names)
        matfile = '.'.join(rawfile.split('.')[:-1]) + '.mat'
        scipy.io.savemat(matfile, data)
        print('\nExported to %s' % matfile)
    print('\nDone.')

if __name__ == '__main__':
    # path to fif file(s)
    data_dir = r'D:\data\Records\fif'
    fif2mat(data_dir)
