from __future__ import print_function, division

"""
Export epochs to one text file per epoch
File name  : EVENT-N.txt (N=0,1,2,...)
Data format: times x channels, space-separated floats

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

DATA_PATH = r'D:\data\BCI_COURSE\raw_corrected\Session2\rx7'
EXCLUDES = ['M1', 'M2', 'EOG']
REF_CH_OLD = None  # original channel used as a reference
REF_CH_NEW = None  # re-reference to the average of these channels
CHANNEL_PICKS = None # list of integers or list of strings
MULTIPLIER = 1 # unit multiplier
EPOCHS = {'stand':240, 'sit':238}
TMIN = 0.0
TMAX = 2.0


import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
import scipy.io
import mne
import numpy as np
import trainer
from multiprocessing import cpu_count
from pycnbi import logger

mne.set_log_level('ERROR')

if __name__ == '__main__':
    rawlist = []
    for f in qc.get_file_list(DATA_PATH, fullpath=True):
        if f[-4:] == '.fif':
            rawlist.append(f)
    if len(rawlist) == 0:
        raise RuntimeError('No fif files found in the path.')

    # make output directory
    out_path = DATA_PATH + '/epochs'
    qc.make_dirs(out_path)

    # load data
    raw, events = pu.load_multi(rawlist, multiplier=MULTIPLIER)
    raw.pick_types(meg=False, eeg=True, stim=False)
    sfreq = raw.info['sfreq']
    if REF_CH_NEW is not None:
        pu.rereference(raw, REF_CH_NEW, REF_CH_OLD)

    # pick channels
    if CHANNEL_PICKS is None:
        picks = [raw.ch_names.index(c) for c in raw.ch_names if c not in EXCLUDES]
    elif type(CHANNEL_PICKS[0]) == str:
        picks = [raw.ch_names.index(c) for c in CHANNEL_PICKS]
    else:
        assert type(CHANNEL_PICKS[0]) is int
        picks = CHANNEL_PICKS

    # do epoching
    for epname, epval in EPOCHS.items():
        epochs = mne.Epochs(raw, events, dict(epname=epval), tmin=TMIN, tmax=TMAX,
                            proj=False, picks=picks, baseline=None, preload=True)
        data = epochs.get_data()  # epochs x channels x times
        for i, ep_data in enumerate(data):
            fout = '%s/%s-%d.txt' % (out_path, epname, i + 1)
            with open(fout, 'w') as f:
                for t in range(ep_data.shape[1]):
                    f.write(qc.list2string(ep_data[:, t], '%.6f') + '\n')
            logger.info('Exported %s' % fout)
