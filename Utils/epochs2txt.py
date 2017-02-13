from __future__ import print_function, division

"""
Export epochs into text files.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

DATA_PATH = r'D:\data\MI\rx1\offline\legs-both\fif'
EXCLUDES = ['M1', 'M2', 'EOG']
REF_CH_OLD = None  # 'CPz' # original channel used as a reference
REF_CH_NEW = None  # ['M1','M2']  # re-reference to the average of these channels
CHANNEL_PICKS = None  # [ x for x in range(1,65) if x not in [13,19,32] ]
MULTIPLIER = 1

EPOCHS = {'READY': 6, 'LEGS': 5}
TMIN = 0.0
TMAX = 2.0

import pycnbi_config
import pycnbi_utils as pu
import scipy.io, mne
import numpy as np
import q_common as qc
import trainer
from multiprocessing import cpu_count
from IPython import embed

if __name__ == '__main__':
    rawlist = []
    for f in qc.get_file_list(DATA_PATH, fullpath=True):
        if f[-4:] == '.fif':
            rawlist.append(f)
    if len(rawlist) == 0:
        raise RuntimeError('No fif files found in the path.')

    raw, events = pu.load_multi(rawlist, multiplier=MULTIPLIER)
    raw.pick_types(meg=False, eeg=True, stim=False)
    sfreq = raw.info['sfreq']

    if REF_CH_NEW is not None:
        pu.rereference(raw, REF_CH_NEW, REF_CH_OLD)

    if CHANNEL_PICKS is None:
        picks = [raw.ch_names.index(p) for p in raw.ch_names if p not in EXCLUDES]
    else:
        picks = CHANNEL_PICKS

    # epoching
    for epname, epval in EPOCHS.items():
        epochs = mne.Epochs(raw, events, dict(epname=epval), tmin=TMIN, tmax=TMAX, proj=False, picks=picks,
                            baseline=None, preload=True, add_eeg_ref=False)
        data = epochs.get_data()  # epochs x channels x times
        for i, ep_data in enumerate(data):
            fout = '%s/%s-%d.txt' % (DATA_PATH, epname, i + 1)
            with open(fout, 'w') as f:
                for t in range(ep_data.shape[1]):
                    f.write(qc.list2string(ep_data[:, t], '%.6f') + '\n')
            print('Exported %s' % fout)
