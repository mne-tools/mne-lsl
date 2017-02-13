from __future__ import print_function, division

"""
Export epochs to mat files

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

DATA_DIR = r'C:\data\BCI_COURSE\raw_corrected\Session1\rx4'
CHANNEL_PICKS = None  # [5,9,11,12,13,14,15,16]
EVENT_ID = {'stand': 240, 'sit': 238}
TMIN = -2.0
TMAX = 4.0

if __name__ == '__main__':
    import pycnbi_config
    import pycnbi_utils as pu
    import scipy.io, mne
    import numpy as np
    import q_common as qc

    for data_file in qc.get_file_list(DATA_DIR, fullpath=True):
        if data_file[-4:] != '.fif':
            continue

        [base, fname, fext] = qc.parse_path(data_file)
        matfile = '%s/%s-epochs.mat' % (base, fname)
        raw, events = pu.load_raw(data_file)
        sfreq = raw.info['sfreq']

        if CHANNEL_PICKS == None:
            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        else:
            picks = CHANNEL_PICKS

        epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX, proj=False, picks=picks, baseline=(TMIN, TMAX),
                            preload=True, add_eeg_ref=False)

        data = dict(tmin=TMIN, tmax=TMAX, sfreq=epochs.info['sfreq'], labels=EVENT_ID.keys(), ch_names=raw.ch_names)
        for eve in EVENT_ID:
            data['epochs_%s' % eve] = epochs[eve].get_data()

        scipy.io.savemat(matfile, data)

        print('\nExported to %s' % matfile)
