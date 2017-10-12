from __future__ import print_function, division

"""
Export epochs from fif files to mat files

if MERGE_EPOCHS is True, merge epochs from all raw files and export to a
single mat file. Otherwise, export epochs of each raw file individually.

Author: Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

import pycnbi
import pycnbi.utils.pycnbi_utils as pu
import scipy.io, mne
import numpy as np
import pycnbi.utils.q_common as qc

def save_mat(raw, events, matfile):
    sfreq = raw.info['sfreq']

    # pick channels
    if CHANNEL_PICKS is None:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
            eog=False, exclude='bads')
    elif type(CHANNEL_PICKS[0]) == str:
        picks = [raw.ch_names.index(c) for c in CHANNEL_PICKS]
    else:
        assert type(CHANNEL_PICKS[0]) is int
        picks = CHANNEL_PICKS

    # epoching
    epochs = mne.Epochs(raw, events, EVENT_ID, tmin=TMIN, tmax=TMAX, proj=False,
        picks=picks, baseline=(TMIN, TMAX), preload=True)

    # save into mat format
    data = dict(tmin=TMIN, tmax=TMAX, sfreq=epochs.info['sfreq'],
        labels=EVENT_ID.keys(), ch_names=[raw.ch_names[ch] for ch in picks])
    for eve in EVENT_ID:
        data['epochs_%s' % eve] = epochs[eve].get_data()
    scipy.io.savemat(matfile, data)

def epochs2mat(data_dir, channel_picks, event_id, tmin, tmax, merge_epochs=False):
    if merge_epochs:
        # load all raw files in the directory and merge epochs
        fiflist = []
        for data_file in qc.get_file_list(data_dir, fullpath=True):
            if data_file[-4:] != '.fif':
                continue
            fiflist.append(data_file)
        raw, events = pu.load_multi(fiflist, spfilter='car', spchannels=channel_picks)
        matfile = data_dir + '/epochs_all.mat'
        save_mat(raw, events, matfile)
    else:
        # process individual raw file separately
        for data_file in qc.get_file_list(data_dir, fullpath=True):
            if data_file[-4:] != '.fif':
                continue
            [base, fname, fext] = qc.parse_path_list(data_file)
            matfile = '%s/%s-epochs.mat' % (base, fname)
            raw, events = pu.load_raw(data_file)
            save_mat(raw, events, matfile)

    print('\nExported to %s' % matfile)

# sample code
if __name__ == '__main__':
    DATA_DIR = r'D:\data\BCI_COURSE\raw_corrected\Session2\rx7'
    MERGE_EPOCHS = True
    CHANNEL_PICKS = ['O2','O1','TP7','CPz','AF8','TP8','Oz','CP1','CP2','CP3',
        'CP4','CP5','CP6','Pz','PO8','FT7','FT8','P10','AF4','AF7','AF3','P2',
        'P3','P1','P6','P7','P4','P5','P8','P9','AFz','FCz','Fz','C3','C2',
        'C1','C6','C5','C4','FC1','FC2','FC3','FC4','FC5','FC6','T7','F1','F2',
        'F3','F4','F5','F6','F7','F8','Cz','POz','Fpz','T8','PO7','PO4','PO3',
        'Fp1','Fp2']
    EVENT_ID = {'stand':240, 'sit':238}
    TMIN = -2.0
    TMAX = 4.0
    epochs2mat(DATA_DIR, CHANNEL_PICKS, EVENT_ID, TMIN, TMAX, MERGE_EPOCHS)
