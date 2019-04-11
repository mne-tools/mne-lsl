from __future__ import print_function, division

"""
Export epochs from fif files to mat files

if MERGE_EPOCHS is True, merge epochs from all raw files and export to a
single mat file. Otherwise, export epochs of each raw file individually.

Author: Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

import mne
import scipy.io
import numpy as np
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
from pycnbi import logger

mne.set_log_level('ERROR')

def save_mat(raw, events, picks, event_id, tmin, tmax, matfile):
    sfreq = raw.info['sfreq']

    # pick channels
    if picks is None:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
            eog=False, exclude='bads')
    elif type(picks[0]) == str:
        picks = [raw.ch_names.index(c) for c in picks]
    else:
        assert type(picks[0]) is int
        picks = picks

    # epoching
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, proj=False,
        picks=picks, baseline=(tmin, tmax), preload=True)

    # save into mat format
    data = dict(tmin=tmin, tmax=tmax, sfreq=epochs.info['sfreq'],
        labels = list(event_id.keys()), ch_names = [raw.ch_names[ch] for ch in picks])
    for eve in event_id:
        data['epochs_%s' % eve] = epochs[eve].get_data()
    scipy.io.savemat(matfile, data)

def epochs2mat(data_dir, channel_picks, event_id, tmin, tmax, merge_epochs=False, spfilter=None, spchannels=None):
    if merge_epochs:
        # load all raw files in the directory and merge epochs
        fiflist = []
        for data_file in qc.get_file_list(data_dir, fullpath=True):
            if data_file[-4:] != '.fif':
                continue
            fiflist.append(data_file)
        raw, events = pu.load_multi(fiflist, spfilter=spfilter, spchannels=spchannels)
        matfile = data_dir + '/epochs_all.mat'
        save_mat(raw, events, channel_picks, event_id, tmin, tmax, matfile)
    else:
        # process individual raw file separately
        for data_file in qc.get_file_list(data_dir, fullpath=True):
            if data_file[-4:] != '.fif':
                continue
            [base, fname, fext] = qc.parse_path_list(data_file)
            matfile = '%s/%s-epochs.mat' % (base, fname)
            raw, events = pu.load_raw(data_file)
            save_mat(raw, events, channel_picks, event_id, tmin, tmax, matfile)

    logger.info('Exported to %s' % matfile)

# sample code
if __name__ == '__main__':
    data_dir = r'd:\data\mi\z2\lr'
    channel_picks = ['F3','F4','C3','Cz','C4','P3','Pz','P4']
    event_id = {'LEFT_GO':11, 'RIGHT_GO':9}
    tmin = -1.0
    tmax = 3.0
    merge_epochs = True
    spfilter = 'car'
    epochs2mat(data_dir, channel_picks, event_id, tmin, tmax, merge_epochs, spfilter=spfilter)
