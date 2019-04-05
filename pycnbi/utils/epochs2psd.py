from __future__ import print_function, division

"""
Compute PSD features over a sliding window in epochs

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

import mne
import scipy.io
import numpy as np
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from multiprocessing import cpu_count
mne.set_log_level('ERROR')


def epochs2psd(raw, channel_picks, event_id, tmin, tmax, fmin, fmax, w_len, w_step, excludes='bads', export_dir=None, n_jobs=None):
    """
    Compute PSD features over a sliding window in epochs

    Input
    =====
    raw: str | mne.RawArray. If str, it is treated as a file name
    channel_picks: None or list of channel names(str) or indices(int)
    event_id: { label(str) : event_id(int) }
    tmin: start time of the PSD window relative to the event onset
    tmax: end time of the PSD window relative to the event onset
    fmin: minimum PSD frequency
    fmax: maximum PSD frequency
    w_len: sliding window length for computing PSD
    w_step: sliding window step in time samples
    excludes: channels to exclude
    export_dir: path to export PSD data. Automatically saved in the same directory of raw if raw is a filename.

    Output
    ======
    4-D numpy array: [epochs] x [times] x [channels] x [freqs]

    """

    if n_jobs is None:
        n_jobs = cpu_count()

    # load raw object or file
    if type(raw) == str:
        rawfile = raw.replace('\\', '/')
        raw, events = pu.load_raw(rawfile)
        [export_dir, export_file, _] = qc.parse_path_list(rawfile)
    else:
        if export_dir is None:
            raise ValueError('export_dir must be given if raw object is given as argument')
        export_file = 'raw'
        events = mne.find_events(raw, stim_channel='TRIGGER', shortest_event=1, uint_cast=True, consecutive=True)
    sfreq = raw.info['sfreq']

    # pick channels of interest and do epoching
    if channel_picks is None:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=excludes)
    elif type(channel_picks[0]) == str:
        picks = []
        for ch in channel_picks:
            picks.append(raw.ch_names.index(ch))
    elif type(channel_picks[0]) == int:
        picks = channel_picks
    else:
        raise ValueError('Unknown data type (%s) in channel_picks' % type(channel_picks[0]))
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, proj=False, picks=picks, baseline=(tmin, tmax), preload=True)

    # compute psd vectors over a sliding window between tmin and tmax
    w_len = int(sfreq * w_len)  # window length
    psde = mne.decoding.PSDEstimator(sfreq, fmin=fmin, fmax=fmax, n_jobs=1, adaptive=False)
    epochmat = {e:epochs[e]._data for e in event_id}
    psdmat = {}
    for e in event_id:
        # psd = [epochs] x [windows] x [channels] x [freqs]
        psd, _ = pu.get_psd(epochs[e], psde, w_len, w_step, flatten=False, n_jobs=n_jobs)
        psdmat[e] = psd

    # export data
    data = dict(psds=psdmat, tmin=tmin, tmax=tmax, sfreq=epochs.info['sfreq'],\
                fmin=fmin, fmax=fmax, w_step=w_step, w_len=w_len, labels=list(epochs.event_id.keys()))
    matfile = '%s/psd-%s.mat' % (export_dir, export_file)
    pklfile = '%s/psd-%s.pkl' % (export_dir, export_file)
    scipy.io.savemat(matfile, data)
    qc.save_obj(pklfile, data)
    print('Exported to %s' % matfile)
    print('Exported to %s' % pklfile)
