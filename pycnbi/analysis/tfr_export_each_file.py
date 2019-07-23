from __future__ import print_function, division

"""
Time-frequency analysis using Morlet wavelets or multitapers
TFR computed on each raw file, from the beginning until the end.
The result is the grand average of TFRs over all epochs.

Kyuhwa Lee, 2018

"""

import sys
import os
import imp
import mne
import scipy
import numpy as np
import mne.time_frequency
import multiprocessing as mp
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
from pycnbi import logger
from builtins import input

def check_cfg(cfg):
    if not hasattr(cfg, 'N_JOBS'):
        cfg.N_JOBS = None
    if not hasattr(cfg, 'T_BUFFER'):
        cfg.T_BUFFER = 1
    if not hasattr(cfg, 'BS_MODE'):
        cfg.BS_MODE = 'logratio'
    if not hasattr(cfg, 'BS_TIMES'):
        cfg.BS_TIMES = (None, 0)
    if not hasattr(cfg, 'EXPORT_PNG'):
        cfg.EXPORT_PNG = False
    if not hasattr(cfg, 'EXPORT_MATLAB'):
        cfg.MATLAB = False
    if not hasattr(cfg, 'EVENT_START'):
        cfg.EVENT_START = None
    return cfg

def get_tfr_each_file(cfg, tfr_type='multitaper', recursive=False, export_path=None, n_jobs=1):
    '''
    @params:
    tfr_type: 'multitaper' or 'morlet'
    recursive: if True, load raw files in sub-dirs recursively
    export_path: path to save plots
    n_jobs: number of cores to run in parallel
    '''

    cfg = check_cfg(cfg)

    t_buffer = cfg.T_BUFFER
    if tfr_type == 'multitaper':
        tfr = mne.time_frequency.tfr_multitaper
    elif tfr_type == 'morlet':
        tfr = mne.time_frequency.tfr_morlet
    else:
        raise ValueError('Wrong TFR type %s' % tfr_type)

    for fifdir in cfg.DATA_PATHS:
        for f in qc.get_file_list(fifdir, fullpath=True, recursive=recursive):
            [fdir, fname, fext] = qc.parse_path_list(f)
            if fext in ['fif', 'bdf', 'gdf']:
                get_tfr(f, cfg, tfr, cfg.N_JOBS)

def get_tfr(fif_file, cfg, tfr, n_jobs=1):
    raw, events = pu.load_raw(fif_file)
    p = qc.parse_path(fif_file)
    fname = p.name
    outpath = p.dir

    export_dir = '%s/plot_%s' % (outpath, fname)
    qc.make_dirs(export_dir)

    # set channels of interest
    picks = pu.channel_names_to_index(raw, cfg.CHANNEL_PICKS)
    spchannels = pu.channel_names_to_index(raw, cfg.SP_CHANNELS)

    if max(picks) > len(raw.info['ch_names']):
        msg = 'ERROR: "picks" has a channel index %d while there are only %d channels.' %\
              (max(picks), len(raw.info['ch_names']))
        raise RuntimeError(msg)

    # Apply filters
    raw = pu.preprocess(raw, spatial=cfg.SP_FILTER, spatial_ch=spchannels, spectral=cfg.TP_FILTER,
                  spectral_ch=picks, notch=cfg.NOTCH_FILTER, notch_ch=picks,
                  multiplier=cfg.MULTIPLIER, n_jobs=n_jobs)

    # MNE TFR functions do not support Raw instances yet, so convert to Epoch
    if cfg.EVENT_START is None:
        raw._data[0][0] = 1
        events = np.array([[0, 0, 1]])
        classes = None
    else:
        classes = {'START':cfg.EVENT_START}
    tmax = (raw._data.shape[1] - 1) / raw.info['sfreq']
    epochs_all = mne.Epochs(raw, events, classes, tmin=0, tmax=tmax,
                    picks=picks, baseline=None, preload=True)
    logger.info('\n>> Processing %s' % fif_file)
    freqs = cfg.FREQ_RANGE  # define frequencies of interest
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power = tfr(epochs_all, freqs=freqs, n_cycles=n_cycles, use_fft=False,
        return_itc=False, decim=1, n_jobs=n_jobs)

    if cfg.EXPORT_MATLAB is True:
        # export all channels to MATLAB
        mout = '%s/%s-%s.mat' % (export_dir, fname, cfg.SP_FILTER)
        scipy.io.savemat(mout, {'tfr':power.data, 'chs':power.ch_names, 'events':events,
            'sfreq':raw.info['sfreq'], 'freqs':cfg.FREQ_RANGE})

    if cfg.EXPORT_PNG is True:
        # Plot power of each channel
        for ch in np.arange(len(picks)):
            ch_name = raw.ch_names[picks[ch]]
            title = 'Channel %s' % (ch_name)
            # mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
            fig = power.plot([ch], baseline=cfg.BS_TIMES, mode=cfg.BS_MODE, show=False,
                colorbar=True, title=title, vmin=cfg.VMIN, vmax=cfg.VMAX, dB=False)
            fout = '%s/%s-%s-%s.png' % (export_dir, fname, cfg.SP_FILTER, ch_name)
            fig.savefig(fout)
            logger.info('Exported %s' % fout)

    logger.info('Finished !')

def config_run(cfg_module):
    cfg = imp.load_source(cfg_module, cfg_module)
    if not hasattr(cfg, 'TFR_TYPE'):
        cfg.TFR_TYPE = 'multitaper'
    get_tfr_each_file(cfg, tfr_type=cfg.TFR_TYPE)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_module = input('Config file name? ')
    else:
        cfg_module = sys.argv[1]
    config_run(cfg_module)
