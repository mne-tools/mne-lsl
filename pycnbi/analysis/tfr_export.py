from __future__ import print_function, division

"""
Time-frequency analysis using Morlet wavelets or multitapers

Kyuhwa Lee
EPFL, 2019

"""

import sys
import os
import mne
import imp
import pdb
import scipy
import traceback
import numpy as np
import multiprocessing as mp
import mne.time_frequency
import matplotlib.pyplot as plt
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
from pycnbi import logger
from builtins import input
from scipy.signal import lfilter
from scipy.signal import butter
from scipy.signal import hilbert

def check_config(cfg):
    if not hasattr(cfg, 'TFR_TYPE'):
        cfg.TFR_TYPE = 'multitaper'
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
    if not hasattr(cfg, 'EXPORT_PATH'):
        cfg.EXPORT_PATH = None
    return cfg

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def get_tfr(cfg, recursive=False, n_jobs=1):
    '''
    @params:
    tfr_type: 'multitaper' or 'morlet'
    recursive: if True, load raw files in sub-dirs recursively
    export_path: path to save plots
    n_jobs: number of cores to run in parallel
    '''

    cfg = check_config(cfg)
    tfr_type = cfg.TFR_TYPE
    export_path = cfg.EXPORT_PATH
    t_buffer = cfg.T_BUFFER
    if tfr_type == 'multitaper':
        tfr = mne.time_frequency.tfr_multitaper
    elif tfr_type == 'morlet':
        tfr = mne.time_frequency.tfr_morlet
    elif tfr_type == 'butter':
        butter_order = 4 # TODO: parameterize
        tfr = lfilter
    elif tfr_type == 'fir':
        raise NotImplementedError
    else:
        raise ValueError('Wrong TFR type %s' % tfr_type)
    n_jobs = cfg.N_JOBS
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    if hasattr(cfg, 'DATA_PATHS'):
        if export_path is None:
            raise ValueError('For multiple directories, cfg.EXPORT_PATH cannot be None')
        else:
            outpath = export_path
        # custom event file
        if hasattr(cfg, 'EVENT_FILE') and cfg.EVENT_FILE is not None:
            events = mne.read_events(cfg.EVENT_FILE)
        file_prefix = 'grandavg'

        # load and merge files from all directories
        flist = []
        for ddir in cfg.DATA_PATHS:
            ddir = ddir.replace('\\', '/')
            if ddir[-1] != '/': ddir += '/'
            for f in qc.get_file_list(ddir, fullpath=True, recursive=recursive):
                if qc.parse_path(f).ext in ['fif', 'bdf', 'gdf']:
                    flist.append(f)
        raw, events = pu.load_multi(flist)
    else:
        logger.info('Loading %s' % cfg.DATA_FILE)
        raw, events = pu.load_raw(cfg.DATA_FILE)

        # custom events
        if hasattr(cfg, 'EVENT_FILE') and cfg.EVENT_FILE is not None:
            events = mne.read_events(cfg.EVENT_FILE)

        if export_path is None:
            [outpath, file_prefix, _] = qc.parse_path_list(cfg.DATA_FILE)
        else:
            file_prefix = qc.parse_path(cfg.DATA_FILE).name
            outpath = export_path
            file_prefix = qc.parse_path(cfg.DATA_FILE).name

    # re-referencing
    if cfg.REREFERENCE is not None:
        pu.rereference(raw, cfg.REREFERENCE[1], cfg.REREFERENCE[0])
        assert cfg.REREFERENCE[0] in raw.ch_names

    sfreq = raw.info['sfreq']

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

    # Read epochs
    classes = {}
    for t in cfg.TRIGGERS:
        if t in set(events[:, -1]):
            if hasattr(cfg, 'tdef'):
                classes[cfg.tdef.by_value[t]] = t
            else:
                classes[str(t)] = t
    if len(classes) == 0:
        raise ValueError('No desired event was found from the data.')

    try:
        tmin = cfg.EPOCH[0]
        tmin_buffer = tmin - t_buffer
        raw_tmax = raw._data.shape[1] / sfreq - 0.1
        if cfg.EPOCH[1] is None:
            if cfg.POWER_AVERAGED:
                raise ValueError('EPOCH value cannot have None for grand averaged TFR')
            else:
                if len(cfg.TRIGGERS) > 1:
                    raise ValueError('If the end time of EPOCH is None, only a single event can be defined.')
                t_ref = events[np.where(events[:,2] == list(cfg.TRIGGERS)[0])[0][0], 0] / sfreq
                tmax = raw_tmax - t_ref - t_buffer
        else:
            tmax = cfg.EPOCH[1]
        tmax_buffer = tmax + t_buffer
        if tmax_buffer > raw_tmax:
            raise ValueError('Epoch length with buffer (%.3f) is larger than signal length (%.3f)' % (tmax_buffer, raw_tmax))
        epochs_all = mne.Epochs(raw, events, classes, tmin=tmin_buffer, tmax=tmax_buffer,
                                proj=False, picks=picks, baseline=None, preload=True)
        if epochs_all.drop_log_stats() > 0:
            logger.error('\n** Bad epochs found. Dropping into a Python shell.')
            logger.error(epochs_all.drop_log)
            logger.error('tmin = %.1f, tmax = %.1f, tmin_buffer = %.1f, tmax_buffer = %.1f, raw length = %.1f' % \
                (tmin, tmax, tmin_buffer, tmax_buffer, raw._data.shape[1] / sfreq))
            logger.error('\nType exit to continue.\n')
            pdb.set_trace()
    except:
        logger.critical('\n*** (tfr_export) Unknown error occurred while epoching ***')
        logger.critical('tmin = %.1f, tmax = %.1f, tmin_buffer = %.1f, tmax_buffer = %.1f, raw length = %.1f' % \
            (tmin, tmax, tmin_buffer, tmax_buffer, raw._data.shape[1] / sfreq))
        pdb.set_trace()

    power = {}
    for evname in classes:
        export_dir = outpath
        qc.make_dirs(export_dir)
        logger.info('>> Processing %s' % evname)
        freqs = cfg.FREQ_RANGE  # define frequencies of interest
        n_cycles = freqs / 2.  # different number of cycle per frequency
        if cfg.POWER_AVERAGED:
            # grand-average TFR
            epochs = epochs_all[evname][:]
            if len(epochs) == 0:
                logger.WARNING('No %s epochs. Skipping.' % evname)
                continue

            if tfr_type == 'butter':
                b, a = butter_bandpass(cfg.FREQ_RANGE[0], cfg.FREQ_RANGE[-1], sfreq, order=butter_order)
                tfr_filtered = lfilter(b, a, epochs, axis=2)
                tfr_hilbert = hilbert(tfr_filtered)
                tfr_power = abs(tfr_hilbert)
                tfr_data = np.mean(tfr_power, axis=0)
            elif tfr_type == 'fir':
                raise NotImplementedError
            else:
                power[evname] = tfr(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                    return_itc=False, decim=1, n_jobs=n_jobs)
                power[evname] = power[evname].crop(tmin=tmin, tmax=tmax)
                tfr_data = power[evname].data

            if cfg.EXPORT_MATLAB is True:
                # export all channels to MATLAB
                mout = '%s/%s-%s-%s.mat' % (export_dir, file_prefix, cfg.SP_FILTER, evname)
                scipy.io.savemat(mout, {'tfr':tfr_data, 'chs':epochs.ch_names,
                    'events':events, 'sfreq':sfreq, 'epochs':cfg.EPOCH, 'freqs':cfg.FREQ_RANGE})
                logger.info('Exported %s' % mout)
            if cfg.EXPORT_PNG is True:
                # Inspect power for each channel
                for ch in np.arange(len(picks)):
                    chname = raw.ch_names[picks[ch]]
                    title = 'Peri-event %s - Channel %s' % (evname, chname)

                    # mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
                    fig = power[evname].plot([ch], baseline=cfg.BS_TIMES, mode=cfg.BS_MODE, show=False,
                        colorbar=True, title=title, vmin=cfg.VMIN, vmax=cfg.VMAX, dB=False)
                    fout = '%s/%s-%s-%s-%s.png' % (export_dir, file_prefix, cfg.SP_FILTER, evname, chname)
                    fig.savefig(fout)
                    plt.close()
                    logger.info('Exported to %s' % fout)
        else:
            # TFR per event
            for ep in range(len(epochs_all[evname])):
                epochs = epochs_all[evname][ep]
                if len(epochs) == 0:
                    logger.WARNING('No %s epochs. Skipping.' % evname)
                    continue
                power[evname] = tfr(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                    return_itc=False, decim=1, n_jobs=n_jobs)
                power[evname] = power[evname].crop(tmin=tmin, tmax=tmax)
                if cfg.EXPORT_MATLAB is True:
                    # export all channels to MATLAB
                    mout = '%s/%s-%s-%s-ep%02d.mat' % (export_dir, file_prefix, cfg.SP_FILTER, evname, ep + 1)
                    scipy.io.savemat(mout, {'tfr':power[evname].data, 'chs':power[evname].ch_names,
                        'events':events, 'sfreq':sfreq, 'tmin':tmin, 'tmax':tmax, 'freqs':cfg.FREQ_RANGE})
                    logger.info('Exported %s' % mout)
                if cfg.EXPORT_PNG is True:
                    # Inspect power for each channel
                    for ch in np.arange(len(picks)):
                        chname = raw.ch_names[picks[ch]]
                        title = 'Peri-event %s - Channel %s, Trial %d' % (evname, chname, ep + 1)
                        # mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
                        fig = power[evname].plot([ch], baseline=cfg.BS_TIMES, mode=cfg.BS_MODE, show=False,
                            colorbar=True, title=title, vmin=cfg.VMIN, vmax=cfg.VMAX, dB=False)
                        fout = '%s/%s-%s-%s-%s-ep%02d.png' % (export_dir, file_prefix, cfg.SP_FILTER, evname, chname, ep + 1)
                        fig.savefig(fout)
                        plt.close()
                        logger.info('Exported %s' % fout)

    if hasattr(cfg, 'POWER_DIFF'):
        export_dir = '%s/diff' % outpath
        qc.make_dirs(export_dir)
        labels = classes.keys()
        df = power[labels[0]] - power[labels[1]]
        df.data = np.log(np.abs(df.data))
        # Inspect power diff for each channel
        for ch in np.arange(len(picks)):
            chname = raw.ch_names[picks[ch]]
            title = 'Peri-event %s-%s - Channel %s' % (labels[0], labels[1], chname)

            # mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
            fig = df.plot([ch], baseline=cfg.BS_TIMES, mode=cfg.BS_MODE, show=False,
                          colorbar=True, title=title, vmin=3.0, vmax=-3.0, dB=False)
            fout = '%s/%s-%s-diff-%s-%s-%s.jpg' % (export_dir, file_prefix, cfg.SP_FILTER, labels[0], labels[1], chname)
            logger.info('Exporting to %s' % fout)
            fig.savefig(fout)
            plt.close()
    logger.info('Finished !')

def load_config(cfg_file):
    cfg_file = qc.forward_slashify(cfg_file)
    if not (os.path.exists(cfg_file) and os.path.isfile(cfg_file)):
        raise IOError('%s cannot be loaded.' % os.path.realpath(cfg_file))
    return imp.load_source(cfg_file, cfg_file)

def batch_run(cfg_file):
    cfg = load_config(cfg_file)
    cfg = check_config(cfg)
    get_tfr(cfg)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_file = input('Config file name? ')
    else:
        cfg_file = sys.argv[1]
    batch_run(cfg_file)
