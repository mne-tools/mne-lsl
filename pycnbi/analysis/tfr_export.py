from __future__ import print_function, division

"""
Time-frequency analysis using Morlet wavelets or multitapers

Kyuhwa Lee, 2017

"""

import pycnbi.utils.pycnbi_utils as pu
import sys
import os
import mne
import scipy
import multiprocessing as mp
import numpy as np
import pycnbi.utils.q_common as qc
import mne.time_frequency
import imp
from builtins import input
from IPython import embed

def check_cfg(cfg):
    if not hasattr(cfg, 'N_JOBS'):
        cfg.N_JOBS = None
    if not hasattr(cfg, 'T_BUFFER'):
        cfg.T_BUFFER = 1
    if not hasattr(cfg, 'BS_MODE'):
        cfg.BS_MODE = 'logratio'
    if not hasattr(cfg, 'EXPORT_PNG'):
        cfg.EXPORT_PNG = True
    if not hasattr(cfg, 'EXPORT_MATLAB'):
        cfg.MATLAB = True
    if not hasattr(cfg, 'TFR_TYPE'):
        cfg.TFR_TYPE = 'multitaper'
    return cfg

def get_tfr(cfg, recursive=False, export_path=None):
    '''
    @params:
    tfr_type: 'multitaper' or 'morlet'
    recursive: if True, load raw files in sub-dirs recursively
    export_path: path to save plots
    '''

    cfg = check_cfg(cfg)
    tfr_type = cfg.TFR_TYPE

    t_buffer = cfg.T_BUFFER
    if tfr_type == 'multitaper':
        tfr = mne.time_frequency.tfr_multitaper
    elif tfr_type == 'morlet':
        tfr = mne.time_frequency.tfr_morlet
    else:
        raise ValueError('Wrong TFR type %s' % tfr_type)
    n_jobs = cfg.N_JOBS
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    if hasattr(cfg, 'DATA_DIRS'):
        # concatenate multiple files
        for ddir in cfg.DATA_DIRS:
            ddir = ddir.replace('\\', '/')
            if ddir[-1] != '/': ddir += '/'
            flist = []
            for f in qc.get_file_list(ddir, fullpath=True, recursive=recursive):
                [fdir, fname, fext] = qc.parse_path_list(f)
                if fext in ['fif', 'bdf', 'gdf']:
                    flist.append(f)
            raw, events = pu.load_multi(flist)

            # custom events
            if hasattr(cfg, 'EVENT_FILE') and cfg.EVENT_FILE is not None:
                events = mne.read_events(cfg.EVENT_FILE)

            sp = ddir.split('/')
            file_prefix = '-'.join(sp[-4:-1])
            if export_path is None:
                outpath = ddir
            else:
                outpath = export_path
    else:
        print('Loading', cfg.DATA_FILE)
        raw, events = pu.load_raw(cfg.DATA_FILE)

        # custom events
        if hasattr(cfg, 'EVENT_FILE') and cfg.EVENT_FILE is not None:
            events = mne.read_events(cfg.EVENT_FILE)

        if export_path is None:
            [outpath, file_prefix, _] = qc.parse_path_list(cfg.DATA_FILE)
        else:
            outpath = export_path

    # set channels of interest
    picks = pu.channel_names_to_index(raw, cfg.CHANNEL_PICKS)
    spchannels = pu.channel_names_to_index(raw, cfg.SP_CHANNELS)

    if max(picks) > len(raw.info['ch_names']):
        msg = 'ERROR: "picks" has a channel index %d while there are only %d channels.' %\
              (max(picks), len(raw.info['ch_names']))
        raise RuntimeError(msg)

    # Apply filters
    pu.preprocess(raw, spatial=cfg.SP_FILTER, spatial_ch=spchannels, spectral=cfg.TP_FILTER,
                  spectral_ch=picks, notch=cfg.NOTCH_FILTER, notch_ch=picks,
                  multiplier=cfg.MULTIPLIER, n_jobs=n_jobs)

    # Read epochs
    try:
        classes = {}
        for t in cfg.TRIGGERS:
            if t in set(events[:, -1]):
                if hasattr(cfg, 'tdef'):
                    classes[cfg.tdef.by_value[t]] = t
                else:
                    classes[str(t)] = t
        assert len(classes) > 0

        epochs_all = mne.Epochs(raw, events, classes, tmin=cfg.EPOCH[0] - t_buffer, tmax=cfg.EPOCH[1] + t_buffer,
                                proj=False, picks=picks, baseline=None, preload=True)
        if epochs_all.drop_log_stats() > 0:
            print('\n** Bad epochs found. Dropping into a Python shell.')
            print(epochs_all.drop_log)
            print('\nType exit to continue.\n')
            embed()
    except:
        import pdb, traceback
        print('\n*** (tfr_export) ERROR OCCURRED WHILE EPOCHING ***')
        traceback.print_exc()
        pdb.set_trace()

    power = {}
    for evname in classes:
        export_dir = '%s/plot_%s' % (outpath, evname)
        qc.make_dirs(export_dir)
        print('\n>> Processing %s' % evname)
        freqs = cfg.FREQ_RANGE  # define frequencies of interest
        n_cycles = freqs / 2.  # different number of cycle per frequency
        if cfg.POWER_AVERAGED: # averaged over epochs
            epochs = epochs_all[evname][:]
            if len(epochs) == 0:
                print('No %s epochs. Skipping.' % evname)
                continue
            power[evname] = tfr(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                return_itc=False, decim=1, n_jobs=n_jobs)
            power[evname] = power[evname].crop(tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1])

            if cfg.EXPORT_MATLAB is True:
                # export all channels to MATLAB
                mout = '%s/%s-%s-%s.mat' % (export_dir, file_prefix, cfg.SP_FILTER, evname)
                scipy.io.savemat(mout, {'tfr':power[evname].data, 'chs':power[evname].ch_names, 'events':events, 'sfreq':raw.info['sfreq'], 'epochs':cfg.EPOCH, 'freqs':cfg.FREQ_RANGE})
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
                    print('Exported to %s' % fout)
        else: # per epoch
            for ep in range(len(epochs_all[evname])):
                epochs = epochs_all[evname][ep]
                if len(epochs) == 0:
                    print('No %s epochs. Skipping.' % evname)
                    continue
                power[evname] = tfr(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                    return_itc=False, decim=1, n_jobs=n_jobs)
                power[evname] = power[evname].crop(tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1])
                if cfg.EXPORT_MATLAB is True:
                    # export all channels to MATLAB
                    mout = '%s/%s-%s-%s-ep%02d.mat' % (export_dir, file_prefix, cfg.SP_FILTER, evname, ep + 1)
                    scipy.io.savemat(mout, {'tfr':power[evname].data, 'chs':power[evname].ch_names, 'events':events, 'sfreq':raw.info['sfreq'], 'epochs':cfg.EPOCH, 'freqs':cfg.FREQ_RANGE})
                if cfg.EXPORT_PNG is True:
                    # Inspect power for each channel
                    for ch in np.arange(len(picks)):
                        chname = raw.ch_names[picks[ch]]
                        title = 'Peri-event %s - Channel %s, Trial %d' % (evname, chname, ep + 1)
                        # mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
                        fig = power[evname].plot([ch], baseline=cfg.BS_TIMES, mode='logratio', show=False,
                            colorbar=True, title=title, vmin=cfg.VMIN, vmax=cfg.VMAX, dB=False)
                        fout = '%s/%s-%s-%s-%s-ep%02d.png' % (export_dir, file_prefix, cfg.SP_FILTER, evname, chname, ep + 1)
                        fig.savefig(fout)
                        print('Exported %s' % fout)

    if hasattr(cfg, 'POWER_DIFF'):
        export_dir = '%s/diff' % outpath
        qc.make_dirs(export_dir)
        labels = classes.keys()
        df = power[labels[0]] - power[labels[1]]
        # df.data= np.abs( df.data )
        df.data = np.log(np.abs(df.data))
        # Inspect power diff for each channel
        for ch in np.arange(len(picks)):
            chname = raw.ch_names[picks[ch]]
            title = 'Peri-event %s-%s - Channel %s' % (labels[0], labels[1], chname)

            # mode= None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
            fig = df.plot([ch], baseline=(None, 0), mode='mean', show=False,
                          colorbar=True, title=title, vmin=3.0, vmax=-3.0, dB=False)
            fout = '%s/%s-%s-diff-%s-%s-%s.jpg' % (export_dir, file_prefix, cfg.SP_FILTER, labels[0], labels[1], chname)
            print('Exporting to %s' % fout)
            fig.savefig(fout)
    print('Finished !')

def config_run(cfg_module):
    cfg = imp.load_source(cfg_module, cfg_module)
    cfg = check_cfg(cfg)
    get_tfr(cfg)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_module = input('Config file name? ')
    else:
        cfg_module = sys.argv[1]
    config_run(cfg_module)
