from __future__ import print_function, division

"""
features.py

Feature computation module.


Kyuhwa Lee, 2019
Swiss Federal Institute of Technology Lausanne (EPFL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import sys
import imp
import mne
import mne.io
import pycnbi
import timeit
import platform
import traceback
import numpy as np
import multiprocessing as mp
import sklearn.metrics as skmetrics
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from mne import Epochs, pick_types
from pycnbi import logger
from pycnbi.decoder.rlda import rLDA
from builtins import input
from IPython import embed  # for debugging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def slice_win(epochs_data, w_starts, w_length, psde, picks=None, title=None, flatten=True, preprocess=None, verbose=False):
    '''
    Compute PSD values of a sliding window

    Params
        epochs_data ([channels]x[samples]): raw epoch data
        w_starts (list): starting indices of sample segments
        w_length (int): window length in number of samples
        psde: MNE PSDEstimator object
        picks (list): subset of channels within epochs_data
        title (string): print out the title associated with PID
        flatten (boolean): generate concatenated feature vectors
            If True: X = [windows] x [channels x freqs]
            If False: X = [windows] x [channels] x [freqs]
        preprocess (dict): None or parameters for pycnbi_utils.preprocess() with the following keys:
            sfreq, spatial, spatial_ch, spectral, spectral_ch, notch, notch_ch,
            multiplier, ch_names, rereference, decim, n_jobs
    Returns:
        [windows] x [channels*freqs] or [windows] x [channels] x [freqs]
    '''

    # raise error for wrong indexing
    def WrongIndexError(Exception):
        logger.error('%s' % Exception)

    if type(w_length) is not int:
        logger.warning('w_length type is %s. Converting to int.' % type(w_length))
        w_length = int(w_length)
    if title is None:
        title = '[PID %d] Frames %d-%d' % (os.getpid(), w_starts[0], w_starts[-1]+w_length-1)
    else:
        title = '[PID %d] %s' % (os.getpid(), title)
    if preprocess is not None and preprocess['decim'] != 1:
        title += ' (decim factor %d)' % preprocess['decim']
    logger.info(title)

    X = None
    for n in w_starts:
        n = int(round(n))
        if n >= epochs_data.shape[1]:
            logger.error('w_starts has an out-of-bounds index %d for epoch length %d.' % (n, epochs_data.shape[1]))
            raise WrongIndexError
        window = epochs_data[:, n:(n + w_length)]

        if preprocess is not None:
            window = pu.preprocess(window,
                sfreq=preprocess['sfreq'],
                spatial=preprocess['spatial'],
                spatial_ch=preprocess['spatial_ch'],
                spectral=preprocess['spectral'],
                spectral_ch=preprocess['spectral_ch'],
                notch=preprocess['notch'],
                notch_ch=preprocess['notch_ch'],
                multiplier=preprocess['multiplier'],
                ch_names=preprocess['ch_names'],
                rereference=preprocess['rereference'],
                decim=preprocess['decim'],
                n_jobs=preprocess['n_jobs'])

        # dimension: psde.transform( [epochs x channels x times] )
        psd = psde.transform(window.reshape((1, window.shape[0], window.shape[1])))
        psd = psd.reshape((psd.shape[0], psd.shape[1] * psd.shape[2]))
        if picks:
            psd = psd[0][picks]
            psd = psd.reshape((1, len(psd)))

        if X is None:
            X = psd
        else:
            X = np.concatenate((X, psd), axis=0)

        if verbose == True:
            logger.info('[PID %d] processing frame %d / %d' % (os.getpid(), n, w_starts[-1]))

    return X


def get_psd(epochs, psde, wlen, wstep, picks=None, flatten=True, preprocess=None, decim=1, n_jobs=1):
    """
    Compute multi-taper PSDs over a sliding window

    Input
    =====
    epochs: MNE Epochs object
    psde: MNE PSDEstimator object
    wlen: window length in frames
    wstep: window step in frames
    picks: channels to be used; use all if None
    flatten: boolean, see Returns section
    n_jobs: nubmer of cores to use, None = use all cores

    Output
    ======
    if flatten==True:
        X_data: [epochs] x [windows] x [channels*freqs]
    else:
        X_data: [epochs] x [windows] x [channels] x [freqs]
    y_data: [epochs] x [windows]

    TODO:
        Accept input as numpy array as well, in addition to Epochs object
    """

    tm = qc.Timer()

    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if n_jobs > 1:
        logger.info('Opening a pool of %d workers' % n_jobs)
        pool = mp.Pool(n_jobs)

    # compute PSD from sliding windows of each epoch
    labels = epochs.events[:, -1]
    epochs_data = epochs.get_data()
    w_starts = np.arange(0, epochs_data.shape[2] - wlen, wstep)
    X_data = None
    y_data = None
    results = []
    for ep in np.arange(len(labels)):
        title = 'Epoch %d / %d, Frames %d-%d' % (ep+1, len(labels), w_starts[0], w_starts[-1] + wlen - 1)
        if n_jobs == 1:
            # no multiprocessing
            results.append(slice_win(epochs_data[ep], w_starts, wlen, psde, picks, title, True, preprocess))
        else:
            # parallel psd computation
            results.append(pool.apply_async(slice_win, [epochs_data[ep], w_starts, wlen, psde, picks, title, True, preprocess]))

    for ep in range(len(results)):
        if n_jobs == 1:
            r = results[ep]
        else:
            r = results[ep].get()  # windows x features
        X = r.reshape((1, r.shape[0], r.shape[1]))  # 1 x windows x features
        if X_data is None:
            X_data = X
        else:
            X_data = np.concatenate((X_data, X), axis=0)

        # speed comparison: http://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
        y = np.empty((1, r.shape[0]))  # 1 x windows
        y.fill(labels[ep])
        if y_data is None:
            y_data = y
        else:
            y_data = np.concatenate((y_data, y), axis=0)

    # close pool
    if n_jobs > 1:
        pool.close()
        pool.join()

    logger.info('Feature computation took %d seconds.' % tm.sec())

    if flatten:
        return X_data, y_data.astype(np.int)
    else:
        xs = X_data.shape
        nch = len(epochs.ch_names)
        return X_data.reshape(xs[0], xs[1], nch, int(xs[2] / nch)), y_data.astype(np.int)


def get_psd_feature(epochs_train, window, psdparam, picks=None, preprocess=None, n_jobs=1):
    """
    Wrapper for get_psd() adding meta information.

    Input
    =====
    epochs_train: mne.Epochs object or list of mne.Epochs object.
    window: [t_start, t_end]. Time window range for computing PSD.
    psdparam: {fmin:float, fmax:float, wlen:float, wstep:int, decim:int}.
              fmin, fmax in Hz, wlen in seconds, wstep in number of samples.
    picks: Channels to compute features from.

    Output
    ======
    dict object containing computed features.
    """

    if type(window[0]) is list:
        sfreq = epochs_train[0].info['sfreq']
        wlen = []
        w_frames = []
        # multiple PSD estimators, defined for each epoch
        if type(psdparam) is list:
            '''
            TODO: implement multi-window PSD for each epoch
            assert len(psdparam) == len(window)
            for i, p in enumerate(psdparam):
                if p['wlen'] is None:
                    wl = window[i][1] - window[i][0]
                else:
                    wl = p['wlen']
                wlen.append(wl)
                w_frames.append(int(sfreq * wl))
            '''
            logger.error('Multiple psd function not implemented yet.')
            raise NotImplementedError
        # same PSD estimator for all epochs
        else:
            for i, e in enumerate(window):
                if psdparam['wlen'] is None:
                    wl = window[i][1] - window[i][0]
                else:
                    wl = psdparam['wlen']
                assert wl > 0
                wlen.append(wl)
                w_frames.append(int(round(sfreq * wl)))
    else:
        sfreq = epochs_train.info['sfreq']
        wlen = window[1] - window[0]
        if psdparam['wlen'] is None:
            psdparam['wlen'] = wlen
        w_frames = int(round(sfreq * psdparam['wlen']))  # window length in number of samples(frames)
    if 'decim' not in psdparam or psdparam['decim'] is None:
        psdparam['decim'] = 1

    psde_sfreq = sfreq / psdparam['decim']
    psde = mne.decoding.PSDEstimator(sfreq=psde_sfreq, fmin=psdparam['fmin'], fmax=psdparam['fmax'],
        bandwidth=None, adaptive=False, low_bias=True, n_jobs=1, normalization='length', verbose='WARNING')

    logger.info_green('PSD computation')
    if type(epochs_train) is list:
        X_all = []
        for i, ep in enumerate(epochs_train):
            X, Y_data = get_psd(ep, psde, w_frames[i], psdparam['wstep'], picks, n_jobs=n_jobs, preprocess=preprocess, decim=psdparam['decim'])
            X_all.append(X)
        # concatenate along the feature dimension
        # feature index order: window block x channel block x frequency block
        # feature vector = [window1, window2, ...]
        # where windowX = [channel1, channel2, ...]
        # where channelX = [freq1, freq2, ...]
        X_data = np.concatenate(X_all, axis=2)
    else:
        # feature index order: channel block x frequency block
        # feature vector = [channel1, channel2, ...]
        # where channelX = [freq1, freq2, ...]
        X_data, Y_data = get_psd(epochs_train, psde, w_frames, psdparam['wstep'], picks, n_jobs=n_jobs, preprocess=preprocess, decim=psdparam['decim'])

    # assign relative timestamps for each feature. time reference is the leading edge of a window.
    w_starts = np.arange(0, epochs_train.get_data().shape[2] - w_frames, psdparam['wstep'])
    t_features = w_starts / sfreq + psdparam['wlen'] + window[0]
    return dict(X_data=X_data, Y_data=Y_data, wlen=wlen, w_frames=w_frames, psde=psde, times=t_features, decim=psdparam['decim'])


def get_timelags(epochs, wlen, wstep, downsample=1, picks=None):
    """
    (DEPRECATED FUNCTION)
    Get concatenated timelag features

    TODO: Unit test.

    Input
    =====
    epochs: input signals
    wlen: window length (# time points) in downsampled data
    wstep: window step in downsampled data
    downsample: downsample signal to be 1/downsample length
    picks: ignored for now

    Output
    ======
    X: [epochs] x [windows] x [channels*freqs]
    y: [epochs] x [labels]
    """
    
    '''
    wlen = int(wlen)
    wstep = int(wstep)
    downsample = int(downsample)
    X_data = None
    y_data = None
    labels = epochs.events[:, -1]  # every epoch must have event id
    epochs_data = epochs.get_data()
    n_channels = epochs_data.shape[1]
    # trim to the nearest divisible length
    epoch_ds_len = int(epochs_data.shape[2] / downsample)
    epoch_len = downsample * epoch_ds_len
    range_epochs = np.arange(epochs_data.shape[0])
    range_channels = np.arange(epochs_data.shape[1])
    range_windows = np.arange(epoch_ds_len - wlen, 0, -wstep)
    X_data = np.zeros((len(range_epochs), len(range_windows), wlen * n_channels))

    # for each epoch
    for ep in range_epochs:
        epoch = epochs_data[ep, :, :epoch_len]
        ds = qc.average_every_n(epoch.reshape(-1), downsample)  # flatten to 1-D, then downsample
        epoch_ds = ds.reshape(n_channels, -1)  # recover structure to channel x samples
        # for each window over all channels
        for i in range(len(range_windows)):
            w = range_windows[i]
            X = epoch_ds[:, w:w + wlen].reshape(1, -1)  # our feature vector
            X_data[ep, i, :] = X

        # fill labels
        y = np.empty((1, len(range_windows)))  # 1 x windows
        y.fill(labels[ep])
        if y_data is None:
            y_data = y
        else:
            y_data = np.concatenate((y_data, y), axis=0)

    return X_data, y_data
    '''
    logger.error('This function is deprecated.')
    raise NotImplementedError

def feature2chz(x, fqlist, ch_names):
    """
    Label channel, frequency pair for PSD feature indices

    Input
    =====
    x: feature index
    fqlist: list of frequency bands
    ch_names: list of complete channel names

    Output
    ======
    (channel, frequency)

    """

    x = np.array(x).astype('int64').reshape(-1)
    fqlist = np.array(fqlist).astype('float64')
    ch_names = np.array(ch_names)

    n_fq = len(fqlist)
    hz = fqlist[x % n_fq]
    ch = (x / n_fq).astype('int64')  # 0-based indexing

    return ch_names[ch], hz


def cva_features(datadir):
    """
    (DEPRECATED FUNCTION)
    """
    for fin in qc.get_file_list(datadir, fullpath=True):
        if fin[-4:] != '.gdf': continue
        fout = fin + '.cva'
        if os.path.exists(fout):
            logger.info('Skipping', fout)
            continue
        logger.info("cva_features('%s')" % fin)
        qc.matlab("cva_features('%s')" % fin)


def compute_features(cfg):
    '''
    Compute features using config specification.

    Performs preprocessing, epcoching and feature computation.

    Input
    =====
    Config file object

    Output
    ======
    Feature data in dictionary
    - X_data: feature vectors
    - Y_data: feature labels
    - wlen: window length in seconds
    - w_frames: window length in frames
    - psde: MNE PSD estimator object
    - picks: channels used for feature computation
    - sfreq: sampling frequency
    - ch_names: channel names
    - times: feature timestamp (leading edge of a window)
    '''
    # Preprocessing, epoching and PSD computation
    ftrain = []
    for f in qc.get_file_list(cfg.DATA_PATH, fullpath=True):
        if f[-4:] in ['.fif', '.fiff']:
            ftrain.append(f)
    if len(ftrain) > 1 and cfg.PICKED_CHANNELS is not None and type(cfg.PICKED_CHANNELS[0]) == int:
        logger.error('When loading multiple EEG files, PICKED_CHANNELS must be list of string, not integers because they may have different channel order.')
        raise RuntimeError
    raw, events = pu.load_multi(ftrain)
    
    reref = cfg.REREFERENCE[cfg.REREFERENCE['selected']]
    if reref is not None:
        #pu.rereference(raw, reref['new'], reref['old'])
        logger.error('Sorry! Channel re-referencing is under development.')
        raise NotImplementedError
    
    if cfg.LOAD_EVENTS[cfg.LOAD_EVENTS['selected']] is not None:
        events = mne.read_events(cfg.LOAD_EVENTS[cfg.LOAD_EVENTS['selected']])
    
    trigger_def_int = set()
    for a in cfg.TRIGGER_DEF:
        trigger_def_int.add(getattr(cfg.tdef, a))
    triggers = {cfg.tdef.by_value[c]:c for c in trigger_def_int}

    # Pick channels
    if cfg.PICKED_CHANNELS is None:
        chlist = [int(x) for x in pick_types(raw.info, stim=False, eeg=True)]
    else:
        chlist = cfg.PICKED_CHANNELS
    picks = []
    for c in chlist:
        if type(c) == int:
            picks.append(c)
        elif type(c) == str:
            picks.append(raw.ch_names.index(c))
        else:
            logger.error('PICKED_CHANNELS has a value of unknown type %s.\nPICKED_CHANNELS=%s' % (type(c), cfg.PICKED_CHANNELS))
            raise RuntimeError
    if cfg.EXCLUDED_CHANNELS is not None:
        for c in cfg.EXCLUDED_CHANNELS:
            if type(c) == str:
                if c not in raw.ch_names:
                    logger.warning('Exclusion channel %s does not exist. Ignored.' % c)
                    continue
                c_int = raw.ch_names.index(c)
            elif type(c) == int:
                c_int = c
            else:
                logger.error('EXCLUDED_CHANNELS has a value of unknown type %s.\nEXCLUDED_CHANNELS=%s' % (type(c), cfg.EXCLUDED_CHANNELS))
                raise RuntimeError
            if c_int in picks:
                del picks[picks.index(c_int)]
    if max(picks) > len(raw.ch_names):
        logger.error('"picks" has a channel index %d while there are only %d channels.' % (max(picks), len(raw.ch_names)))
        raise ValueError
    if hasattr(cfg, 'SP_CHANNELS') and cfg.SP_CHANNELS is not None:
        logger.warning('SP_CHANNELS parameter is not supported yet. Will be set to PICKED_CHANNELS.')
    if hasattr(cfg, 'TP_CHANNELS') and cfg.TP_CHANNELS is not None:
        logger.warning('TP_CHANNELS parameter is not supported yet. Will be set to PICKED_CHANNELS.')
    if hasattr(cfg, 'NOTCH_CHANNELS') and cfg.NOTCH_CHANNELS is not None:
        logger.warning('NOTCH_CHANNELS parameter is not supported yet. Will be set to PICKED_CHANNELS.')
    if 'decim' not in cfg.FEATURES['PSD']:
        cfg.FEATURES['PSD']['decim'] = 1
        logger.warning('PSD["decim"] undefined. Set to 1.')

    # Read epochs
    try:
        # Experimental: multiple epoch ranges
        if type(cfg.EPOCH[0]) is list:
            epochs_train = []
            for ep in cfg.EPOCH:
                epoch = Epochs(raw, events, triggers, tmin=ep[0], tmax=ep[1],
                    proj=False, picks=picks, baseline=None, preload=True,
                    verbose=False, detrend=None)
                # Channels are already selected by 'picks' param so use all channels.
                '''
                epoch = pu.preprocess(epoch, spatial=cfg.SP_FILTER, spatial_ch=None, spectral=cfg.TP_FILTER, spectral_ch=None,
                    notch=cfg.NOTCH_FILTER, notch_ch=None, multiplier=cfg.MULTIPLIER, n_jobs=cfg.N_JOBS, decim=cfg.FEATURES['PSD']['decim'])
                '''
                epochs_train.append(epoch)
        else:
            # Usual method: single epoch range
            epochs_train = Epochs(raw, events, triggers, tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1], proj=False,
                picks=picks, baseline=None, preload=True, verbose=False, detrend=None, on_missing='warning')
            # Channels are already selected by 'picks' param so use all channels.
            '''
            epochs_train = pu.preprocess(epochs_train, spatial=cfg.SP_FILTER, spatial_ch=None, spectral=cfg.TP_FILTER, spectral_ch=None,
                notch=cfg.NOTCH_FILTER, notch_ch=None, multiplier=cfg.MULTIPLIER, n_jobs=cfg.N_JOBS, decim=cfg.FEATURES['PSD']['decim'])
            '''
    except:
        logger.exception('Problem while epoching.')
        raise RuntimeError

    label_set = np.unique(triggers.values())

    # Compute features
    if cfg.FEATURES['selected'] == 'PSD':
        preprocess = dict(sfreq=epochs_train.info['sfreq'],
            spatial=cfg.SP_FILTER,
            spatial_ch=None,
            spectral=cfg.TP_FILTER[cfg.TP_FILTER['selected']],
            spectral_ch=None,
            notch=cfg.NOTCH_FILTER[cfg.NOTCH_FILTER['selected']],
            notch_ch=None,
            multiplier=cfg.MULTIPLIER,
            ch_names=None,
            rereference=None,
            decim=cfg.FEATURES['PSD']['decim'],
            n_jobs=cfg.N_JOBS
        )
        featdata = get_psd_feature(epochs_train, cfg.EPOCH, cfg.FEATURES['PSD'], picks=None, preprocess=preprocess, n_jobs=cfg.N_JOBS)
    elif cfg.FEATURES == 'TIMELAG':
        '''
        TODO: Implement multiple epochs for timelag feature
        '''
        logger.error('MULTIPLE EPOCHS NOT IMPLEMENTED YET FOR TIMELAG FEATURE.')
        raise NotImplementedError
    elif cfg.FEATURES == 'WAVELET':
        '''
        TODO: Implement multiple epochs for wavelet feature
        '''
        logger.error('MULTIPLE EPOCHS NOT IMPLEMENTED YET FOR WAVELET FEATURE.')
        raise NotImplementedError
    else:
        logger.error('%s feature type is not supported.' % cfg.FEATURES)
        raise NotImplementedError

    featdata['picks'] = picks
    featdata['sfreq'] = raw.info['sfreq']
    featdata['ch_names'] = raw.ch_names
    return featdata
