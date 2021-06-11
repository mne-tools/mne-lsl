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
import mne
import mne.io
import numpy as np
import multiprocessing as mp

from .. import logger
from ..utils.timer import Timer
# from ..utils.preprocess import set_eeg_reference
from ..utils.io import read_raw_fif_multi, get_file_list
from ..utils.preprocess import preprocess as apply_preprocess

#----------------------------------------------------------------------
def feature2chz(x, fqlist, ch_names):
    """
    Label channel, frequency pair for PSD feature indices

    Parameters
    ----------
    x : list
        The features' indexes
    fqlist : list
        The frequency bands
    ch_names : list
    The channels' names

    Returns
    -------
    list : The associated channels
    list : The associated frequencies
    """

    x = np.array(x).astype('int64').reshape(-1)
    fqlist = np.array(fqlist).astype('float64')
    ch_names = np.array(ch_names)

    n_fq = len(fqlist)
    hz = fqlist[x % n_fq]
    ch = (x / n_fq).astype('int64')  # 0-based indexing

    return ch_names[ch], hz

#----------------------------------------------------------------------
def compute_features(cfg):
    '''
    Compute features using config specification.

    Performs preprocessing, epoching and feature computation.

    Parameters
    ----------
    cfg : dict
        The config dict containing all the necessary parameters/info

    Returns
    -------
    dict : The dictionnary containing:

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
    #-----------------------------------------------------------
    # Load the data from files
    ftrain = []
    for f in get_file_list(cfg.DATA_PATH, fullpath=True):
        if f[-4:] in ['.fif', '.fiff']:
            ftrain.append(f)
    if len(ftrain) > 1 and cfg.PICKED_CHANNELS is not None and type(cfg.PICKED_CHANNELS[0]) == int:
        logger.error('When loading multiple EEG files, PICKED_CHANNELS must be list of string, not integers because they may have different channel order.')
        raise RuntimeError
    raw, events = read_raw_fif_multi(ftrain)

    #-----------------------------------------------------------
    # Rereference
    # reref = cfg.REREFERENCE[cfg.REREFERENCE['selected']]
    # if reref is not None:
        # set_eeg_reference(raw, reref['New'], reref['Old'])

    #-----------------------------------------------------------
    # Load events from file
    if cfg.LOAD_EVENTS[cfg.LOAD_EVENTS['selected']] is not None:
        events = mne.read_events(cfg.LOAD_EVENTS[cfg.LOAD_EVENTS['selected']])

    #-----------------------------------------------------------
    # Load triggers from file
    trigger_def_int = set()
    for a in cfg.TRIGGER_DEF:
        trigger_def_int.add(getattr(cfg.tdef, a))
    triggers = {cfg.tdef.by_value[c]:c for c in trigger_def_int}

    #-----------------------------------------------------------
    # Downsampling
    if 'decim' not in cfg.FEATURES['PSD']:
        cfg.FEATURES['PSD']['decim'] = 1
        logger.warning('PSD["decim"] undefined. Set to 1.')

    #-----------------------------------------------------------
    # Read epochs
    try:
        # Experimental: multiple epoch ranges
        if type(cfg.EPOCH[0]) is list:
            epochs_train = []
            for ep in cfg.EPOCH:
                epoch = mne.Epochs(raw, events, triggers, tmin=ep[0], tmax=ep[1],
                    proj=False, picks=['data'], baseline=None, preload=True,
                    verbose=False, detrend=None)
                epochs_train.append(epoch)
        else:
            # Usual method: single epoch range
            epochs_train = mne.Epochs(raw, events, triggers, tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1], proj=False,
                picks=['data'], baseline=None, preload=True, verbose=False, detrend=None, on_missing='warning')
    except:
        logger.exception('Problem while epoching.')
        raise RuntimeError

    #-----------------------------------------------------------
    # Pick channels
    if cfg.PICKED_CHANNELS is None:
        chlist = list(range(len(epochs_train.info.ch_names)))
    else:
        chlist = cfg.PICKED_CHANNELS

    picks = []
    for c in chlist:
        if type(c) == int:
            picks.append(c)
        elif type(c) == str:
            picks.append(epochs_train.info.ch_names.index(c))
        else:
            logger.error('PICKED_CHANNELS has a value of unknown type %s.\nPICKED_CHANNELS=%s' % (type(c), cfg.PICKED_CHANNELS))
            raise RuntimeError
    #-----------------------------------------------------------
    #  Exclude channels
    if cfg.EXCLUDED_CHANNELS is not None:
        for c in cfg.EXCLUDED_CHANNELS:
            if type(c) == str:
                if c not in epochs_train.info.ch_names:
                    logger.warning('Exclusion channel %s does not exist. Ignored.' % c)
                    continue
                c_int = epochs_train.info.ch_names.index(c)
            elif type(c) == int:
                c_int = c
            else:
                logger.error('EXCLUDED_CHANNELS has a value of unknown type %s.\nEXCLUDED_CHANNELS=%s' % (type(c), cfg.EXCLUDED_CHANNELS))
                raise RuntimeError
            if c_int in picks:
                del picks[picks.index(c_int)]
    if max(picks) > len(epochs_train.info.ch_names):
        logger.error('"picks" has a channel index %d while there are only %d channels.' % (max(picks), len(epochs_train.info.ch_names)))
        raise ValueError

    #-----------------------------------------------------------
    #  Preprocessing
    if cfg.FEATURES['selected'] == 'PSD':
        # TODO: This is not compatible with the new preprocess structure.
        preprocess = dict(sfreq=epochs_train.info['sfreq'],
            spatial=cfg.SP_FILTER,
            spatial_ch=cfg.SP_CHANNELS,
            spectral=cfg.TP_FILTER[cfg.TP_FILTER['selected']],
            spectral_ch=cfg.TP_CHANNELS,
            notch=cfg.NOTCH_FILTER[cfg.NOTCH_FILTER['selected']],
            notch_ch=cfg.NOTCH_CHANNELS,
            multiplier=cfg.MULTIPLIER,
            ch_names=epochs_train.ch_names,
            rereference=cfg.REREFERENCE[cfg.REREFERENCE['selected']],
            decim=cfg.FEATURES['PSD']['decim'],
            n_jobs=cfg.N_JOBS
        )

        #-----------------------------------------------------------
        # Compute features
        featdata = _get_psd_feature(epochs_train, cfg.EPOCH, cfg.FEATURES['PSD'], picks=picks, preprocess=preprocess, n_jobs=cfg.N_JOBS)

    #-----------------------------------------------------------
    #  Other possible features
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

    featdata['sfreq'] = epochs_train.info['sfreq']
    featdata['ch_names'] = [epochs_train.info.ch_names[p] for p in picks]
    featdata['picks'] = list(range(len(featdata['ch_names'])))

    return featdata

#----------------------------------------------------------------------
def _slice_win(epochs_data, w_starts, w_length, psde, picks=None, title=None, preprocess=None, verbose=False):
    '''
    Compute PSD values over sliding windows accross one epoch

    Parameters
    ----------
    epochs_data : raw epoch data
        The data to slice ([channels]x[samples]):
    w_starts : list
        The starting indices of the slices
    w_length : int
        The window length in number of samples
    psde: MNE PSDEstimator
        The PSD estimator used to compute PSD
    picks : list
        The subset of channels within epochs_data
    title : str
        To print out the title associated with PID
    preprocess : dict
        The parameters needed by preprocess.preprocess() with the following keys:
            sfreq, spatial, spatial_ch, spectral, spectral_ch, notch, notch_ch,
            multiplier, ch_names, rereference, decim, n_jobs

    Returns
    -------
    numpy.Array : The PSD for each slices [windows] x [channels x freqs] or [windows] x [channels] x [freqs]
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

        #  Apply preprocessing on slices
        if preprocess is not None:
            # TODO: This is not compatible with the new preprocess structure.
            window = apply_preprocess(window,
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

        # Keep only the channels of interest
        window = window[picks, :]

        # dimension: psde.transform( [epochs x channels x times] )
        psd = psde.transform(window.reshape((1, window.shape[0], window.shape[1])))
        psd = psd.reshape((psd.shape[0], psd.shape[1] * psd.shape[2]))

        #if picks:
            #psd = psd[0][picks]
            #psd = psd.reshape((1, len(psd)))

        if X is None:
            X = psd
        else:
            X = np.concatenate((X, psd), axis=0)

        if verbose == True:
            logger.info('[PID %d] processing frame %d / %d' % (os.getpid(), n, w_starts[-1]))

    return X

#----------------------------------------------------------------------
def _get_psd(epochs, psde, wlen, wstep, picks=None, flatten=True, preprocess=None, decim=1, n_jobs=1):
    """
    Compute multi-taper PSDs over a sliding window accross all provided epochs

    Parameters
    ----------
    epochs : MNE Epochs
        The epochs to compute PSDs
    psde : MNE PSDEstimator
        The PSD estimator used to compute PSD
    wlen : int
        The window length in samples
    wstep : int
        The window step in samples
    picks : list
        The channels to use; use all if None
    flatten : bool
        If True, generate concatenated feature vectors ([windows] x [channels x freqs])
    preprocess : dict
        The preprocessing info (spatial, temporal, notch, reref...)
    n_jobs: int
        The number of cores to use, None = use all cores

    Returns
    -------
    numpy.Array : if flatten==True X_data: [epochs] x [windows] x [channels*freqs]
    numpy.Array : [epochs] x [windows]

    TODO:
        Accept input as numpy array as well, in addition to Epochs object
    """

    tm = Timer()

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
            results.append(_slice_win(epochs_data[ep], w_starts, wlen, psde, picks, title, preprocess, False))
        else:
            # parallel psd computation
            results.append(pool.apply_async(_slice_win, [epochs_data[ep], w_starts, wlen, psde, picks, title, preprocess, False]))

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

#----------------------------------------------------------------------
def _get_psd_feature(epochs_train, window, psdparam, picks=None, preprocess=None, n_jobs=1):
    """
    Wrapper for _get_psd() adding meta information.

    Parameters
    ----------
    epochs_train : mne.Epochs object or list of mne.Epochs object.
        The epoched data
    window : [t_start, t_end]. Time window range for computing PSD.
    psdparam: {fmin:float, fmax:float, wlen:float, wstep:int, decim:int}.
              fmin, fmax in Hz, wlen in seconds, wstep in number of samples.
    picks: Channels to compute features from.

    Returns
    -------
    dict : Dict containing the computed PSD features.
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

    logger.info('PSD computation')
    if type(epochs_train) is list:
        X_all = []
        for i, ep in enumerate(epochs_train):
            X, Y_data = _get_psd(ep, psde, w_frames[i], psdparam['wstep'], picks, n_jobs=n_jobs, preprocess=preprocess, decim=psdparam['decim'])
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
        X_data, Y_data = _get_psd(epochs_train, psde, w_frames, psdparam['wstep'], picks, n_jobs=n_jobs, preprocess=preprocess, decim=psdparam['decim'])

    # assign relative timestamps for each feature. time reference is the leading edge of a window.
    w_starts = np.arange(0, epochs_train.get_data().shape[2] - w_frames, psdparam['wstep'])
    t_features = w_starts / sfreq + psdparam['wlen'] + window[0]

    return dict(X_data=X_data, Y_data=Y_data, wlen=psdparam['wlen'], w_frames=w_frames, psde=psde, times=t_features, decim=psdparam['decim'])
