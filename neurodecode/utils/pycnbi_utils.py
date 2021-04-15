from __future__ import print_function, division

"""
neurodecode utility functions

Note:
When exporting to Panda Dataframes format, raw.as_data_frame() silently
scales data to the Volts unit by default, which is the convention in MNE.
Try raw.as_data_frame(scalings=dict(eeg=1.0, misc=1.0))

Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import os
import sys
import mne
import scipy.io
import importlib
import numpy as np
from pathlib import Path
import multiprocessing as mp
from scipy.signal import butter
from neurodecode import logger

mne.set_log_level('ERROR')

#----------------------------------------------------------------------
def find_event_channel(raw=None, ch_names=None):
    """
    Find the event channel using heuristics.

    Disclaimer: Not 100% guaranteed to find it.
    If raw is None, ch_names must be given.

    Parameters
    ----------
    raw : mne.io.Raw or numpy.ndarray (n_channels x n_samples)
        The data
    ch_names : list
        The channels name list

    Returns:
    --------
    int : The event channel index or None if not found.
    """
    # For numpy array
    if type(raw) == np.ndarray:
        if ch_names is not None:
            for ch_name in ch_names:
                if 'TRIGGER' in ch_name or 'STI ' in ch_name or 'TRG' in ch_name or 'CH_Event' in ch_name:
                    return ch_names.index(ch_name)

        # data range between 0 and 255 and all integers?
        for ch in range(raw.shape[0]):
            if (raw[ch].astype(int) == raw[ch]).all()\
                    and max(raw[ch]) < 256 and min(raw[ch]) == 0:
                return ch
    
    # For mne.Array
    elif hasattr(raw, 'ch_names'):
        if 'stim' in raw.get_channel_types():
            return raw.get_channel_types().index('stim')

        for ch_name in raw.ch_names:
            if 'TRIGGER' in ch_name or 'STI ' in ch_name or 'TRG' in ch_name or 'CH_Event' in ch_name:
                return raw.ch_names.index(ch_name)
    
    # For unknown data type
    else:
        if ch_names is None:
            raise ValueError('ch_names cannot be None when raw is None.')
        for ch_name in ch_names:
            if 'TRIGGER' in ch_name or 'STI ' in ch_name or 'TRG' in ch_name or 'CH_Event' in ch_name:
                return ch_names.index(ch_name)
    return None

#----------------------------------------------------------------------
def add_events_raw(rawfile, outfile, eventfile, overwrite=True):
    """
    Add events from a file and save

    Note: If the event values already exists in raw file, the new event values
    will be added to the previous value instead of replacing them.
    
    Parameters
    ----------
    rawfile : str
        The (absolute) .fif file path
    outfile : str
        The (absolute) .fif output file path
    overwrite : bool
        If True, it will overwrite the existing output file
    """
    raw = mne.io.Raw(rawfile, preload=True, proj=False)
    events = mne.read_events(eventfile)
    raw.add_events(events, stim_channel='TRIGGER', replace=False)
    raw.save(outfile, overwrite=overwrite)

#----------------------------------------------------------------------
def rereference(raw, ref_new, ref_old=None):
    """
    Reference to new channels. raw object is modified in-place for efficiency.

    The average of the new reference channel values are substracted from all channel values.

    Parameters
    ----------
    raw : mne.io.RawArray
        The data
    ref_new : None | list of str (RawArray) | list of int (numpy array)
        Channel(s) to re-reference, e.g. M1, M2.
    ref_old: None | str
        Channel to recover, assuming this channel was originally used as a reference.
    """

    # Re-reference and recover the original reference channel values if possible
    if type(raw) == np.ndarray:
        if raw_ch_old is not None:
            logger.error('Recovering original reference channel is not yet supported for numpy arrays.')
            raise NotImplementedError
        if type(raw_ch_new[0]) is not int:
            logger.error('Channels must be integer values for numpy arrays')
            raise ValueError
        raw -= np.mean(raw[ref_new], axis=0)
    else:
        if ref_old is not None:
            # Add a blank (zero-valued) channel
            mne.add_reference_channels(raw, ref_old, copy=False)
        # Re-reference
        mne.set_eeg_reference(raw, ref_new, copy=False)

#----------------------------------------------------------------------
def preprocess(raw, sfreq=None, spatial=None, spatial_ch=None, spectral=None, spectral_ch=None,
               notch=None, notch_ch=None, ch_names=None, rereference=None, decim=None, n_jobs=1):
    """
    Apply spatial, spectral, notch filters, rereference and decim.
    
    raw is modified in-place.Neurodecode puts trigger channel as index 0, data channel starts from index 1.

    Parameters
    ----------
    raw : mne.io.Raw | mne.io.RawArray | mne.Epochs | numpy.array (n_channels x n_samples)
        The raw data (numpy.array type assumes the data has only pure EEG channnels without event channels)
    sfreq : float
        Only required if raw is numpy array.
    spatial: None | 'car' | 'laplacian'
        Spatial filter type.
    spatial_ch: None | list (for CAR) | dict (for LAPLACIAN)
        Reference channels for spatial filtering. May contain channel names.
        'car': channel indices used for CAR filtering. If None, use all channels except the trigger channel (index 0).
        'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
    spectral : None | [l_freq, h_freq]
        Spectral filter.
        if l_freq is None: lowpass filter is applied.
        if h_freq is None: highpass filter is applied.
        if l_freq < h_freq: bandpass filter is applied.
        if l_freq > h_freq: band-stop filter is applied.
    spectral_ch : None | list
        Channel picks for spectral filtering. May contain channel names.
    notch: None | float | list
        Notch filter.
    notch_ch: None | list
        Channel picks for notch filtering. May contain channel names.
    ch_names: None | list
        If raw is numpy array and channel picks are list of strings, ch_names will
        be used as a look-up table to convert channel picks to channel numbers.
    rereference : Unknown
        Not supported yet.
    decim: None | int
        Apply low-pass filter and decimate (downsample). sfreq must be given. Ignored if 1.

    Output
    ------
    Same input data structure.

    Note: To save computation time, input data may be modified in-place.
    TODO: Add an option to disable in-place modification.
    """

    # Check datatype
    if type(raw) == np.ndarray:
        # Numpy array: assume we don't have event channel
        data = raw
        assert sfreq is not None and sfreq > 0, 'Wrong sfreq value.'
        assert 2 <= len(data.shape) <= 3, 'Unknown data shape. The dimension must be 2 or 3.'
        if len(data.shape) == 3:
            n_channels = data.shape[1]
        elif len(data.shape) == 2:
            n_channels = data.shape[0]
        eeg_channels = list(range(n_channels))
        if decim is not None and decim != 1:
            if sfreq is None:
                logger.error('Decimation cannot be applied if sfreq is None.')
                raise ValueError
    else:
        # MNE Raw object: exclude event channel
        ch_names = raw.ch_names
        data = raw._data
        sfreq = raw.info['sfreq']
        assert 2 <= len(data.shape) <= 3, 'Unknown data shape. The dimension must be 2 or 3.'
        if len(data.shape) == 3:
            # assert type(raw) is mne.epochs.Epochs
            n_channels = data.shape[1]
        elif len(data.shape) == 2:
            n_channels = data.shape[0]
        eeg_channels = list(range(n_channels))
        tch = find_event_channel(raw)
        if tch is None:
            logger.warning('No trigger channel found. Using all channels.')
        else:
            tch_name = ch_names[tch]
            eeg_channels.pop(tch)

    # Re-reference channels
    if rereference is not None:
        logger.error('re-referencing not implemented yet. Sorry.')
        raise NotImplementedError
    
    # Apply spatial filter
    if spatial is None:
        pass
    
    elif spatial == 'car':
    
        if spatial_ch is None:
            spatial_ch = eeg_channels
            logger.warning('preprocess(): For CAR, no specified channels, all channels selected')
        elif len(spatial_ch):
            if type(spatial_ch[0]) == str:
                assert ch_names is not None, 'preprocess(): ch_names must not be None'
                spatial_ch_i = [ch_names.index(c) for c in spatial_ch]
            else:
                spatial_ch_i = spatial_ch
    
            if len(spatial_ch_i) > 1:
                if len(data.shape) == 2:
                    data[spatial_ch_i] -= np.mean(data[spatial_ch_i], axis=0)
                elif len(data.shape) == 3:
                    means = np.mean(data[:, spatial_ch_i, :], axis=1)
                    data[:, spatial_ch_i, :] -= means[:, np.newaxis, :]
                else:
                    logger.error('Unknown data shape %s' % str(data.shape))
                    raise ValueError
        else:
            logger.error('preprocess(): For CAR, no specified channels!')
            raise ValueError            
    
    elif spatial == 'laplacian':
        if type(spatial_ch) is not dict:
            logger.error('preprocess(): For laplacian, spatial_ch must be of form {CHANNEL:[NEIGHBORS], ...}')
            raise TypeError
        if type(spatial_ch.keys()[0]) == str:
            spatial_ch_i = {}
            for c in spatial_ch:
                ref_ch = ch_names.index(c)
                spatial_ch_i[ref_ch] = [ch_names.index(n) for n in spatial_ch[c]]
        else:
            spatial_ch_i = spatial_ch

        if len(spatial_ch_i) > 1:
            rawcopy = data.copy()
            for src in spatial_ch:
                nei = spatial_ch[src]
                if len(data.shape) == 2:
                    data[src] = rawcopy[src] - np.mean(rawcopy[nei], axis=0)
                elif len(data.shape) == 3:
                    data[:, src, :] = rawcopy[:, src, :] - np.mean(rawcopy[:, nei, :], axis=1)
                else:
                    logger.error('preprocess(): Unknown data shape %s' % str(data.shape))
                    raise ValueError
    else:
        logger.error('preprocess(): Unknown spatial filter %s' % spatial)
        raise ValueError

    # Downsample
    if decim is not None and decim != 1:
        if type(raw) == np.ndarray:
            data = mne.filter.resample(data, down=decim, npad='auto', window='boxcar', n_jobs=1)
        else:
            # resample() of Raw* and Epochs object internally calls mne.filter.resample()
            raw = raw.resample(raw.info['sfreq'] / decim, npad='auto', window='boxcar', n_jobs=1)
            data = raw._data
        sfreq /= decim

    # Apply spectral filter
    if spectral is not None:
        if spectral_ch is None:
            spectral_ch = eeg_channels
            logger.warning('preprocess(): For temporal filter, all channels selected')
        elif len(spatial_ch):
            if type(spectral_ch[0]) == str:
                assert ch_names is not None, 'preprocess(): ch_names must not be None'
                spectral_ch_i = [ch_names.index(c) for c in spectral_ch]
            else:
                spectral_ch_i = spectral_ch
    
            # fir_design='firwin' is especially important for ICA analysis. See:
            # http://martinos.org/mne/dev/generated/mne.preprocessing.ICA.html?highlight=score_sources#mne.preprocessing.ICA.score_sources
            mne.filter.filter_data(data, sfreq, spectral[0], spectral[1], picks=spectral_ch_i,
                                   filter_length='auto', l_trans_bandwidth='auto',
                                   h_trans_bandwidth='auto', n_jobs=n_jobs, method='fir',
                                   iir_params=None, copy=False, phase='zero',
                                   fir_window='hamming', fir_design='firwin', verbose='ERROR')
        else:
            logger.error('preprocess(): For temporal filter, no specified channels!')
            raise ValueError                     

    # Apply notch filter
    if notch is not None:
        if notch_ch is None:
            notch_ch = eeg_channels
            logger.warning('preprocess(): For temporal filter, all channels selected')
        elif len(notch_ch):
            if type(notch_ch[0]) == str:
                assert ch_names is not None, 'preprocess(): ch_names must not be None'
                notch_ch_i = [ch_names.index(c) for c in notch_ch]
            else:
                notch_ch_i = notch_ch
    
            mne.filter.notch_filter(data, Fs=sfreq, freqs=notch, notch_widths=3,
                                    picks=notch_ch_i, method='fft', n_jobs=n_jobs, copy=False)
        else:
            logger.error('preprocess(): For temporal filter, no specified channels!')
            raise ValueError             

    if type(raw) == np.ndarray:
        raw = data
    
    return raw


#----------------------------------------------------------------------
def butter_bandpass(highcut, lowcut, fs, num_ch):
    """
    Calculation of bandpass coefficients.
    Order is computed automatically.
    Note that if filter is unstable this function crashes.

    TODO: handle exceptions
    """

    low = lowcut / (0.5 * fs)
    high = highcut / (0.5 * fs)
    ord = buttord(high, low, 2, 40)
    b, a = butter(2, [low, high], btype='band')
    zi = np.zeros([a.shape[0] - 1, num_ch])
    return b, a, zi


#----------------------------------------------------------------------
def channel_names_to_index(raw, channel_names=None):
    """
    Return channel indicies among EEG channels
    """
    if channel_names is None:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    else:
        picks = []
        for c in channel_names:
            if type(c) == int:
                picks.append(c)
            elif type(c) == str:
                if c not in raw.ch_names:
                    raise IndexError('Channel %s not found in raw.ch_names' % c)
                picks.append(raw.ch_names.index(c))
            else:
                raise TypeError('channel_names is unknown format.\nchannel_names=%s' % channel_names)

    return picks

#----------------------------------------------------------------------
def raw_crop(raw, tmin, tmax):
    """
    Perform a real cropping of a Raw object

    mne.Raw.crop() updates a very confusing variable "first_samp", which reuslts
    in the mismatch of real event indices when run with mne.find_events().
    """
    trigch = find_event_channel(raw)
    ch_types = ['eeg'] * len(raw.ch_names)
    if trigch is not None:
        ch_types[trigch] = 'stim'
    info = mne.create_info(raw.ch_names, raw.info['sfreq'], ch_types)
    tmin_index = int(round(raw.info['sfreq'] * tmin))
    tmax_index = int(round(raw.info['sfreq'] * tmax))
    return mne.io.RawArray(raw._data[:, tmin_index:tmax_index], info)

#----------------------------------------------------------------------
def load_config(cfg_module):
    """
    Dynamic loading of a config file module.
    
    cfg_module = absolute path to the config file to load 
    """
    cfg_module = Path(cfg_module)
    cfg_path, cfg_name = os.path.split(cfg_module)
    sys.path.append(cfg_path)
    
    return importlib.import_module(cfg_name.split('.')[0])