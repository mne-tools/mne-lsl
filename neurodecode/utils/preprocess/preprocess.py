from __future__ import print_function, division

"""
Note:
When exporting to Panda Dataframes format, raw.as_data_frame() silently
scales data to the Volts unit by default, which is the convention in MNE.
Try raw.as_data_frame(scalings=dict(eeg=1.0, misc=1.0))

Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import os
import mne
import numpy as np

import neurodecode.utils.io as io

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
    raw : mne.io.Raw | numpy.ndarray (n_channels x n_samples)
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
def rereference(raw, ref_new, ref_old=None):
    """
    Reference to new channels. raw object is modified in-place for efficiency.

    The average of the new reference channel values are substracted from all channel values.

    Parameters
    ----------
    raw : mne.io.RawArray
        The data
    ref_new : None | list of str (for RawArray) | list of int (for numpy array)
        Channel(s) to re-reference, e.g. M1, M2.
    ref_old: | list of str (for RawArray) | list of int (for numpy array)
        Channel to recover, assuming this channel was originally used as a reference.
    """

    # Re-reference and recover the original reference channel values if possible
    if type(raw) == np.ndarray:
        if ref_old is not None:
            logger.error('Recovering original reference channel is not yet supported for numpy arrays.')
            raise NotImplementedError
        if type(ref_new[0]) is not int:
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
               notch=None, notch_ch=None, multiplier=1, ch_names=None, rereference=None, decim=None, n_jobs=1):
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
    multiplier : int
        Used for changing eeg data unit. Mne assumes Volts.
    ch_names: None | list
        If raw is numpy array and channel picks are list of strings, ch_names will
        be used as a look-up table to convert channel picks to channel numbers.
    rereference : list
        List of new and old references. [[ref_new_1, ..., ref_new_N], [ref_old_1, ..., ref_old_N]]
    decim: None | int
        Apply low-pass filter and decimate (downsample). sfreq must be given. Ignored if 1.

    Output
    ------
    Same input data structure.

    Note: To save computation time, input data may be modified in-place.
    TODO: Add an option to disable in-place modification.
    """

    # Re-reference channels
    if rereference is not None:
        rereference(raw, rereference[0], rereference[1])    

    # Downsample
    if decim is not None and decim != 1:
        assert sfreq is not None and sfreq > 0, 'Wrong sfreq value.'
        if type(raw) == np.ndarray:
            logger.error('Decimation cannot be applied if sfreq is None.')
            raise ValueError        
        raw, sfreq = _apply_downsampling(raw, decim, sfreq, n_jobs)

    # Format data to numpy array
    data, eeg_channels, ch_names= _format_eeg_data_for_preprocessing(raw, ch_names)
    
    # Do unit conversion
    if multiplier != 1:
        data[eeg_channels] *= multiplier
    
    # Apply spatial filter
    if spatial is not None:
        _apply_spatial_filtering(data, spatial, eeg_channels, spatial_ch, ch_names)

    # Apply spectral filter
    if spectral is not None:
        _apply_spectral_filtering(data, spectral_ch, eeg_channels, ch_names, n_jobs, spectral, sfreq)

    # Apply notch filter
    if notch is not None:
        _apply_notch_filtering(data, notch, notch_ch, eeg_channels, ch_names, n_jobs, sfreq)

    if type(raw) == np.ndarray:
        raw = data
    
    return raw

#----------------------------------------------------------------------
def _apply_spatial_filtering(data, spatial, eeg_channels, spatial_ch, ch_names):
    """
    Apply spatial filtering to the data. Supported: CAR or Laplacian.
    """
    if spatial == 'car':
        _apply_car_filtering(data, spatial_ch, eeg_channels, ch_names)
        
    elif spatial == 'laplacian':
        _apply_laplacian_filtering(data, spatial_ch, ch_names)
    else:
        logger.error('preprocess(): Unknown spatial filter %s' % spatial)
        raise ValueError

#----------------------------------------------------------------------
def _apply_notch_filtering(data, notch, notch_ch, eeg_channels, ch_names, n_jobs, sfreq):
    """
    Apply a notch filter to the data.
    """
    if notch_ch is None:
        notch_ch = eeg_channels
        logger.warning('preprocess(): For notch filter, all channels selected')
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

#----------------------------------------------------------------------
def _apply_downsampling(raw, decim, sfreq, n_jobs):
    """
    Apply downsampling with factor decim.
    """
    if type(raw) == np.ndarray:
        raw = mne.filter.resample(raw, down=decim, npad='auto', window='boxcar', n_jobs=n_jobs)
    else:
        # resample() of Raw* and Epochs object internally calls mne.filter.resample()
        raw = raw.resample(raw.info['sfreq'] / decim, npad='auto', window='boxcar', n_jobs=n_jobs)
        sfreq = raw.info['sfreq']
    
    sfreq /= decim
    
    return raw, sfreq

#----------------------------------------------------------------------
def _apply_spectral_filtering(data, spectral_ch, eeg_channels, ch_names, n_jobs, spectral, sfreq):
    """
    Apply temporal filtering
    """
    if spectral_ch is None:
        spectral_ch = eeg_channels
        logger.warning('preprocess(): For temporal filter, all channels selected')
    elif len(spectral_ch):
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

#----------------------------------------------------------------------
def _apply_laplacian_filtering(data, spatial_ch, ch_names):
    """
    Apply the lapacian spatial filtering
    """
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
    
    return data

#----------------------------------------------------------------------
def _apply_car_filtering(data, spatial_ch, eeg_channels, ch_names):
    """
    Apply Common Average Reference to data.
    """
    if spatial_ch is None or not len(spatial_ch):
        spatial_ch = eeg_channels
        logger.warning('preprocess(): For CAR, no specified channels, all channels selected')
        
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
    
    return data

#----------------------------------------------------------------------
def _format_eeg_data_for_preprocessing(raw, ch_names=None):
    # Check datatype
    if type(raw) == np.ndarray:
        # Numpy array: assume we don't have event channel
        data = raw
        assert 2 <= len(data.shape) <= 3, 'Unknown data shape. The dimension must be 2 or 3.'
        if len(data.shape) == 3:
            n_channels = data.shape[1]
        elif len(data.shape) == 2:
            n_channels = data.shape[0]
        eeg_channels = list(range(n_channels))
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
            eeg_channels.pop(tch)
            
    return data, eeg_channels, ch_names

#----------------------------------------------------------------------
def _check_fif_path(rawfile):
    """
    Ensure that the fif file exists.
    """
    if not os.path.exists(rawfile):
        logger.error('File %s not found' % rawfile)
        raise IOError
    if not os.path.isfile(rawfile):
        logger.error('%s is not a file' % rawfile)
        raise IOError

    extension = io.parse_path(rawfile).ext
    assert extension in ['fif', 'fiff'], 'only fif format is supported'
