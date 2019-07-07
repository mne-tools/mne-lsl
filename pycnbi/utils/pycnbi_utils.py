from __future__ import print_function, division

"""
PyCNBI utility functions

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
import pylsl
import scipy.io
import numpy as np
import multiprocessing as mp
import xml.etree.ElementTree as ET
import pycnbi.utils.q_common as qc
from scipy.signal import butter, lfilter, lfiltic, buttord
from pycnbi.pycnbi_config import CAP, LAPLACIAN
from pycnbi import logger
from builtins import input

mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper


# note that MNE already has find_events function
def find_events(events_raw):
    """
    Find trigger values, rising from zero to non-zero
    """
    events = []  # triggered event values other than zero

    # set epochs (frame start, frame end)
    ev_last = 0
    for et in range(len(events_raw)):
        ev = events_raw[et]
        if ev != ev_last:
            if ev > 0:
                events.append([et, 0, ev])
            ev_last = ev

    return events


def find_event_channel(raw=None, ch_names=None):
    """
    Find event channel using heuristics for pcl files.

    Disclaimer: Not 100% guaranteed to find.

    Input:
        raw: mne.io.RawArray-like object or numpy array (n_channels x n_samples)
        if raw is None, ch_names must be given.

    Output:
        channel index or None if not found.
    """

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
    elif hasattr(raw, 'ch_names'):
        for ch_name in raw.ch_names:
            if 'TRIGGER' in ch_name or 'STI ' in ch_name or 'TRG' in ch_name or 'CH_Event' in ch_name:
                return raw.ch_names.index(ch_name)
    else:
        if ch_names is None:
            raise ValueError('ch_names cannot be None when raw is None.')
        for ch_name in ch_names:
            if 'TRIGGER' in ch_name or 'STI ' in ch_name or 'TRG' in ch_name or 'CH_Event' in ch_name:
                return ch_names.index(ch_name)
    return None


def raw2mat(infile, outfile):
    '''
    Convert raw data file to MATLAB file
    '''
    raw, events = load_raw(infile)
    header = dict(bads=raw.info['bads'], ch_names=raw.info['ch_names'],\
                  sfreq=raw.info['sfreq'], events=events)
    scipy.io.savemat(outfile, dict(signals=raw._data, header=header))
    logger.info('Exported to %s' % outfile)


def add_events_raw(rawfile, outfile, eventfile, overwrite=True):
    """
    Add events from a file and save

    Note: If the event values already exists in raw file, the new event values
        will be added to the previous value instead of replacing them.
    """

    raw = mne.io.Raw(rawfile, preload=True, proj=False)
    events = mne.read_events(eventfile)
    raw.add_events(events, stim_channel='TRIGGER')
    raw.save(outfile, overwrite=overwrite)


def export_morlet(epochs, filename):
    """
    Export wavelet tranformation decomposition into Matlab format
    """
    freqs = np.array(DWT['freqs'])  # define frequencies of interest
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs,
        n_cycles=n_cycles, use_fft=False, return_itc=True, n_jobs=mp.cpu_count())
    scipy.io.savemat(filename, dict(power=power.data, itc=itc.data, freqs=freqs,
        channels=epochs.ch_names, sfreq=epochs.info['sfreq'], onset=-epochs.tmin))


def event_timestamps_to_indices(sigfile, eventfile):
    """
    Convert LSL timestamps to sample indices for separetely recorded events.

    Parameters:
    sigfile: raw signal file (Python Pickle) recorded with stream_recorder.py.
    eventfile: event file where events are indexed with LSL timestamps.

    Returns:
    events list, which can be used as an input to mne.io.RawArray.add_events().
    """

    raw = qc.load_obj(sigfile)
    ts = raw['timestamps'].reshape(-1)
    ts_min = min(ts)
    ts_max = max(ts)
    events = []

    with open(eventfile) as f:
        for l in f:
            data = l.strip().split('\t')
            event_ts = float(data[0])
            event_value = int(data[2])
            # find the first index not smaller than ts
            next_index = np.searchsorted(ts, event_ts)
            if next_index >= len(ts):
                logger.warning('Event %d at time %.3f is out of time range (%.3f - %.3f).' % (event_value, event_ts, ts_min, ts_max))
            else:
                events.append([next_index, 0, event_value])
    return events


def rereference(raw, ref_new, ref_old=None):
    """
    Reference to new channels. raw object is modified in-place for efficiency.

    raw: mne.io.RawArray

    ref_new: None | list of str (RawArray) | list of int (numpy array)
        Channel(s) to re-reference, e.g. M1, M2.
        Average of these channel values are substracted from all channel values.

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
            mne.io.add_reference_channels(raw, ref_old, copy=False)
        # Re-reference
        mne.io.set_eeg_reference(raw, ref_new, copy=False)

    return True


def preprocess(raw, sfreq=None, spatial=None, spatial_ch=None, spectral=None, spectral_ch=None,
               notch=None, notch_ch=None, multiplier=1, ch_names=None, rereference=None, decim=None, n_jobs=1):
    """
    Apply spatial, spectral, notch filters and convert unit.
    raw is modified in-place.

    Input
    ------
    raw: mne.io.Raw | mne.io.RawArray | mne.Epochs | numpy.array (n_channels x n_samples)
         numpy.array type assumes the data has only pure EEG channnels without event channels

    sfreq: required only if raw is numpy array.

    spatial: None | 'car' | 'laplacian'
        Spatial filter type.

    spatial_ch: None | list (for CAR) | dict (for LAPLACIAN)
        Reference channels for spatial filtering. May contain channel names.
        'car': channel indices used for CAR filtering. If None, use all channels except
               the trigger channel (index 0).
        'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
        *** Note ***
        Since PyCNBI puts trigger channel as index 0, data channel starts from index 1.

    spectral: None | [l_freq, h_freq]
        Spectral filter.
        if l_freq is None: lowpass filter is applied.
        if h_freq is None: highpass filter is applied.
        if l_freq < h_freq: bandpass filter is applied.
        if l_freq > h_freq: band-stop filter is applied.

    spectral_ch: None | list
        Channel picks for spectra filtering. May contain channel names.

    notch: None | float | list of frequency in floats
        Notch filter.

    notch_ch: None | list
        Channel picks for notch filtering. May contain channel names.

    multiplier: float
        If not 1, multiply data values excluding trigger values.

    ch_names: None | list
        If raw is numpy array and channel picks are list of strings, ch_names will
        be used as a look-up table to convert channel picks to channel numbers.

    rereference: Not supported yet.

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

    # Do unit conversion
    if multiplier != 1:
        data[eeg_channels] *= multiplier

    # Apply spatial filter
    if spatial is None:
        pass
    elif spatial == 'car':
        if spatial_ch is None:
            spatial_ch = eeg_channels

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
    elif spatial == 'laplacian':
        if type(spatial_ch) is not dict:
            logger.error('preprocess(): For Lapcacian, spatial_ch must be of form {CHANNEL:[NEIGHBORS], ...}')
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

    # Apply notch filter
    if notch is not None:
        assert False
        if notch_ch is None:
            notch_ch = eeg_channels

        if type(notch_ch[0]) == str:
            assert ch_names is not None, 'preprocess(): ch_names must not be None'
            notch_ch_i = [ch_names.index(c) for c in notch_ch]
        else:
            notch_ch_i = notch_ch

        mne.filter.notch_filter(data, Fs=sfreq, freqs=notch, notch_widths=3,
                                picks=notch_ch_i, method='fft', n_jobs=n_jobs, copy=False)

    if type(raw) == np.ndarray:
        raw = data
    return raw


def load_raw(rawfile, spfilter=None, spchannels=None, events_ext=None, multiplier=1, verbose='ERROR'):
    """
    Loads data from a fif-format file.
    You can convert non-fif files (.eeg, .bdf, .gdf, .pcl) to fif format.

    Parameters:
    rawfile: (absolute) data file path
    spfilter: 'car' | 'laplacian' | None
    spchannels: None | list (for CAR) | dict (for LAPLACIAN)
        'car': channel indices used for CAR filtering. If None, use all channels except
               the trigger channel (index 0).
        'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
        *** Note ***
        Since PyCNBI puts trigger channel as index 0, data channel starts from index 1.
    events_ext: Add externally recorded events.
                [ [sample_index1, 0, event_value1],... ]
    multiplier: Multiply all values except triggers (to convert unit).

    Returns:
    raw: mne.io.RawArray object. First channel (index 0) is always trigger channel.
    events: mne-compatible events numpy array object (N x [frame, 0, type])
    spfilter= {None | 'car' | 'laplacian'}

    """

    if not os.path.exists(rawfile):
        logger.error('File %s not found' % rawfile)
        raise IOError
    if not os.path.isfile(rawfile):
        logger.error('%s is not a file' % rawfile)
        raise IOError

    extension = qc.parse_path(rawfile).ext
    assert extension in ['fif', 'fiff'], 'only fif format is supported'
    raw = mne.io.Raw(rawfile, preload=True, verbose=verbose)
    if spfilter is not None or multiplier is not 1:
        preprocess(raw, spatial=spfilter, spatial_ch=spchannels, multiplier=multiplier)
    if events_ext is not None:
        events = mne.read_events(events_ext)
    else:
        tch = find_event_channel(raw)
        if tch is not None:
            events = mne.find_events(raw, stim_channel=raw.ch_names[tch], shortest_event=1, uint_cast=True, consecutive='increasing')
            # MNE's annoying hidden cockroach: first_samp
            events[:, 0] -= raw.first_samp
        else:
            events = np.array([], dtype=np.int64)

    return raw, events


def load_multi(src, spfilter=None, spchannels=None, multiplier=1):
    """
    Load multiple data files and concatenate them into a single series

    - Assumes all files have the same sampling rate and channel order.
    - Event locations are updated accordingly with new offset.

    @params:
        src: directory or list of files.
        spfilter: apply spatial filter while loading.
        spchannels: list of channel names to apply spatial filter.
        multiplier: to change units for better numerical stability.

    See load_raw() for more low-level details.

    """

    if type(src) == str:
        if not os.path.isdir(src):
            logger.error('%s is not a directory or does not exist.' % src)
            raise IOError
        flist = []
        for f in qc.get_file_list(src):
            if qc.parse_path_list(f)[2] == 'fif':
                flist.append(f)
    elif type(src) in [list, tuple]:
        flist = src
    else:
        logger.error('Unknown input type %s' % type(src))
        raise TypeError

    if len(flist) == 0:
        logger.error('load_multi(): No fif files found in %s.' % src)
        raise RuntimeError
    elif len(flist) == 1:
        return load_raw(flist[0], spfilter=spfilter, spchannels=spchannels, multiplier=multiplier)

    # load raw files
    rawlist = []
    for f in flist:
        logger.info('Loading %s' % f)
        raw, _ = load_raw(f, spfilter=spfilter, spchannels=spchannels, multiplier=multiplier)
        rawlist.append(raw)

    # concatenate signals
    signals = None
    for raw in rawlist:
        if signals is None:
            signals = raw._data
        else:
            signals = np.concatenate((signals, raw._data), axis=1) # append samples

    # create a concatenated raw object and update channel names
    raw = rawlist[0]
    trigch = find_event_channel(raw)
    ch_types = ['eeg'] * len(raw.ch_names)
    if trigch is not None:
        ch_types[trigch] = 'stim'
    info = mne.create_info(raw.ch_names, raw.info['sfreq'], ch_types)
    raw_merged = mne.io.RawArray(signals, info)

    # re-calculate event positions
    events = mne.find_events(raw_merged, stim_channel='TRIGGER', shortest_event=1, consecutive=True)

    return raw_merged, events


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


def search_lsl(ignore_markers=False):
    import time

    # look for LSL servers
    amp_list = []
    amp_list_backup = []
    while True:
        streamInfos = pylsl.resolve_streams()
        if len(streamInfos) > 0:
            for index, si in enumerate(streamInfos):
                # LSL XML parser has a bug which crashes so do not use for now
                #desc = pylsl.StreamInlet(si).info().desc()
                #amp_serial = desc.child('acquisition').child_value('serial_number').strip()
                amp_serial = 'N/A' # serial number not supported yet
                amp_name = si.name()
                if 'Markers' in amp_name:
                    amp_list_backup.append((index, amp_name, amp_serial))
                else:
                    amp_list.append((index, amp_name, amp_serial))
            break
        logger.info('No server available yet on the network...')
        time.sleep(1)

    if ignore_markers is False:
        amp_list += amp_list_backup

    logger.info('-- List of servers --')
    for i, (index, amp_name, amp_serial) in enumerate(amp_list):
        if amp_serial == '':
            amp_ser = 'N/A'
        else:
            amp_ser = amp_serial
        logger.info('%d: %s (Serial %s)' % (i, amp_name, amp_ser))

    if len(amp_list) == 1:
        index = 0
    else:
        index = input('Amp index? Hit enter without index to select the first server.\n>> ')
        if index.strip() == '':
            index = 0
        else:
            index = int(index.strip())
    amp_index, amp_name, amp_serial = amp_list[index]
    si = streamInfos[amp_index]
    assert amp_name == si.name()
    # LSL XML parser has a bug which crashes so do not use for now
    #assert amp_serial == pylsl.StreamInlet(si).info().desc().child('acquisition').child_value('serial_number').strip()
    logger.info('Selected %s (Serial: %s)' % (amp_name, amp_serial))

    return amp_name, amp_serial


def lsl_channel_list(inlet):
    """
    Reads XML description of LSL header and returns channel list

    Input:
        pylsl.StreamInlet object
    Returns:
        ch_list: [ name1, name2, ... ]
    """
    if not type(inlet) is pylsl.StreamInlet:
        logger.error('lsl_channel_list(): wrong input type %s' % type(inlet))
        raise TypeError
    root = ET.fromstring(inlet.info().as_xml())
    desc = root.find('desc')
    ch_list = []
    for ch in desc.find('channels').getchildren():
        ch_name = ch.find('label').text
        ch_list.append(ch_name)

    ''' This code may throw access violation error due to bug in pylsl.XMLElement
    # for some reason type(inlet) returns 'instance' type in Python 2.
    ch = inlet.info().desc().child('channels').first_child()
    ch_list = []
    for k in range(inlet.info().channel_count()):
        ch_name = ch.child_value('label')
        ch_list.append(ch_name)
        ch = ch.next_sibling()
    '''
    return ch_list


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
   
