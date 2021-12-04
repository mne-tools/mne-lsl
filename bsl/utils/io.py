"""
Convert known file format to FIF.
"""
import os
import pickle
from pathlib import Path

import mne
import numpy as np
from mne.io._read_raw import supported

from . import find_event_channel
from ._logs import logger


mne.set_log_level('ERROR')


# ------------------------- Stream Recorder PCL -------------------------
def pcl2fif(fname, out_dir=None, external_event=None, external_annotation=None,
            precision='double', replace=False, overwrite=True):
    """
    Convert BSL Python pickle format to MNE `~mne.io.Raw`.

    Parameters
    ----------
    fname : `str` | `~pathlib.Path`
        Pickle file path to convert to ``.fif`` format.
    out_dir : `str` | `~pathlib.Path`
        Saving directory. If `None`, it will be the directory
        ``fname.parent/'fif'``.
    external_event : `str`
        Event file path in text format, following MNE event structure.
        Each row should be: ``index 0 event``
    external_annotation : `str`
        Annotation file path in json format.
    precision : `str`
        Data matrix format. ``[single|double|int|short]``, ``'single'``
        improves backward compatability.
    replace : `bool`
        If `True`, previous events will be overwritten by the new ones from
        the external events file.
    overwrite : `bool`
        If ``True``, overwrite the previous file.
    """
    fname = Path(fname)
    if not fname.is_file():
        raise IOError('File %s not found.' % fname)
    if fname.suffix != '.pcl':
        raise IOError("File type %s is not '.pcl'." % fname.suffix)

    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = fname.parent / 'fif'
    os.makedirs(out_dir, exist_ok=True)

    fiffile = out_dir / str(fname.stem + '.fif')

    # Load from file
    with open(fname, 'rb') as file:
        data = pickle.load(file)

    # MNE format
    raw = _format_pcl_to_mne_RawArray(data)

    # Add events from txt file
    if external_event is not None:
        events_index = _event_timestamps_to_indices(
            raw.times, external_event, data["timestamps"][0])
        _add_events_from_txt(
            raw, events_index, stim_channel='TRIGGER', replace=replace)

    # Add annotation from json file
    if external_annotation is not None:
        pass

    # Save
    raw.save(fiffile, verbose=False, overwrite=overwrite, fmt=precision)
    logger.info("Data saved to: '%s'", fiffile)


def _format_pcl_to_mne_RawArray(data):
    """
    Format the raw data to the MNE RawArray structure.
    Data must be recorded with BSL StreamRecorder.

    Parameters
    ----------
    data : dict
        Data loaded from the .pcl file.

    Returns
    -------
    raw : Raw
        MNE raw structure.
    """
    if isinstance(data['signals'], list):
        signals_raw = np.array(data['signals'][0]).T    # to channels x samples
    else:
        signals_raw = data['signals'].T                 # to channels x samples

    sample_rate = data['sample_rate']

    # Look for channels name
    if 'ch_names' not in data:
        ch_names = [f'CH{x+1}' for x in range(signals_raw.shape[0])]
    else:
        ch_names = data['ch_names']

    # search for the trigger channel
    trig_ch = find_event_channel(signals_raw, ch_names)
    # TODO: patch to be improved for multi-trig channel recording
    if isinstance(trig_ch, list):
        trig_ch = trig_ch[0]

    # move trigger channel to index 0
    if trig_ch is None:
        # Add a event channel to index 0 for consistency.
        logger.warning(
            'Event channel was not found. '
            'Adding a blank event channel to index 0.')
        eventch = np.zeros([1, signals_raw.shape[1]])
        signals = np.concatenate((eventch, signals_raw), axis=0)
        # data['channels'] is not reliable any more
        num_eeg_channels = signals_raw.shape[0]
        trig_ch = 0
        ch_names = ['TRIGGER'] + ch_names

    elif trig_ch == 0:
        signals = signals_raw
        num_eeg_channels = data['channels'] - 1

    else:
        logger.info('Moving event channel %s to 0.', trig_ch)
        signals = np.concatenate((signals_raw[[trig_ch]],
                                  signals_raw[:trig_ch],
                                  signals_raw[trig_ch + 1:]),
                                 axis=0)
        assert signals_raw.shape == signals.shape
        num_eeg_channels = data['channels'] - 1
        ch_names.pop(trig_ch)
        trig_ch = 0
        ch_names.insert(trig_ch, 'TRIGGER')
        logger.info('New channel list:')
        for channel in ch_names:
            logger.info('%s', channel)

    ch_info = ['stim'] + ['eeg'] * num_eeg_channels
    info = mne.create_info(ch_names, sample_rate, ch_info)

    # create Raw object
    raw = mne.io.RawArray(signals, info)

    return raw


def _event_timestamps_to_indices(raw_timestamps, eventfile, offset):
    """
    Convert LSL timestamps to sample indices for separetely recorded events.

    Parameters
    ----------
    raw_timestamps : list
        Data's timestamps (MNE: start at 0.0 sec).
    eventfile : str
        Event file containing the events, indexed with LSL timestamps.
    offset : float
        LSL timestamp of the first sample, to start at 0.0 sec.

    Returns
    -------
    events : np.array
        MNE-compatible events [shape=(n_events, 3)]
        Used as input to mne.io.Raw.add_events.
    """

    ts_min = min(raw_timestamps)
    ts_max = max(raw_timestamps)
    events = []

    with open(eventfile) as file:
        for line in file:
            data = line.strip().split('\t')
            event_ts = float(data[0]) - offset
            event_value = int(data[2])
            next_index = np.searchsorted(raw_timestamps, event_ts)
            if next_index >= len(raw_timestamps):
                logger.warning(
                    'Event %d at time %.3f is out of time range (%.3f - %.3f).'
                    % (event_value, event_ts, ts_min, ts_max))
            else:
                events.append([next_index, 0, event_value])

    return events


def _add_events_from_txt(raw, events_index, stim_channel='TRIGGER',
                         replace=False):
    """
    Merge the events extracted from a .txt file to the trigger channel.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw instance.
    events_index : np.array
        MNE-compatible events [shape=(n_events, 3)].
        Used as input to raw.add_events.
    stim_channel : str
        Stim channel where the events are added.
    replace : bool
        If True, the old events on the stim channel are removed before
        adding the new ones.
    """
    if len(events_index) == 0:
        logger.warning('No events were found in the event file.')
    else:
        logger.info('Found %i events', len(events_index))
        raw.add_events(events_index, stim_channel=stim_channel,
                       replace=replace)


# ------------------------- General converter -------------------------
# Edit readers with BSL '.pcl' reader.
supported['.pcl'] = pcl2fif


def any2fif(fname, out_dir=None, overwrite=True, precision='double'):
    """
    Generic file format converter to `mne.io.Raw`.
    Uses `mne.io.read_raw`.

    Parameters
    ----------
    fname : `str` | `~pathlib.Path`
        File path to convert to ``.fif`` format.
    out_dir : `str` | `~pathlib.Path`
        Saving directory. If `None`, it will be the directory
        ``fname.parent/'fif'.``
    overwrite : `bool`
        If ``True``, overwrite previously converted files with the same name.
    precision : `str`
        Data matrix format. ``[single|double|int|short]``, ``'single'``
        improves backward compatability.
    """
    fname = Path(fname)
    if not fname.is_file():
        raise IOError('File %s not found.' % fname)
    if fname.suffix not in supported:
        raise IOError('File type %s is not supported.' % fname.suffix)

    if fname.suffix == '.pcl':
        eve_file = fname.parent / (fname.stem[:-4] + 'eve.txt')
        if eve_file.exists():
            logger.info("Adding events from '%s'", eve_file)
        else:
            logger.info("No SOFTWARE event file '%s'", eve_file)
            eve_file = None

        pcl2fif(fname, out_dir=out_dir, external_event=eve_file,
                precision=precision, replace=False, overwrite=overwrite)

    else:
        if out_dir is not None:
            out_dir = Path(out_dir)
        else:
            out_dir = fname.parent / 'fif'
        os.makedirs(out_dir, exist_ok=True)

        fiffile = out_dir / fname.stem + '.fif'

        raw = mne.io.read_raw(fname)
        raw.save(fiffile, verbose=False, overwrite=overwrite, fmt=precision)
