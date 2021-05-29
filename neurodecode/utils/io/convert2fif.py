"""
Convert known file format to FIF.
"""

import mne
import pickle
import numpy as np
from pathlib import Path
from mne.io._read_raw import readers

from .io_file_dir import get_file_list, make_dirs
from ..preprocess.events import find_event_channel
from ... import logger


mne.set_log_level('ERROR')

# ------------------------- Stream Recorder PCL -------------------------
def pcl2fif(filename, out_dir=None, external_event=None,
            precision='double', replace=False):
    """
    Convert NeuroDecode Python pickle format to mne.io.raw.

    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format.
    out_dir : str
        Saving directory. If None, it will be the directory of the .pkl file.
    external_event : str
        Event file path in text formatm following mne event struct. Each row should be: index 0 event
    precision : str
        Data matrix format. [single|double|int|short], 'single' improves backward compatability.
    replace : bool
        If true, previous events will be overwritten by the new ones from the external events file.
    """
    filename = Path(filename)
    if not filename.is_file():
        logger.error(f"File '{filename}' not found.")
        raise IOError
    if not filename.suffix == '.pcl':
        logger.error(f"File '{filename}' is not '.pcl'.")
        raise IOError

    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = filename.parent / 'fif'
    make_dirs(out_dir)

    fiffile = out_dir / filename.stem + '.fif'

    # Load from file
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # MNE format
    raw = _format_pcl_to_mne_RawArray(data)

    # Add events from txt file
    if external_event is not None:
        events_index = _event_timestamps_to_indices(
            raw.times, external_event, data["timestamps"][0])
        _add_events_from_txt(
            raw, events_index, stim_channel='TRIGGER', replace=replace)

    # Save
    raw.save(fiffile, verbose=False, overwrite=True, fmt=precision)
    _saveChannels2txt(out_dir, raw.info["ch_names"])
    logger.info(f"Data saved to: '{fiffile}'")


def _format_pcl_to_mne_RawArray(data):
    """
    Format the raw data to the mne rawArray structure.
    Data must be recorded with NeuroDecode StreamRecorder.

    Parameters
    ----------
    data : dict
        Data loaded from the pcl file.

    Returns
    -------
    mne.io.raw
        The mne raw structure.
    """
    if type(data['signals']) == list:
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

    # move trigger channel to index 0
    if trig_ch is None:
        # assuming no event channel exists, add a event channel to index 0 for consistency.
        logger.warning(
            'No event channel was not found. Adding a blank event channel to index 0.')
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
        logger.info(f'Moving event channel{trig_ch} to 0.')
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
        for c in ch_names:
            logger.info(f'{c}')

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
        The whole data's timestamps (mne: start at 0.0 sec).
    eventfile : str
        Event file containing the events, indexed with LSL timestamps.
    offset : float
        The first sample's LSL timestamp, to start at 0.0 sec.

    Returns
    -------
    np.array
        The events [shape=(n_events, 3)]; used as input to raw.add_events().
    """

    ts_min = min(raw_timestamps)
    ts_max = max(raw_timestamps)
    events = []

    with open(eventfile) as f:
        for l in f:
            data = l.strip().split('\t')
            event_ts = float(data[0]) - offset
            event_value = int(data[2])
            next_index = np.searchsorted(raw_timestamps, event_ts)
            if next_index >= len(raw_timestamps):
                logger.warning('Event %d at time %.3f is out of time range (%.3f - %.3f).'
                               % (event_value, event_ts, ts_min, ts_max))
            else:
                events.append([next_index, 0, event_value])

    return events


def _add_events_from_txt(raw, events_index, stim_channel='TRIGGER', replace=False):
    """
    Merge the events extracted from a txt file to the trigger channel.

    Parameters
    ----------
    raw : mne.io.raw
        The mne raw data structure.
    events_index : np.array
        The events [shape=(n_events, 3)]; used as input to raw.add_events().
    stim_channel : str
        The stin channel to add the events.
    replace : bool
        If True the old events on the stim channel are removed before adding
        the new ones.
    """
    if len(events_index) == 0:
        logger.warning('No events were found in the event file.')
    else:
        logger.info(f'Found {len(events_index)} events')
        raw.add_events(events_index, stim_channel=stim_channel,
                       replace=replace)


def _saveChannels2txt(out_dir, ch_names):
    """
    Save the channels list to a txt file for the GUI
    """
    filename = out_dir + "channelsList.txt"
    config = Path(filename)

    if config.is_file() is False:
        file = open(filename, "w")
        for x in range(len(ch_names)):
            file.write(ch_names[x] + "\n")
        file.close()


# ------------------------- General converter -------------------------
# Edit readers with NeuroDecode '.pcl' reader.
readers['.pcl'] = pcl2fif


def any2fif(filename, out_dir=None, overwrite=True, precision='double'):
    """
    Generic file format converter to mne.io.raw.
    Uses mne.io.read_raw():
        https://mne.tools/stable/generated/mne.io.read_raw.html

    Parameters
    ----------
    filename : str
        The pickle file path to convert to fif format.
    out_dir : str
        Saving directory. If None, it will be the directory of the .pkl file.
    overwrite : bool
        If true, overwrite previously converted files with the same name.
    precision : str
        Data matrix format. [single|double|int|short], 'single' improves backward compatability.
    """
    filename = Path(filename)
    if not filename.is_file():
        logger.error(f"File '{filename}' not found.")
        raise IOError

    if filename.suffix == '.pcl':
        eve_file = filename.parent / (filename.stem[:-4] + 'eve.txt')
        if eve_file.exists():
            logger.info(f"Adding events from '{eve_file}'")
        else:
            logger.info(f"No SOFTWARE event file '{eve_file}'")
            eve_file = None

        pcl2fif(filename, out_dir=out_dir, external_event=eve_file,
                precision=precision, replace=False)

    else:
        if out_dir is not None:
            out_dir = Path(out_dir)
        else:
            out_dir = filename.parent / 'fif'
        make_dirs(out_dir)

        fiffile = out_dir / filename.stem + '.fif'

        raw = mne.io.read_raw(filename)
        raw.save(fiffile, verbose=False, overwrite=overwrite, fmt=precision)


def dir_any2fif(directory, recursive, out_dir=None, overwrite=False, **kwargs):
    """
    Converts all the compatible files to raw fif files in a given directory.

    https://mne.tools/stable/generated/mne.io.read_raw.html

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    out_dir : str | None
        The path to the output directory. If None, the directory
        'corrected' is used.
    overwrite : bool
        If true, overwrite previously corrected files.
    **kwargs : Additional arguments are passed to any2fif().
    """
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory '{directory}' not found.")
        raise IOError
    if not directory.is_dir():
        logger.error(f"'{directory}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = 'corrected'
        if not (directory / out_dir).is_dir():
            make_dirs(directory / out_dir)
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_dir():
            make_dirs(out_dir)

    for file in get_file_list(directory, fullpath=True, recursive=recursive):
        file = Path(file)

        if not file.suffix in readers.keys():
            continue

        any2fif(file, out_dir, overwrite, **kwargs)
        logger.info(f"Converted '{file}'.")
