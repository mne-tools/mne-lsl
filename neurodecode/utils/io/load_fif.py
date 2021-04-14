import os
import mne
import numpy as np

from neurodecode import logger
import neurodecode.utils.q_common as qc
from neurodecode.utils.pycnbi_utils import find_event_channel

#----------------------------------------------------------------------
def load_raw(rawfile, events_ext=None):
    """
    Loads data from a fif-format file.

    Parameters
    ----------
    rawfile : str
        The (absolute) data file path
    events_ext : str
        The txt file containing external events 

    Returns:
    --------
    mne.io.Raw : MNE raw data with the trigger channel at index 0.
    np.array : mne-compatible events numpy array object (N x [frame, 0, type])
    """

    if not os.path.exists(rawfile):
        logger.error('File %s not found' % rawfile)
        raise IOError
    if not os.path.isfile(rawfile):
        logger.error('%s is not a file' % rawfile)
        raise IOError
    
    # Check the fif extension
    extension = qc.parse_path(rawfile).ext
    assert extension in ['fif', 'fiff'], 'only fif format is supported'
    
    # Load mne raw data
    raw = mne.io.Raw(rawfile, preload=True)
    
    # Process events
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

#----------------------------------------------------------------------
def load_multi(src):
    """
    Load multiple data files and concatenate them into a single series
    - Assumes all files have the same sampling rate and channel order.
    - Event locations are updated accordingly with new offset.
    
    See load_raw() for more low-level details.
    
    Parameters
    ----------
    src : str
        The directory or a files list/tuple.
    
    Returns
    -------
    mne.io.Raw : MNE raw data with the trigger channel at index 0.
    np.array : mne-compatible events numpy array object (N x [frame, 0, type])
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
        return load_raw(flist[0])

    # load raw files
    rawlist = []
    for f in flist:
        logger.info('Loading %s' % f)
        raw, _ = load_raw(f)
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
