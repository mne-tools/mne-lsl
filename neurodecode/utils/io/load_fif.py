import os
import mne
import numpy as np

from neurodecode import logger
import neurodecode.utils.preprocess as preprocess

import neurodecode.utils.io as io 

#----------------------------------------------------------------------
def load_fif_raw(rawfile, events_ext=None, preload=True):
    """
    Load data from a .fif file.

    Parameters
    ----------
    rawfile : str
        The (absolute) .fif file path
    events_ext : str
        The txt file containing external events
    preload : bool
        Preload data into memory for data manipulation and faster indexing.

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
    extension = io.parse_path(rawfile).ext
    assert extension in ['fif', 'fiff'], 'only fif format is supported'
    
    # Load mne raw data
    raw = mne.io.read_raw_fif(rawfile, preload=preload)
    
    # Process events
    if events_ext is not None:
        events = mne.read_events(events_ext)
    else:
        tch = preprocess.find_event_channel(raw)
        if tch is not None:
            events = mne.find_events(raw, stim_channel=raw.ch_names[tch], shortest_event=1, uint_cast=True, consecutive='increasing')
            # MNE's annoying hidden cockroach: first_samp
            # events[:, 0] -= raw.first_samp
        else:
            events = np.array([], dtype=np.int64)

    return raw, events

#----------------------------------------------------------------------
def load_fif_multi(src):
    """
    Load multiple data .fif files and concatenate them into a single series
    - Assumes all files have the same sampling rate and channel order.
    - Event locations are updated accordingly with new offset.
    
    See load_fif_raw() for more low-level details.
    
    Parameters
    ----------
    src : str
        The directory or a files list/tuple.
    
    Returns
    -------
    mne.io.Raw : MNE raw data with the trigger channel at index 0.
    np.array : mne-compatible events numpy array object (N x [frame, 0, type])
    """

    if isinstance(src, str):
        if not os.path.isdir(src):
            logger.error('%s is not a directory or does not exist.' % src)
            raise IOError
        flist = [f for f in io.get_file_list(src) if io.parse_path(f).ext in ['fif', 'fiff']]
    elif isinstance(src, (list, tuple)):
        flist = [f for f in src if io.parse_path(f).ext in ['fif', 'fiff']]
    else:
        logger.error('Unknown input type %s' % type(src))
        raise TypeError

    if len(flist) == 0:
        logger.error('load_multi(): No fif files found in %s.' % src)
        raise RuntimeError
    elif len(flist) == 1:
        return load_fif_raw(flist[0], preload=True)

    # load raw files
    raws = []
    for f in flist:
        logger.info('Loading %s' % f)
        raw, _ = load_fif_raw(f, preload=False)
        raws.append(raw)
    
    # create a concatenated raw object and update channel names
    raw_merged = mne.concatenate_raws(raws, preload=True)
    
    # re-calculate event positions
    events = mne.find_events(raw_merged, stim_channel='TRIGGER', shortest_event=1, consecutive=True)
    
    return raw_merged, events
