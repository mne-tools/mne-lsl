from pathlib import Path

import mne
import numpy as np

from .. import find_event_channel


def read_raw_fif(fname, events_ext=None, preload=True):
    """
    Load data from a ``.fif`` file.

    Parameters
    ----------
    fname : `str` | `~pathlib.Path`
        Path to the ``.fif`` file.
    events_ext : `str` | `~pathlib.Path`
        Path to the ``.txt`` file containing external events.
    preload : `bool`
        Preload data into memory for data manipulation and faster indexing.

    Returns:
    --------
    raw : `~mne.io.Raw`
        MNE `~mne.io.Raw` instance.
    events : `~numpy.array`
        MNE-compatible events numpy array object (N x [frame, 0, type])
    """
    fname = Path(fname)

    # Load mne raw data
    raw = mne.io.read_raw_fif(fname, preload=preload)

    # Process events
    if events_ext is not None:
        events = mne.read_events(events_ext)
    else:
        tch = find_event_channel(raw)
        if tch is not None:
            events = mne.find_events(
                raw, stim_channel=raw.ch_names[tch], shortest_event=1,
                uint_cast=True, consecutive='increasing')
        else:
            events = np.array([], dtype=np.int64)

    return raw, events
