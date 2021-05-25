import mne
import numpy as np
from pathlib import Path

from neurodecode import logger
import neurodecode.utils.io as io
from neurodecode.utils.events import find_event_channel


def read_raw_fif(fname, events_ext=None, preload=True):
    """
    Load data from a .fif file.

    Parameters
    ----------
    fname : str
        The (absolute) .fif file path.
    events_ext : str
        The txt file containing external events.
    preload : bool
        Preload data into memory for data manipulation and faster indexing.

    Returns:
    --------
    mne.io.Raw : MNE raw data with the trigger channel at index 0.
    np.array : mne-compatible events numpy array object (N x [frame, 0, type])
    """
    fname = Path(fname)

    if not fname.exists():
        logger.error(f"File '{fname}' not found.")
        raise IOError
    if fname.is_file():
        logger.error(f"'{fname}' is not a file.")
        raise IOError

    # Check the fif extension
    assert fname.suffix == '.fif', "Only '.fif' format is supported."

    # Load mne raw data
    raw = mne.io.read_raw_fif(fname, preload=preload)

    # Process events
    if events_ext is not None:
        events = mne.read_events(events_ext)
    else:
        tch = find_event_channel(raw)
        if tch is not None:
            events = mne.find_events(raw, stim_channel=raw.ch_names[tch],
                                     shortest_event=1, uint_cast=True,
                                     consecutive='increasing')
            # MNE's annoying hidden cockroach: first_samp
            # events[:, 0] -= raw.first_samp
        else:
            events = np.array([], dtype=np.int64)

    return raw, events


def read_raw_fif_multi(src):
    """
    Load multiple data .fif files and concatenate them into a single series.
    - Assumes all files have the same sampling rate and channel order.
    - Event locations are updated accordingly with new offset.

    See read_raw_fif() for more low-level details.

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
        src = Path(src)
    if isinstance(src, Path):
        if not src.is_dir():
            logger.error(f"'{src} is not a directory or does not exist.")
            raise IOError
        flist = [file for file in io.get_file_list(
            src) if Path(file).suffix == '.fif']
    elif isinstance(src, (list, tuple)):
        flist = [file for file in src if Path(file).suffix == '.fif']
        if len(flist) != len(src):
            logger.warning("Some of the files are not '.fif'.")
    else:
        logger.error(f'Unknown input type {src}')
        raise TypeError

    if len(flist) == 0:
        logger.error(f'No fif files found in {src}.')
        raise RuntimeError
    elif len(flist) == 1:
        return read_raw_fif(flist[0], preload=True)
    else:
        raws = []
        for fname in flist:
            logger.info(f'Loading {fname}')
            raw, _ = read_raw_fif(fname, preload=False)
            raws.append(raw)

        # create a concatenated raw object and update channel names
        raw_merged = mne.concatenate_raws(raws, preload=True)

        # re-calculate event positions
        events = mne.find_events(
            raw_merged, stim_channel='TRIGGER', shortest_event=1, consecutive=True)

        return raw_merged, events
