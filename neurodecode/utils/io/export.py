"""
Export fif raw data to different format.

Supported:
    - EEGLAB '.set'
"""
import mne
from pathlib import Path
from scipy.io import savemat
from numpy.core.records import fromarrays

from .io_file_dir import get_file_list, make_dirs
from ... import logger


# ----------------------------- EEG LAB -----------------------------
def write_set(raw, fname):
    """
    Export raw to EEGLAB .set file.

    Source: MNELAB
    https://github.com/cbrnr/mnelab/blob/main/mnelab/io/writers.py

    Parameters
    ----------
    raw : mne.io.Raw | mne.io.RawArray
        MNE instance of Raw.
    fname : str
        Name/Path of the '.set' file created.
    """
    data = raw.get_data() * 1e6  # convert to microvolts
    fs = raw.info["sfreq"]
    times = raw.times
    ch_names = raw.info["ch_names"]
    chanlocs = fromarrays([ch_names], names=["labels"])
    events = fromarrays([raw.annotations.description,
                         raw.annotations.onset * fs + 1,
                         raw.annotations.duration * fs],
                        names=["type", "latency", "duration"])
    savemat(fname,
            dict(EEG=dict(data=data,
                          setname=fname,
                          nbchan=data.shape[0],
                          pnts=data.shape[1],
                          trials=1,
                          srate=fs,
                          xmin=times[0],
                          xmax=times[-1],
                          chanlocs=chanlocs,
                          event=events,
                          icawinv=[],
                          icasphere=[],
                          icaweights=[])),
            appendmat=False)


def dir_write_set(fif_dir, recursive, out_dir=None):
    """
    Converts all raw fif file in a given directory to EEGLAB '.set' format.
    The file name must respect MNE convention and end with '-raw.fif'.

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    out_dir : str | None
        The path to the output directory. If None, the directory
        f'fif_dir/fif2set' is used.
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = 'fif2set'
        if not (fif_dir / out_dir).is_dir():
            make_dirs(fif_dir / out_dir)
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_dir():
            make_dirs(out_dir)

    for fif_file in get_file_list(fif_dir, fullpath=True, recursive=recursive):
        fif_file = Path(fif_file)

        if not fif_file.suffix == '.fif':
            continue
        if not fif_file.stem.endswith('-raw'):
            continue

        raw = mne.io.read_raw(fif_file, preload=True)

        relative = fif_file.relative_to(fif_dir).parent
        if not (fif_dir / out_dir / relative).is_dir():
            make_dirs(fif_dir / out_dir / relative)

        logger.info(
            f"Exporting to '{fif_dir / out_dir / relative / fif_file.stem}.set'")

        write_set(raw, str(fif_dir / out_dir / relative / fif_file.stem) + '.set')
