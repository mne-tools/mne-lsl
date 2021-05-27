import mne
from pathlib import Path

from .. import io
from ... import logger


def resample(inst, sfreq, **kwargs):
    """
    Resample the inst.
    /!\ Resample goal is to speed up computation. As it add a jitter to the
    trigger/events, it is recommanded to first create the epochs and then
    downsample the epochs.

    Parameters
    ----------
    inst : inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked.
    sfreq : float
        Tne desired sampling rate in Hz.
    **kwargs : Additional arguments are passed to inst.resample().
        c.f. https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
    """
    inst.resample(sfreq, **kwargs)


def dir_resample(fif_dir, recursive, sfreq, out_dir=None, overwrite=False):
    """
    Change the sampling rate of all raw and epochs fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif' or
    '-epo.fif'.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    sfreq : float
        Tne desired sampling rate in Hz.
    out_dir : str | None
        The path to the output directory. If None, the directory
        f'fif_resampled_{sfreq}' is used.
    overwrite : bool
        If true, overwrite previously corrected files.
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = f'fif_resampled_{sfreq}'
        if not (fif_dir / out_dir).is_dir():
            io.make_dirs(fif_dir / out_dir)
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_dir():
            io.make_dirs(out_dir)

    for fif_file in io.get_file_list(fif_dir, fullpath=True, recursive=recursive):
        fif_file = Path(fif_file)

        if not fif_file.suffix == '.fif':
            continue
        if not (fif_file.stem.endswith('-raw') or
                fif_file.stem.endswith('-epo')):
            continue

        if fif_file.stem.endswith('-raw'):
            inst = mne.io.read_raw(fif_file, preload=True)
        elif fif_file.stem.endswith('-epo'):
            inst = mne.read_epochs(fif_file, preload=True)
        resample(inst, sfreq)

        relative = fif_file.relative_to(fif_dir).parent
        if not (fif_dir / out_dir / relative).is_dir():
            io.make_dirs(fif_dir / out_dir / relative)

        logger.info(
            f"Exporting to '{fif_dir / out_dir / relative / fif_file.name}'")
        try:
            inst.save(fif_dir / out_dir / relative / fif_file.name,
                      overwrite=overwrite)
        except FileExistsError:
            logger.warning(
                f'The corrected file already exist for {fif_file.name}. '
                'Use overwrite=True to force overwriting.')
        except:
            raise
