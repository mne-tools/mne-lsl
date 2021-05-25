#!/usr/bin/env python3

import mne
from pathlib import Path

from neurodecode import logger
import neurodecode.utils.io as io


def dir_resample(fif_dir, recursive, sfreq_target, overwrite=False):
    """
    Change the sampling rate of all raw fif files in a given directory.

    Parameters
    ----------
    fif_dir : str
        The absolute path to the directory containing the .fif files to resample.
    sfreq_target : float
        Tne desired sampling rate in Hz.
    out_dir : str
        The absolute path to the directory where the new files will be saved.
        If None, they will be saved in %fif_dir%/fif_resample%sfreq_target%
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    for fif_file in io.get_file_list(fif_dir, fullpath=True, recursive=recursive):
        fif_file = Path(fif_file)

        if fif_file.suffix == '.fif':
            continue  # skip
        if not fif_file.stem.endswith('-raw'):
            continue

        raw = mne.io.read_raw(fif_file, preload=True)
        raw.resample(sfreq_target)

        out_dir = f'fif_resampled_{sfreq_target}'
        if not (fif_file.parent / out_dir).is_dir():
            io.make_dirs(fif_file.parent / out_dir)

        logger.info(
            f"Exporting to '{fif_file.parent / out_dir / fif_file.name}'")

        try:
            raw.save(fif_file.parent / out_dir / fif_file.name,
                     overwrite=overwrite)
        except FileExistsError:
            logger.warning(
                f'The resampled file already exist for {fif_file.name}. '
                'Use overwrite=True to force overwriting.')
        except:
            raise


if __name__ == '__main__':

    import sys

    if len(sys.argv) > 3:
        raise IOError(
            "Too many arguments provided. Max is 2: fif_dir; sfreq_target")

    if len(sys.argv) == 3:
        fif_dir = sys.argv[1]
        sfreq_target = sys.argv[2]

    if len(sys.argv) == 2:
        fif_dir = sys.argv[1]
        sfreq_target = float(input('Target sampling frequency? \n>> '))

    if len(sys.argv) == 1:
        fif_dir = input('Directory path containing the file files? \n>> ')
        sfreq_target = float(input('Target sampling frequency? \n>> '))

    dir_resample(fif_dir, recursive=False,
                 sfreq_target=sfreq_target, overwrite=False)

    print('Finished.')
