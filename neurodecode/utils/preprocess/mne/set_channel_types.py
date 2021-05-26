#!/usr/bin/env python3

import mne
from pathlib import Path

from neurodecode import logger
import neurodecode.utils.io as io


def dir_set_channel_types(fif_dir, recursive, mapping, overwrite=False):
    """
    Change the channel types of all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif' or
    '-raw.fiff'.

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    mapping : dict
        The channel type mapping. c.f.
        https://mne.tools/dev/generated/mne.io.Raw.html#mne.io.Raw.set_channel_types
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

    for fif_file in io.get_file_list(fif_dir, fullpath=True, recursive=recursive):
        fif_file = Path(fif_file)

        if not fif_file.suffix == '.fif':
            continue  # skip
        if not fif_file.stem.endswith('-raw'):
            continue

        raw = mne.io.read_raw(fif_file, preload=True)
        raw.set_channel_types(mapping)

        if not (fif_file.parent / 'corrected').is_dir():
            io.make_dirs(fif_file.parent / 'corrected')

        logger.info(
            f"Exporting to '{fif_file.parent / 'corrected' / fif_file.name}'")
        try:
            raw.save(fif_file.parent / 'corrected' / fif_file.name,
                     overwrite=overwrite)
        except FileExistsError:
            logger.warning(
                f'The corrected file already exist for {fif_file.name}. '
                'Use overwrite=True to force overwriting.')
        except:
            raise


if __name__ == '__main__':

    import sys

    if len(sys.argv) > 3:
        raise IOError(
            "Too many arguments provided. Max is 2: fif_dir; mapping.")

    if len(sys.argv) == 3:
        fif_dir = sys.argv[1]
        mapping = sys.argv[2]

    if len(sys.argv) == 2:
        fif_dir = sys.argv[1]
        mapping = eval(input('New channel types (dict)? \n>> '))

    if len(sys.argv) == 1:
        fif_dir = input('Directory path containing the file files? \n>> ')
        mapping = eval(input('New channel types (dict)? \n>> '))

    dir_set_channel_types(fif_dir, recursive=False,
                          mapping=mapping, overwrite=False)

    print('Finished.')
