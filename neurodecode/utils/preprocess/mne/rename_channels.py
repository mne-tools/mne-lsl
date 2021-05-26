#!/usr/bin/env python3

import mne
from pathlib import Path

from neurodecode import logger
import neurodecode.utils.io as io


def rename_channels(inst, new_channel_names, copy=False, **kwargs):
    """
    Change the channel names of the MNE instance.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked.
    new_channel_names : list
        The list of the new channel names.
    copy : bool, optional
        If False (default), the MNE instance is modified in-place.
        If True, the MNE instance is copied and returned.
    **kwargs : Additional arguments are passed to mne.rename_channels()
        c.f. https://mne.tools/stable/generated/mne.rename_channels.html

    Returns
    -------
    inst : None | mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        None if copy is set to False, else MNE instance of Raw | Epochs | Evoked
        modified.
    """
    if len(inst.ch_names) != len(new_channel_names):
        raise RuntimeError(
            'The number of new channels does not match that of fif file.')

    mapping = {inst.info['ch_names'][k]: new_ch
               for k, new_ch in enumerate(new_channel_names)}

    if copy:
        copied_inst = inst.copy()
        mne.rename_channels(copied_inst.info, mapping, **kwargs)
        return copied_inst
    else:
        mne.rename_channels(inst.info, mapping, **kwargs)


def dir_rename_channels(fif_dir, recursive, new_channel_names, overwrite=False):
    """
    Change the channel names of all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif' or
    '-raw.fiff'.

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    new_channel_names : list
        The list of the new channel names.
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
        rename_channels(raw, new_channel_names, copy=False)

        if not (fif_file.parent / 'renamed').is_dir():
            io.make_dirs(fif_file.parent / 'renamed')

        logger.info(
            f"Exporting to '{fif_file.parent / 'renamed' / fif_file.name}'")
        try:
            raw.save(fif_file.parent / 'renamed' / fif_file.name,
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
            "Too many arguments provided. Max is 2: fif_dir; new_channel_names.")

    if len(sys.argv) == 3:
        fif_dir = sys.argv[1]
        new_channel_names = sys.argv[2]

    if len(sys.argv) == 2:
        fif_dir = sys.argv[1]
        new_channel_names = eval(input('New channel names (list)? \n>> '))

    if len(sys.argv) == 1:
        fif_dir = input('Directory path containing the file files? \n>> ')
        new_channel_names = eval(input('New channel names (list)? \n>> '))

    dir_rename_channels(fif_dir, recursive=False,
                        new_channel_names=new_channel_names)

    print('Finished.')
