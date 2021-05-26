import mne
from pathlib import Path

from ... import io
from .... import logger


def set_channel_types(raw, mapping):
    """
    Change the channel types of the raw instance.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_channel_types

    Parameters
    ----------
    raw : mne.io.Raw | mne.io.RawArray
        MNE instance of Raw.
    new_channel_names : list
        The list of the new channel names.
    """
    raw.set_channel_types(mapping)


def dir_set_channel_types(fif_dir, recursive, mapping, overwrite=False):
    """
    Change the channel types of all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/dev/generated/mne.io.Raw.html#mne.io.Raw.set_channel_types

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    mapping : dict
        The channel type mapping.
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
        set_channel_types(raw, mapping)

        out_dir = 'corrected'
        if not (fif_file.parent / out_dir).is_dir():
            io.make_dirs(fif_file.parent / out_dir)

        logger.info(
            f"Exporting to '{fif_file.parent / out_dir / fif_file.name}'")
        try:
            raw.save(fif_file.parent / out_dir / fif_file.name,
                     overwrite=overwrite)
        except FileExistsError:
            logger.warning(
                f'The corrected file already exist for {fif_file.name}. '
                'Use overwrite=True to force overwriting.')
        except:
            raise
