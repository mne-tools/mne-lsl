"""
BrainVision amplifiers have a default value of -1 for the trigger channel
instead of the conventional 0.
/!\ Tested on actiCHamp amplifiers only, verify the default value for your
amplifier.

Correction is done via:
https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.apply_function
"""

import mne
from pathlib import Path

from .. import find_event_channel, change_event_values_arr
from .... import io
from ..... import logger


def fix_default_event_values(raw, default_value=-1):
    """
    Apply the function change_event_values_arr(timearr, default_value, 0)
    to the raw instance. The raw instance is modified in-place.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.apply_function

    Parameters
    ----------
    raw : mne.io.Raw
        MNE instance of Raw.
    default_value : int, optional
        Trigger channel default value to replace with 0. The default is -1.
    """
    tch = find_event_channel(inst=raw, ch_names=None)
    if tch is None:
        logger.error(
            f'Could not find the trigger channel for {raw.__repr__()}')
        raise RuntimeError

    try:
        raw.apply_function(change_event_values_arr,
                           event_value_old=default_value,
                           event_value_new=0,
                           picks=raw.ch_names[tch], channel_wise=True)
    except RuntimeError:
        logger.warning('MNE raw data should be (pre)loaded. Loading now.')
        raw.load_data()
        raw.apply_function(change_event_values_arr,
                           event_value_old=default_value,
                           event_value_new=0,
                           picks=raw.ch_names[tch], channel_wise=True)
    except:
        raise


def dir_fix_default_event_values(fif_dir, recursive, default_value=-1,
                                 out_dir=None, overwrite=False):
    """
    Replace the values 'default_value' with 0 for the trigger channel.
    The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.apply_function

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    default_value : int
        Trigger channel default value to replace with 0. The default is -1.
    out_dir : str | None
        The path to the output directory. If None, the directory
        'event_fixed' is used.
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
        out_dir = 'event_fixed'
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
        if not fif_file.stem.endswith('-raw'):
            continue

        raw = mne.io.read_raw(fif_file, preload=True)
        fix_default_event_values(raw, default_value=default_value)

        relative = fif_file.relative_to(fif_dir).parent
        if not (fif_dir / out_dir / relative).is_dir():
            io.make_dirs(fif_dir / out_dir / relative)

        logger.info(
            f"Exporting to '{fif_dir / out_dir / relative / fif_file.name}'")
        try:
            raw.save(fif_dir / out_dir / relative / fif_file.name,
                     overwrite=overwrite)
        except FileExistsError:
            logger.warning(
                f'The corrected file already exist for {fif_file.name}. '
                'Use overwrite=True to force overwriting.')
        except:
            raise
