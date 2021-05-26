import mne
import numpy as np
from pathlib import Path

from .find_event_channel import find_event_channel
from .. import io
from ... import logger


def change_event_values_arr(timearr, event_value_old, event_value_new):
    """
    Replace the values 'event_value_old' with 'event_value_new'
    for the channel timearr.

    Parameters
    ----------
    timearr : numpy.ndarray (n_samples, )
        Channel sample array.
    event_value_old : int
        The old event value.
    event_value_new : int
        The new event value.

    Returns
    -------
    timearr : numpy.ndarray (n_samples, )
        Modified channel sample array.
    """
    timearr[np.where(timearr == event_value_old)] = event_value_new
    return timearr


def change_event_values(raw, event_value_old, event_value_new):
    """
    Apply the function change_event_values_arr to the raw instance.
    The raw instance is modified in-place.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.apply_function

    Parameters
    ----------
    raw : mne.io.Raw
        MNE instance of Raw.
    event_value_old : int
        The old event value.
    event_value_new : int
        The new event value.
    """
    tch = find_event_channel(inst=raw, ch_names=None)
    if tch is None:
        logger.error(
            f'Could not find the trigger channel for {raw.__repr__()}')
        raise RuntimeError

    try:
        raw.apply_function(change_event_values_arr,
                           event_value_old=event_value_old,
                           event_value_new=event_value_new,
                           picks=raw.ch_names[tch], channel_wise=True)
    except RuntimeError:
        raw.load_data()
        raw.apply_function(change_event_values_arr,
                           event_value_old=event_value_old,
                           event_value_new=event_value_new,
                           picks=raw.ch_names[tch], channel_wise=True)
    except:
        raise


def dir_change_event_values(fif_dir, recursive, event_value_old,
                            event_value_new, out_dir=None, overwrite=False):
    """
    Replace the value 'event_value_old' with 'event_value_new' for the trigger
    channel. The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.apply_function

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    event_value_old : int
        The old event value.
    event_value_new : int
        The new event value.
    out_dir : str | None
        The path to the output directory. If None, the directory
        f'fif_dir/event_{event_value_old}_changed_to_{event_value_new}'
        is used.
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
        out_dir = f'event_{event_value_old}_changed_to_{event_value_new}'
        if not (fif_dir / out_dir).is_dir():
            io.make_dirs(fif_dir / out_dir)
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_dir():
            io.make_dirs(out_dir)

    for fif_file in io.get_file_list(fif_dir, fullpath=True, recursive=recursive):
        fif_file = Path(fif_file)

        if not fif_file.suffix == '.fif':
            continue  # skip
        if not fif_file.stem.endswith('-raw'):
            continue

        raw = mne.io.read_raw(fif_file, preload=True)
        change_event_values(raw, event_value_old, event_value_new)

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
