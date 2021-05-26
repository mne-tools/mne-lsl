import mne
from pathlib import Path

from ... import io
from .... import logger


def rename_channels(inst, new_channel_names, **kwargs):
    """
    Change the channel names of the MNE instance.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked.
    new_channel_names : list
        The list of the new channel names.
    **kwargs : Additional arguments are passed to mne.rename_channels()
        c.f. https://mne.tools/stable/generated/mne.rename_channels.html
    """
    if len(inst.ch_names) != len(new_channel_names):
        logger.error(
            'The number of new channels does not match that of fif file.')
        raise RuntimeError

    mapping = {inst.info['ch_names'][k]: new_ch
               for k, new_ch in enumerate(new_channel_names)}
    mne.rename_channels(inst.info, mapping, **kwargs)


def dir_rename_channels(fif_dir, recursive, new_channel_names,
                        out_dir=None, overwrite=False):
    """
    Change the channel names of all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/stable/generated/mne.rename_channels.html

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    new_channel_names : list
        The list of the new channel names.
    out_dir : str | None
        The path to the output directory. If None, the directory
        f'fif_dir/renamed' is used.
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
        out_dir = 'renamed'
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
        rename_channels(raw, new_channel_names)

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
