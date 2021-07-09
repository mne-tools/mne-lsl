from pathlib import Path

import mne

from .. import io
from ... import logger


def set_eeg_reference(inst, ref_channels, ref_old=None, **kwargs):
    """
    Reference to new channels. MNE raw object is modified in-place for
    efficiency.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked. Assumes the 'eeg' type is
        correctly set for EEG channels.
    ref_channels : list of str | str
        Can be:
        - The name(s) of the channel(s) used to construct the reference.
        - 'average' to apply an average reference (CAR)
        - 'REST' to use the reference electrode standardization technique
        infinity reference (requires instance with montage forward kwarg).
    ref_old : list of str | str
        Channel(s) to recover.
    **kwargs : Additional arguments are passed to mne.set_eeg_reference()
        c.f. https://mne.tools/dev/generated/mne.set_eeg_reference.html
    """
    if not (all(isinstance(ref_ch, str) for ref_ch in ref_channels)
            or isinstance(ref_channels, str)):
        logger.error(
            "The new reference channel must be a list of strings "
            "or 'average' or 'REST'.")
        raise ValueError
    if ref_old is not None:
        mne.add_reference_channels(inst, ref_old, copy=False)

    mne.set_eeg_reference(inst, ref_channels, copy=False, **kwargs)


def dir_set_eeg_reference(fif_dir, recursive, ref_channels, ref_old=None,
                          out_dir=None, overwrite=False, **kwargs):
    """
    Change the eeg reference of all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/dev/generated/mne.set_eeg_reference.html

    Parameters
    ----------
    fif_dir : str
        Path to the directory containing fif files.
    recursive : bool
        If True, search recursively.
    ref_channels : list of str | str
        Can be:
        - The name(s) of the channel(s) used to construct the reference.
        - 'average' to apply an average reference (CAR)
        - 'REST' to use the reference electrode standardization technique
        infinity reference (requires raw with montage and forward model/kwarg).
    ref_old : list of str | str
        Channel(s) to recover.
    out_dir : str | None
        Path to the output directory. If None, the directory
        'fif_dir/rereferenced' is used.
    overwrite : bool
        If True, overwrite previously corrected files.
    **kwargs : Additional arguments are passed to mne.set_eeg_reference()
        c.f. https://mne.tools/dev/generated/mne.set_eeg_reference.html
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = 'rereferenced'
        if not (fif_dir / out_dir).is_dir():
            io.make_dirs(fif_dir / out_dir)
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_dir():
            io.make_dirs(out_dir)

    for fif_file in io.get_file_list(fif_dir, fullpath=True,
                                     recursive=recursive):
        fif_file = Path(fif_file)

        if fif_file.suffix != '.fif':
            continue
        if not fif_file.stem.endswith('-raw'):
            continue

        raw = mne.io.read_raw(fif_file, preload=True)
        set_eeg_reference(raw, ref_channels, ref_old, **kwargs)

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
