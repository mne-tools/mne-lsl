import mne
from pathlib import Path

from .. import io
from ... import logger


def current_source_density(inst, montage=None, **kwargs):
    """
    Apply current source density transformation, also called laplacian filter.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked.
    montage : str | DigMontage
        The montage to used, e.g. 'standard_1020'.
        c.f. https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_montage
    **kwargs : Additional arguments are passed to mne.preprocessing.compute_current_source_density()
        c.f. https://mne.tools/stable/generated/mne.preprocessing.compute_current_source_density.html

    Returns
    -------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE processed instance of Raw | Epochs | Evoked.
    """

    if inst.info['dig'] is None and \
            (kwargs.get('sphere') is None or kwargs.get('sphere') == 'auto'):

        if montage is None:
            logger.error(
                "Current Source Density requires either a sphere (arg sphere) "
                "or digitization (montage) if sphere is set to 'auto' (default).")
            raise ValueError
        else:
            inst.set_montage(montage)

    return mne.preprocessing.compute_current_source_density(inst, **kwargs)


def dir_current_source_density(fif_dir, recursive, montage=None,
                               out_dir=None, overwrite=False, **kwargs):
    """
    Apply a current source density to all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/stable/generated/mne.preprocessing.compute_current_source_density.html

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    montage : str | DigMontage
        The montage to used, e.g. 'standard_1020'.
        c.f. https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_montage
    out_dir : str | None
        The path to the output directory. If None, the directory
        f'fif_dir/csd' is used.
    overwrite : bool
        If true, overwrite previously corrected files.
    **kwargs : Additional arguments are passed to mne.preprocessing.compute_current_source_density()
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = 'csd'
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
        current_source_density(raw, montage, **kwargs)

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
