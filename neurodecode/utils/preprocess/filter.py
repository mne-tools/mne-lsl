from pathlib import Path

import mne
import numpy as np

from .current_source_density import (current_source_density,
                                     dir_current_source_density)
from .. import io
from ... import logger


def notch_filter(raw, freqs=np.arange(50, 151, 50), picks=None, **kwargs):
    """
    Apply a notch filter to the data of the MNE Raw instance.

    Parameters
    ----------
    raw : mne.io.Raw | mne.io.RawArray
        MNE instance of Raw
    freqs : list | numpy.ndarray
        Specific frequencies to filter out from data.
            np.arange(50, 251, 50) for EU powerline noise.
            np.arange(60, 241, 60) for US powerline noise.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. The default None will pick all data channels.
        Examples:
            picks='eeg'
            picks=['eog', 'ecg']
    **kwargs : Additional arguments are passed to raw.notch_filter()
        c.f. https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter
    """
    if kwargs.get('notch_widths') is None:
        kwargs['notch_widths'] = 3

    if kwargs.get('method') == 'iir' and kwargs.get('iir_params') is None:
        kwargs['iir_params'] = dict(order=4, ftype='butter', output='sos')

    try:
        raw.notch_filter(freqs, picks, **kwargs)
    except RuntimeError:
        logger.warning('MNE raw data should be (pre)loaded. Loading now.')
        raw.load_data()
        raw.notch_filter(freqs, picks, **kwargs)


def dir_notch_filter(fif_dir, recursive, freqs, picks=None,
                     out_dir=None, overwrite=False, **kwargs):
    """
    Apply a notch filter to all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    freqs : list | numpy.ndarray
        Specific frequencies to filter out from data.
            np.arange(50, 251, 50) for EU powerline noise.
            np.arange(60, 241, 60) for US powerline noise.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. The default None will pick all data channels.
        Examples:
            picks='eeg'
            picks=['eog', 'ecg']
    out_dir : str | None
        The path to the output directory. If None, the directory
        f'fif_dir/notched' is used.
    overwrite : bool
        If true, overwrite previously corrected files.
    **kwargs : Additional arguments are passed to raw.notch_filter()
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = 'notched'
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
        notch_filter(raw, freqs, picks, **kwargs)

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


def spectral_filter(inst, l_freq, h_freq, picks=None, **kwargs):
    """
    Apply a filter to the data of the MNE instance.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked.
    l_freq : float | None
        For FIR filters, the lower pass-band edge;.
        For IIR filters, the lower cutoff frequency.
        If None the data are only low-passed.
    h_freq : float | None
        For FIR filters, the upper pass-band edge.
        For IIR filters, the upper cutoff frequency.
        If None the data are only high-passed.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. The default None will pick all data channels.
        Examples:
            picks='eeg'
            picks=['eog', 'ecg']
    **kwargs : Additional arguments are passed to raw.filter()
        c.f. https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter
    """
    if kwargs.get('method') == 'iir' and kwargs.get('iir_params') is None:
        kwargs['iir_params'] = dict(order=4, ftype='butter', output='sos')

    inst.filter(l_freq, h_freq, picks, **kwargs)


def dir_spectral_filter(fif_dir, recursive, l_freq, h_freq, picks=None,
                        out_dir=None, overwrite=False, **kwargs):
    """
    Apply a filter to all raw fif files in a given directory.
    The file name must respect MNE convention and end with '-raw.fif'.

    https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files.
    recursive : bool
        If true, search recursively.
    l_freq : float | None
        For FIR filters, the lower pass-band edge;.
        For IIR filters, the lower cutoff frequency.
        If None the data are only low-passed.
    h_freq : float | None
        For FIR filters, the upper pass-band edge.
        For IIR filters, the upper cutoff frequency.
        If None the data are only high-passed.
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. The default None will pick all data channels.
        Examples:
            picks='eeg'
            picks=['eog', 'ecg']
    out_dir : str | None
        The path to the output directory. If None, the directory
        f'fif_dir/filtered' is used.
    overwrite : bool
        If true, overwrite previously corrected files.
    **kwargs : Additional arguments are passed to raw.filter()
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = 'filtered'
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
        spectral_filter(raw, l_freq, h_freq, picks, **kwargs)

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


def laplacian_filter(inst, montage=None, **kwargs):
    """
    Apply current source density transformation, also called laplacian filter.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked.
    montage : str | DigMontage
        The montage to used, e.g. 'standard_1020'.
        c.f. https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_montage
    **kwargs : Additional arguments are passed to
        mne.preprocessing.compute_current_source_density()
        c.f. https://mne.tools/stable/generated/mne.preprocessing.compute_current_source_density.html

    Returns
    -------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE processed instance of Raw | Epochs | Evoked.
    """
    logger.info(
        'Laplacin filter is called Current Source Density by MNE. '
        'Using neurodecode.utils.preprocess.current_source_density().')
    return current_source_density(inst, montage, *kwargs)


def dir_laplacian_filter(fif_dir, recursive, montage=None,
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
    **kwargs : Additional arguments are passed to
        mne.preprocessing.compute_current_source_density()
    """
    logger.info(
        'Laplacin filter is called Current Source Density by MNE. '
        'Using neurodecode.utils.preprocess.dir_current_source_density().')
    dir_current_source_density(fif_dir, recursive, montage,
                               out_dir, overwrite, **kwargs)
