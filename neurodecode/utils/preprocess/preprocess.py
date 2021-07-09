from pathlib import Path

import mne
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne.evoked import Evoked

from .current_source_density import current_source_density
from .filter import spectral_filter, notch_filter, laplacian_filter
from .rename_channels import rename_channels
from .resample import resample
from .set_channel_types import set_channel_types
from .set_eeg_reference import set_eeg_reference
from .events import change_event_values
from .events.brainvision import fix_default_event_values
from .set_montage import set_montage

from .. import io
from ... import logger

# -------------- List of available functions --------------
FUNCTIONS_IN_PLACE = {
    'notch_filter': notch_filter,
    'spectral_filter': spectral_filter,
    'rename_channels': rename_channels,
    'resample': resample,
    'set_channel_types': set_channel_types,
    'set_eeg_reference': set_eeg_reference,
    'change_event_values': change_event_values,
    'brainvision.fix_default_event_values': fix_default_event_values,
    'set_montage': set_montage
}

FUNCTIONS_OUT_OF_PLACE = {
    'current_source_density': current_source_density,
    'laplacian_filter': laplacian_filter
}

# ---------------- List of function's input ---------------
SUPPORTED = (BaseRaw, BaseEpochs, Evoked)
INPUTS = {
    'notch_filter': (BaseRaw),
    'spectral_filter': (BaseRaw, BaseEpochs, Evoked),
    'laplacian_filter': (BaseRaw, BaseEpochs, Evoked),
    'current_source_density': (BaseRaw, BaseEpochs, Evoked),
    'rename_channels': (BaseRaw, BaseEpochs, Evoked),
    'resample': (BaseRaw, BaseEpochs, Evoked),
    'set_channel_types': (BaseRaw),
    'set_eeg_reference': (BaseRaw),
    'change_event_values': (BaseRaw),
    'brainvision.fix_default_event_values': (BaseRaw),
    'set_montage': (BaseRaw, BaseEpochs, Evoked)
}

# ----------- List of supported files extension -----------
EXT = {
    BaseRaw: ['-raw.fif'],
    BaseEpochs: ['-epo.fif'],
    Evoked: ['-ave.fif']
}
# ---------------------------------------------------------


def available_transformation(verbose=False):
    """
    Function returning the list of available transformation.

    Parameters
    ----------
    verbose : bool
        If True, prints the available transformations.
    """
    if verbose:
        print('Available transformations are:')
        for key in INPUTS:
            print(f'  | {key}')

    return list(INPUTS.keys())


def preprocess(inst, transformations):
    """
    Apply the transformations to the MNE instance.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked.
    transformations : dict
        Dict containing the transformations and arguments to pass to each
        transformation.
            key: str
                The name of the transformation function.
            value: list | dict
                The list of arguments (*args) to pass to the transformation
                function or the dictionnary of kwargs (*kwargs) to pass to the
                transformation function.
        Example:
            {
                'notch_filter': {'freqs': np.arange(50, 151, 50)},
                'spectral_filter': [1, 40, 'eeg'],
                'set_eeg_reference': ['average', 'CPz']
            }
    """
    if not isinstance(inst, SUPPORTED):
        logger.error('Preprocess supports Raw, Epochs and Evoked instances.')
        raise TypeError

    for function_name, args in transformations.items():
        if function_name in FUNCTIONS_IN_PLACE.keys():

            if not isinstance(inst, INPUTS[function_name]):
                logger.warning(
                    f"Preprocessing function '{function_name}' supports "
                    f"{INPUTS[function_name]}.\n"
                    "Skipping.")
                continue
            if isinstance(args, list):
                FUNCTIONS_IN_PLACE[function_name](inst, *args)
            elif isinstance(args, dict):
                FUNCTIONS_IN_PLACE[function_name](inst, **args)
            else:
                logger.error('The argument provided must be a list or a dict.')
                raise TypeError

            logger.info(f"Preprocessing function '{function_name}' applied.")

        elif function_name in FUNCTIONS_OUT_OF_PLACE.keys():
            if not isinstance(inst, INPUTS[function_name]):
                logger.warning(
                    f"Preprocessing function '{function_name}' supports "
                    f"{INPUTS[function_name]}.\n"
                    "Skipping.")
                continue
            if isinstance(args, list):
                inst = FUNCTIONS_OUT_OF_PLACE[function_name](inst, *args)
            elif isinstance(args, dict):
                inst = FUNCTIONS_OUT_OF_PLACE[function_name](inst, **args)
            else:
                logger.error('The argument provided must be a list or a dict.')
                raise TypeError

            logger.info(f"Preprocessing function '{function_name}' applied.")

        else:
            logger.warning(
                f"Preprocessing function '{function_name}' not recognized. "
                "Skipping.")
            continue

    return inst


def dir_preprocess(fif_dir, recursive, transformations,
                   out_dir=None, overwrite=False):
    """
    Apply the transformation to all fif files in a given directory.

    Parameters
    ----------
    fif_dir : str
        Path to the directory containing fif files.
    recursive : bool
        If True, search recursively.
    transformations : dict
        Dict containing the transformations and arguments to pass to each
        transformation.
            key: str
                The name of the transformation function.
            value: list | dict
                The list of arguments (*args) to pass to the transformation
                function or the dictionnary of kwargs (*kwargs) to pass to the
                transformation function.
        Example:
            {
                'notch_filter': {'freqs': np.arange(50, 151, 50)},
                'spectral_filter': [1, 40, 'eeg'],
                'set_eeg_reference': ['average', 'CPz']
            }
    out_dir : str | None
        Path to the output directory. If None, the directory
        'fif_dir/preprocessed' is used.
    overwrite : bool
        If True, overwrite previously corrected files.
    """
    fif_dir = Path(fif_dir)
    if not fif_dir.exists():
        logger.error(f"Directory '{fif_dir}' not found.")
        raise IOError
    if not fif_dir.is_dir():
        logger.error(f"'{fif_dir}' is not a directory.")
        raise IOError

    if out_dir is None:
        out_dir = 'preprocessed'
        if not (fif_dir / out_dir).is_dir():
            io.make_dirs(fif_dir / out_dir)
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_dir():
            io.make_dirs(out_dir)

    for fif_file in io.get_file_list(fif_dir, fullpath=True,
                                     recursive=recursive):
        fif_file = Path(fif_file)

        if any(fif_file.name.endswith(ending) for ending in EXT[BaseRaw]):
            inst = mne.io.read_raw(fif_file, preload=True)
        elif any(fif_file.name.endswith(ending) for ending in EXT[BaseEpochs]):
            inst = mne.read_epochs(fif_file, preload=True)
        elif any(fif_file.name.endswith(ending) for ending in EXT[Evoked]):
            inst = mne.read_evokeds(fif_file)
        else:
            continue

        preprocess(inst, transformations)

        relative = fif_file.relative_to(fif_dir).parent
        if not (fif_dir / out_dir / relative).is_dir():
            io.make_dirs(fif_dir / out_dir / relative)

        logger.info(
            f"Exporting to '{fif_dir / out_dir / relative / fif_file.name}'")
        if any(fif_file.name.endswith(ending) for ending in EXT[BaseRaw]) or\
           any(fif_file.name.endswith(ending) for ending in EXT[BaseEpochs]):
            try:
                inst.save(fif_dir / out_dir / relative / fif_file.name,
                          overwrite=overwrite)
            except FileExistsError:
                logger.warning(
                    'The preprocessed file already exist for '
                    f'{fif_file.name}. '
                    'Use overwrite=True to force overwriting.')

        elif any(fif_file.name.endswith(ending) for ending in EXT[Evoked]):
            try:
                inst.save(fif_dir / out_dir / relative / fif_file.name)
            except FileExistsError:
                logger.warning(
                    'The preprocessed file already exist for '
                    f'{fif_file.name}.')
