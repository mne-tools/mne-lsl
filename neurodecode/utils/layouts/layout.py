"""
Provide a class to contain:
 - name
 - montage
 - channel names
 - channel types
"""
from mne.io.pick import get_channel_type_constants
from mne.channels.montage import DigMontage, _BUILT_IN_MONTAGES

from ... import logger


CAP = {
    'GTEC_16': ['TRIGGER', 'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'],

    'BIOSEMI_64': ['TRIGGER',
                   'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                   'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                   'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                   'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                   'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
                   'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                   'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                   'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
                   'EXG1', 'EXG2', 'EXG3', 'EXG4',
                   'EXG5', 'EXG6', 'EXG7', 'EXG8'],

    'SMARTBCI_24': ['TRIGGER',
                    'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6',
                    'CH7', 'CH8', 'CH9', 'CH10', 'CH11', 'CH12',
                    'CH13', 'CH14', 'CH15', 'CH16', 'CH17', 'CH18',
                    'CH19', 'CH20', 'CH21', 'CH22', 'CH23'],

    'ANTNEURO_64': ['TRIGGER',
                    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                    'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz',
                    'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7',
                    'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2',
                    'EOG',
                    'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
                    'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3',
                    'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4',
                    'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz'],

    'DSI_24': ['TRIGGER',
               'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz',
               'Pz', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2', 'X3',
               'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4'],
}

CH_TYPES = {
    'GTEC_16': ['stim'] + ['eeg'] * 16,

    'BIOSEMI_64': ['stim'] + ['eeg'] * 64 + ['misc'] * 8,

    'SMARTBCI_24': ['stim'] + ['eeg'] * 23,

    'ANTNEURO_64': ['stim'] + ['eeg'] * 31 + ['eog'] + ['eeg'] * 32,

    'DSI_24': ['stim'] + ['eeg'] * 15 + ['misc'] * 2 + ['eeg'] * 2 +
              ['misc'] * 2 + ['eeg'] * 2,
}


def available_layouts(verbose=False):
    """
    Function returning the list of available layouts.

    Parameters
    ----------
    verbose : bool
        If True, display the available layout in the logger.
    """
    if verbose:
        logger.info('-- Available layouts --')
        for name in CAP:
            if CH_TYPES.get(name) is not None:
                if len(CAP[name]) == len(CH_TYPES[name]):
                    logger.info(f'  | {name}')
                else:
                    logger.warning(
                        f"  | Error with '{name}': "
                        "len() differs in CAP and CH_TYPES.")
            else:
                logger.warning(f"  | Error with '{name}': "
                               "KeyError in CH_TYPES.")

    return [name for name in CAP if CH_TYPES.get(name) is not None and
            len(CAP[name]) == len(CH_TYPES[name])]


class Layout:
    """
    Class containing the layout (Cap + AUX) information.
    Supports: GTEC_16, BIOSEMI_64, SMARTBCI_24, ANTNEURO_64, DSI_24.

    Parameters
    ----------
    name : str
        The name of the cap. Supported cap names have associated channel names
        and channel types saved in NeuroDecode.
    montage : str | DigMontage
        The montage used by the layout.
        https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_montage
    ch_names : list | None
        The list of channels' name in the order receied from LSL.
        If None, looks for a known list based on the layout name.
    ch_types : list | str | None
        The list of channels' type in the order received from LSL.
        If None, looks for a known list based on the layout name.
    """

    def __init__(self, name, montage=None, ch_names=None, ch_types=None):
        self._name = name
        self._montage = Layout._check_montage(montage)

        if ch_names is not None and ch_types is not None:
            ch_names = Layout._check_ch_names(ch_names)
            ch_types = Layout._check_ch_types(ch_names, ch_types)
            Layout._check_ch_number(ch_names, ch_types)

        else:
            try:
                ch_names = Layout._check_ch_names(CAP[name])
                ch_types = Layout._check_ch_types(ch_names, CH_TYPES[name])
                Layout._check_ch_number(ch_names, ch_types)
            except KeyError as error:
                logger.warning(
                    "The provided cap name is not yet included. "
                    "Add it first to NeuroDecode or provided the arguments "
                    "'ch_names' and 'ch_types'.")
                raise KeyError from error

        self._ch_names = ch_names
        self._ch_types = ch_types

    def add_channels(self, ch_names, ch_types):
        """
        Add a list of channels to the existing layout.

        ch_names : list
            The list of channels' name to add.
        ch_types : list | str
            The list of channels' type to add.
        """
        ch_names = Layout._check_ch_names(ch_names)
        ch_types = Layout._check_ch_types(ch_names, ch_types)
        Layout._check_ch_number(ch_names, ch_types)
        self._ch_names += ch_names
        self._ch_types += ch_types

    # --------------------------------------------------------------------
    @staticmethod
    def _check_ch_names(ch_names):
        """
        Checks that the channels names are a list of strings.
        """
        if not isinstance(ch_names, (list, tuple)):
            logger.error('The channel names must be provided as a list.')
            raise TypeError
        ch_names = [str(ch).strip() for ch in ch_names]
        return ch_names

    @staticmethod
    def _check_ch_types(ch_names, ch_types):
        supported = get_channel_type_constants(include_defaults=True)
        if isinstance(ch_types, str):
            ch_types = ch_types.strip().lower()
            if ch_types in supported:
                ch_types = [ch_types] * len(ch_names)
            else:
                logger.error(
                    f'Channel type {ch_types} is not supported.')
                raise ValueError

        elif isinstance(ch_types, (list, tuple)):
            ch_types = [str(ch_type).strip().lower() for ch_type in ch_types]
            if not all(ch_type in supported for ch_type in ch_types):
                logger.error('All provided channel types are not supported.')
                raise ValueError

        else:
            logger.error(
                'Channel types must be a list of the type of each channel or '
                'a string to apply the same type to every channel.')
            raise TypeError

        return ch_types

    @staticmethod
    def _check_ch_number(ch_names, ch_types):
        if len(ch_names) != len(ch_types):
            logger.error(
                f"The number of channels provided {len(ch_names)} does "
                f"not match the number of channels types {len(ch_types)}.")
            raise ValueError

    @staticmethod
    def _check_montage(montage):
        if montage is None:
            return montage

        if isinstance(montage, str):
            montage = montage.strip()
            if montage not in _BUILT_IN_MONTAGES:
                logger.error(
                    f'The provided montage name {montage} is not supported.')
                raise ValueError

        elif isinstance(montage, DigMontage):
            # TODO: Check the number of points vs the number of channels.
            pass

        else:
            logger.error(
                'The montage must be provided as a supported string or '
                'as a MNE DigMontage. ')
            raise TypeError

        return montage

    # --------------------------------------------------------------------

    @property
    def name(self):
        """
        The name of the layout.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def montage(self):
        """
        The montage used by the layout.
        """
        return self._montage

    @montage.setter
    def montage(self, montage):
        self._montage = Layout._check_montage(montage)

    @property
    def ch_names(self):
        """
        The list of channels' name in the order receied from LSL.
        """
        return self._ch_names

    @ch_names.setter
    def ch_names(self, ch_names):
        ch_names = Layout._check_ch_names(ch_names)
        if len(ch_names) == len(self._ch_types):
            self._ch_names = ch_names
        else:
            logger.warning(
                'The mumber of channel differs from the number of channel '
                'types. Skipping.')

    @property
    def ch_types(self):
        """
        The list of channels' type in the order received from LSL.
        """
        return self._ch_types

    @ch_types.setter
    def ch_types(self, ch_types):
        ch_types = Layout._check_ch_types(self.ch_names, ch_types)
        if len(ch_types) == len(self._ch_names):
            self._ch_types = ch_types
        else:
            logger.warning(
                'The mumber of channel types differs from the number of '
                'channels. Skipping.')
