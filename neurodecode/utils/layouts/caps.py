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

LAPLACIAN = {
    'GTEC_16': {1: [4], 2: [3, 7], 3: [2, 4, 8], 4: [3, 5, 9],
                5: [4, 6, 10], 6: [5, 11], 7: [2, 8, 12], 8: [3, 7, 9, 13],
                9: [4, 8, 10, 14], 10: [5, 9, 11, 15], 11: [6, 10, 16], 12: [7, 13],
                13: [8, 12, 14], 14: [9, 13, 15], 15: [10, 14, 16], 16: [11, 15]}
}


class Cap:
    """
    Class containing the cap information.
    Supports: GTEC_16, BIOSEMI_64, SMARTBCI_24, ANTNEURO_64, DSI_24.

    It provides the layout, channels' type and laplacian for spatial filtering.
    """

    def __init__(self, cap_name):
        """
        Constructor.

        Parameters
        ----------
        cap_name : str
        """
        try:
            self._layout = CAP[cap_name]
            self._ch_types = CH_TYPES[cap_name]
        except:
            raise KeyError(
                "The provided cap name is not yet included. "
                "Add it first to NeuroDecode.")
        try:
            self._laplacian = LAPLACIAN[cap_name]
        except:
            logger.warning(
                "The laplacian is not yet defined for this cap. "
                "Add it first to NeuroDecode to apply laplacian spatial filtering.")
            self._laplacian = None

    @property
    def layout(self):
        """
        Cap layout: channels' name in the order received from LSL.
        """
        return self._layout

    @layout.setter
    def layout(self, new_layout):
        self._layout = new_layout

    @property
    def ch_types(self):
        """
        The list of channels' type in the order received from LSL.
        """
        return self._ch_types

    @ch_types.setter
    def ch_types(self, new_ch_types):
        self._ch_types = new_ch_types

    @property
    def laplacian(self):
        """
        The laplacian for spatial filtering (neighbours of each electrodes).
        """
        return self._laplacian

    @laplacian.setter
    def laplacian(self, new_laplacian):
        self.laplacian = new_laplacian
