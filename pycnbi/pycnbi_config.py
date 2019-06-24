"""
PyCNBI global configurations

Kyuhwa Lee, 2014

"""

import os
PYCNBI_ROOT = os.path.dirname(os.path.realpath(__file__))

# channel names (trigger channel is index 0 so that eeg channels start from index 1)
CAP = {
    'GTEC_16': ['TRIGGER', 'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
               'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4'],
    'GTEC_16_INFO': ['stim'] + ['eeg'] * 16,
    'BIOSEMI_64': ['TRIGGER', 'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7',
                  'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
                  'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz',
                  'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz',
                  'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                  'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4',
                  'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'EXG1', 'EXG2', 'EXG3',
                  'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'],
    'BIOSEMI_64_INFO': ['stim'] + ['eeg'] * 64 + ['misc'] * 8,
    'SMARTBCI_24': ['TRIGGER',
                   'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8', 'CH9', 'CH10',
                   'CH11', 'CH12', 'CH13', 'CH14', 'CH15', 'CH16', 'CH17', 'CH18', 'CH19', 'CH20',
                   'CH21', 'CH22', 'CH23'],
    # 'SMARTBCI_24_INFO': ['stim'] + ['eeg']*8 + ['misc'] + ['eeg']*6 + ['misc'] + ['eeg']*7 #+ ['misc']
    'SMARTBCI_24_INFO': ['stim'] + ['eeg'] * 23,  # CHANGE TO 24
    'ANTNEURO_MI15': ['TRIGGER'] + ['Fz', 'FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'FC3',
                                   'FCz', 'FC4', 'C1', 'C2', 'CP3', 'CP4'],
    'ANTNEURO_MI15_INFO': ['stim'] + ['eeg'] * 15,
    'ANTNEURO_64': ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz'],
    'ANTNEURO_64_NO_PERIPHERAL': ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'P3', 'Pz', 'P4', 'POz', 'AF3', 'AF4', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6'],
    'DSI_24': ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'T5', 'T6', 'Pz', 'P3', 'P4', 'O1', 'O2', 'A1', 'A2', 'X1', 'X2', 'X3']
}

# spatial laplacian channel definitions
# TODO: Add BioSemi 64 channels
LAPLACIAN = {
    'GTEC_16':{1:[4], 2:[3, 7], 3:[2, 4, 8], 4:[3, 5, 9], 5:[4, 6, 10], 6:[5, 11],
               7:[2, 8, 12], 8:[3, 7, 9, 13], 9:[4, 8, 10, 14], 10:[5, 9, 11, 15], 11:[6, 10, 16],
               12:[7, 13], 13:[8, 12, 14], 14:[9, 13, 15], 15:[10, 14, 16], 16:[11, 15]}
}
