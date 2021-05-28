"""
Module for simple MNE data preprocessing.
"""

from .current_source_density import (current_source_density,
                                     dir_current_source_density)
from .filter import (spectral_filter, dir_spectral_filter,
                     notch_filter, dir_notch_filter)
from .rename_channels import rename_channels, dir_rename_channels
from .resample import resample, dir_resample
from .set_channel_types import set_channel_types, dir_set_channel_types
from .set_eeg_reference import set_eeg_reference, dir_set_eeg_reference
