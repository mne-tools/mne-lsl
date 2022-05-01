"""Datasets utilities. Inspired from MNE."""

from . import eeg_auditory_stimuli
from . import eeg_resting_state
from . import eeg_resting_state_short
from . import trigger_def

__all__ = [
    "eeg_auditory_stimuli",
    "eeg_resting_state",
    "eeg_resting_state_short",
    "trigger_def",
]


def _download_all_datasets():
    """
    Download all the datasets.
    """
    eeg_auditory_stimuli.data_path()
    eeg_resting_state.data_path()
    eeg_resting_state_short.data_path()
    trigger_def.data_path()
