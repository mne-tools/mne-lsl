"""Datasets utilities. Inspired from MNE."""

from . import eeg_auditory_stimuli  # noqa: F401
from . import eeg_resting_state  # noqa: F401
from . import eeg_resting_state_short  # noqa: F401
from . import trigger_def  # noqa: F401


def _download_all_datasets():
    """
    Download all the datasets.
    """
    eeg_auditory_stimuli.data_path()
    eeg_resting_state.data_path()
    eeg_resting_state_short.data_path()
    trigger_def.data_path()
