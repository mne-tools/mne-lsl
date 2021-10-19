import numpy as np
from mne.io import BaseRaw
from mne.evoked import Evoked
from mne.epochs import BaseEpochs

from ._checks import _check_type


def find_event_channel(inst=None, ch_names=None):  # noqa: E501
    """
    Find the event channel using heuristics.

    Disclaimer: Not 100% guaranteed to find it.
    If ``inst`` is `None`, ``ch_names`` must be given.

    Parameters
    ----------
    inst : `None` | `~mne.io.Raw` | `~mne.Epochs` | `~mne.Evoked` | `~numpy.array` ``(n_channels, n_samples)``
        Data instance.
    ch_names : `None` | `list`
        Channels name list.

    Returns
    -------
    `int` | `None`
        Event channel index or `None` if not found.
    """
    _check_type(inst, (None, np.ndarray, BaseRaw, BaseEpochs, Evoked), 'inst')
    _check_type(ch_names, (None, list, tuple), 'ch_names')

    valid_trigger_ch_names = ['TRIGGER', 'STI', 'TRG', 'CH_Event']

    # For numpy array
    if isinstance(inst, np.ndarray):
        if ch_names is not None:
            for ch_name in ch_names:
                if any(trigger_ch_name in ch_name
                       for trigger_ch_name in valid_trigger_ch_names):
                    return ch_names.index(ch_name)

        # data range between 0 and 255 and all integers?
        for ch_idx in range(inst.shape[0]):
            if (inst[ch_idx].astype(int, copy=False) == inst[ch_idx]).all() \
                    and max(inst[ch_idx]) < 256 and min(inst[ch_idx]) == 0:
                return ch_idx

    # For MNE inst
    elif hasattr(inst, 'ch_names'):
        if 'stim' in inst.get_channel_types():
            return inst.get_channel_types().index('stim')

        for ch_name in inst.ch_names:
            if any(trigger_ch_name in ch_name
                   for trigger_ch_name in valid_trigger_ch_names):
                return inst.ch_names.index(ch_name)

    # For unknown data type
    else:
        if ch_names is None:
            raise ValueError('ch_names cannot be None when inst is None.')
        for ch_name in ch_names:
            if any(trigger_ch_name in ch_name
                   for trigger_ch_name in valid_trigger_ch_names):
                return ch_names.index(ch_name)

    return None
