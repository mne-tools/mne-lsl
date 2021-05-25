import numpy as np


def find_event_channel(inst=None, ch_names=None):
    """
    Find the event channel using heuristics.

    Disclaimer: Not 100% guaranteed to find it.
    If raw is None, ch_names must be given.

    Parameters
    ----------
    inst : mne.io.Raw | numpy.ndarray (n_channels x n_samples)
        The data
    ch_names : list
        The channels name list

    Returns:
    --------
    int : The event channel index or None if not found.
    """
    valid_trigger_ch_names = ['TRIGGER', 'STI', 'TRG', 'CH_Event']

    # For numpy array
    if isinstance(inst, np.ndarray):
        if ch_names is not None:
            for ch_name in ch_names:
                if any(trigger_ch_name in ch_name for trigger_ch_name in valid_trigger_ch_names):
                    return ch_names.index(ch_name)

        # data range between 0 and 255 and all integers?
        for ch in range(inst.shape[0]):
            if (inst[ch].astype(int) == inst[ch]).all()\
                    and max(inst[ch]) < 256 and min(inst[ch]) == 0:
                return ch

    # For MNE raw
    elif hasattr(inst, 'ch_names'):
        if 'stim' in inst.get_channel_types():
            return inst.get_channel_types().index('stim')

        for ch_name in inst.ch_names:
            if any(trigger_ch_name in ch_name for trigger_ch_name in valid_trigger_ch_names):
                return inst.ch_names.index(ch_name)

    # For unknown data type
    else:
        if ch_names is None:
            raise ValueError('ch_names cannot be None when raw is None.')
        for ch_name in ch_names:
            if any(trigger_ch_name in ch_name for trigger_ch_name in valid_trigger_ch_names):
                return ch_names.index(ch_name)

    return None
