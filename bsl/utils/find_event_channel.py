import numpy as np
from mne.io import BaseRaw
from mne.evoked import Evoked
from mne.epochs import BaseEpochs

from ._checks import _check_type


def find_event_channel(inst=None, ch_names=None):  # noqa: E501
    """
    Find the event channel using heuristics.

    .. warning::

        Not 100% guaranteed to find it.
        If ``inst`` is `None`, ``ch_names`` must be given.
        If ``inst`` is an MNE instance, ``ch_names`` is ignored.

    Parameters
    ----------
    inst : `None` | `~mne.io.Raw` | `~mne.Epochs` | `~numpy.array` ``(n_channels, n_samples)``
        Data instance.
    ch_names : `None` | `list`
        Channels name list.

    Returns
    -------
    `int` | `list` | `None`
        Event channel index, list of event channel indexes or `None` if not
        found.
    """
    _check_type(inst, (None, np.ndarray, BaseRaw, BaseEpochs, Evoked), 'inst')
    _check_type(ch_names, (None, list, tuple), 'ch_names')

    valid_trigger_ch_names = ['TRIGGER', 'STI', 'TRG', 'CH_Event']
    tchs = list()

    # For numpy array
    if isinstance(inst, np.ndarray):
        if ch_names is not None:
            for ch_name in ch_names:
                if any(trigger_ch_name in ch_name
                       for trigger_ch_name in valid_trigger_ch_names):
                    tchs.append(ch_names.index(ch_name))
        else:
            # data range between 0 and 255 and all integers?
            for ch_idx in range(inst.shape[0]):
                all_ints = (inst[ch_idx].astype(int, copy=False) == \
                            inst[ch_idx]).all()
                max255 = max(inst[ch_idx]) <= 255
                min0 = min(inst[ch_idx]) == 0
                if all_ints and max255 and min0:
                    tchs.append(ch_idx)

    # For MNE raw/epochs
    elif hasattr(inst, 'ch_names'):
        stim_types = [idx for idx, type_ in enumerate(inst.get_channel_types())
                      if type_ == 'stim']
        if len(stim_types) != 0:
            tchs.extend(stim_types)
        else:
            for ch_name in inst.ch_names:
                if any(trigger_ch_name in ch_name
                       for trigger_ch_name in valid_trigger_ch_names):
                    tchs.append(inst.ch_names.index(ch_name))

    # For unknown data type
    else:
        if ch_names is None:
            raise ValueError('ch_names cannot be None when inst is None.')
        for ch_name in ch_names:
            if any(trigger_ch_name in ch_name
                   for trigger_ch_name in valid_trigger_ch_names):
                tchs.append(ch_names.index(ch_name))

    # clean up
    tchs = list(set(tchs))

    if len(tchs) == 0:
        return None
    elif len(tchs) == 1:
        return tchs[0]
    else:
        return tchs
