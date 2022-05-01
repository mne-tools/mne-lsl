import numpy as np
from mne.io import BaseRaw
from mne.epochs import BaseEpochs

from ._checks import _check_type


def find_event_channel(inst=None, ch_names=None):
    """
    Find the event channel using heuristics.

    .. warning::

        Not 100% guaranteed to find it.
        If ``inst`` is ``None``, ``ch_names`` must be given.
        If ``inst`` is an MNE instance, ``ch_names`` is ignored if some
        channels types are ``'stim'``.

    Parameters
    ----------
    inst : None | Raw | Epochs | `~numpy.array`
        Data instance. If a `~numpy.array` is provided, the shape must be
        ``(n_channels, n_samples)``.
    ch_names : None | list
        Channels name list.

    Returns
    -------
    event_channel : int | list | None
        Event channel index, list of event channel indexes or ``None`` if not
        found.
    """
    _check_type(
        inst, (None, np.ndarray, BaseRaw, BaseEpochs), item_name="inst"
    )
    _check_type(ch_names, (None, list, tuple), item_name="ch_names")

    # numpy array + ch_names
    if isinstance(inst, np.ndarray) and ch_names is not None:
        tchs = _search_in_ch_names(ch_names)

    # numpy array without ch_names
    elif isinstance(inst, np.ndarray) and ch_names is None:
        # data range between 0 and 255 and all integers?
        tchs = [
            idx
            for idx in range(inst.shape[0])
            if (inst[idx].astype(int, copy=False) == inst[idx]).all()
            and max(inst[idx]) <= 255
            and min(inst[idx]) == 0
        ]

    # For MNE raw/epochs + ch_names
    elif isinstance(inst, (BaseRaw, BaseEpochs)) and ch_names is not None:
        tchs = [
            idx
            for idx, type_ in enumerate(inst.get_channel_types())
            if type_ == "stim"
        ]
        if len(tchs) == 0:
            tchs = _search_in_ch_names(ch_names)

    # For MNE raw/epochs without ch_names
    elif isinstance(inst, (BaseRaw, BaseEpochs)) and ch_names is None:
        tchs = [
            idx
            for idx, type_ in enumerate(inst.get_channel_types())
            if type_ == "stim"
        ]
        if len(tchs) == 0:
            tchs = _search_in_ch_names(inst.ch_names)

    # For unknown data type
    elif inst is None:
        if ch_names is None:
            raise ValueError("ch_names cannot be None when inst is None.")
        tchs = _search_in_ch_names(ch_names)

    # output
    if len(tchs) == 0:
        return None
    elif len(tchs) == 1:
        return tchs[0]
    else:
        return tchs


def _search_in_ch_names(ch_names):
    """Search trigger channel by name in a list of valid names."""
    valid_trigger_ch_names = ["TRIGGER", "STI", "TRG", "CH_Event"]

    tchs = list()
    for idx, ch_name in enumerate(ch_names):
        if any(
            trigger_ch_name in ch_name
            for trigger_ch_name in valid_trigger_ch_names
        ):
            tchs.append(idx)

    return tchs
