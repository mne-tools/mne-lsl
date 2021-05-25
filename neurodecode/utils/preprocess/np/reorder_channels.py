import numpy as np


def reorder_channels(inst, new_order):
    """
    Reorder the channels (rows) of the data array.

    Parameters
    ----------
    inst : numpy.ndarray (n_channels x n_samples)
        The raw data.
    new_order : list
        The new channel order (indices).

    Returns
    -------
    numpy.ndarray (n_channels x n_samples)
        The data re-ordered.
    """
    if isinstance(new_order, np.ndarray):
        new_order = list(new_order)
    if not inst.shape[0] == len(new_order):
        raise ValueError(
            "The new order does not have the same number of channels.")

    return inst[new_order, :]
