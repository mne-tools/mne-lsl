import numpy as np

from .... import logger


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
        logger.error(
            "The new order does not have the same number of channels.")
        raise ValueError

    return inst[new_order, :]
