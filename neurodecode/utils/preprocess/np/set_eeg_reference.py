import copy
import numpy as np

from .... import logger


def set_eeg_reference(inst, ref_channels, ref_old=None, bads=None):
    """
    Reference to new channels.

    Parameters
    ----------
    inst : numpy.ndarray (n_channels x n_samples)
        The raw data. Assumes all data is 'eeg'.
    ref_channels : list of int | int
        List of channel indices to use as reference.
    ref_old : list of str | list of int | str
        Channel(s) to recover. List of indices, list of names.
    bads : None | list of int
        List of indices of bad channels to ignore for an average reference.

    Returns
    -------
    numpy.ndarray (n_channels x n_samples)
        The data re-referenced.
    """
    if ref_channels != 'average':
        if isinstance(ref_channels, int):
            ref_channels = [ref_channels]
        if not (all(isinstance(ref_ch, int) for ref_ch in ref_channels)
                and all(0 <= ref_ch <= inst.shape[0] for ref_ch in ref_channels)):
            raise ValueError(f'The new reference channel indices {ref_channels} '
                             f'are not in raw.shape {inst.shape[0]}.')

    if ref_old is not None:
        refs = np.zeros((len(ref_old), inst.shape[1]))
        inst = np.vstack((inst, refs))  # this can not be done in-place

    if isinstance(ref_channels, (list, tuple, np.ndarray)):
        if bads is not None:
            bad_in_new_reference = set(ref_channels).intersection(set(bads))
            if len(bad_in_new_reference) != 0:
                logger.warning(
                    'Channels marked as bad are present in '
                    f'the new reference {ref_channels}')
        inst -= np.mean(inst[ref_channels], axis=0)

    elif ref_channels == 'average':
        if bads is None:
            car = np.mean(inst, axis=0)
        else:
            car = np.mean(np.delete(copy.deepcopy(inst), bads), axis=0)

        inst -= car

    return inst
