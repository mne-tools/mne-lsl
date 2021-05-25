import copy
import numpy as np

from neurodecode import logger


def set_eeg_reference(inst, ref_channels, ref_old=None, bads=None):
    """
    Reference to new channels.

    Parameters
    ----------
    inst : numpy.ndarray (n_channels x n_samples)
        The raw data.
    ref_channels : list of int | int
        Can be:
        - The name(s) of the channel(s) used to construct the reference.
        - 'average' to apply an average reference (CAR)
    ref_old : list of str | list of int | str
        Channel(s) to recover.
    bads : None | list of int
        The bad channels to ignore for an average reference.
    """
    if ref_channels != 'average':
        if isinstance(ref_channels, int):
            ref_channels = [ref_channels]
        if not (all(isinstance(ref_ch, int) for ref_ch in ref_channels)
                and all(0 <= ref_ch <= inst.shape[0] for ref_ch in ref_channels)):
            raise ValueError(f'The new reference channel indices {ref_channels} '
                             f'are not in raw.shape {inst.shape[0]}.')

    if ref_old is not None:
        if isinstance(ref_old, (list, tuple, np.ndarray)):
            ref_old = len(ref_old)

        refs = np.zeros((ref_old, inst.shape[1]))
        inst = np.vstack((inst, refs))  # this can not be done in-place

    if isinstance(ref_channels, (list, tuple)):
        if bads is not None:
            bad_in_new_reference = set(ref_channels).intersection(set(bads))
            if len(bad_in_new_reference) != 0:
                logger.warning(
                    f'Channels marked as bad are present in the new reference {ref_channels}')
        inst -= np.mean(inst[ref_channels], axis=0)

    elif ref_channels == 'average':
        if bads is None:
            car = np.mean(inst, axis=0)
        else:
            car = np.mean(np.delete(copy.deepcopy(inst), bads), axis=0)

        inst -= car
