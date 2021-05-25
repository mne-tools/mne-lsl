import mne


def set_eeg_reference(inst, ref_channels, ref_old=None, **kwargs):
    """
    Reference to new channels. MNE raw object is modified in-place for efficiency.

    Parameters
    ----------
    inst : mne.io.Raw | mne.io.RawArray | mne.Epochs | mne.Evoked
        MNE instance of Raw | Epochs | Evoked. Assumes the 'eeg' type is
        correctly set for EEG channels.
    ref_channels : list of str | str
        Can be:
        - The name(s) of the channel(s) used to construct the reference.
        - 'average' to apply an average reference (CAR)
        - 'REST' to use the reference electrode standardization technique
        infinity reference (requires instance with montage forward arg).
    ref_old : list of str | str
        Channel(s) to recover.
    **kwargs : Additional arguments are passed to mne.set_eeg_reference()
        c.f. https://mne.tools/dev/generated/mne.set_eeg_reference.html
    """
    if not (all(isinstance(ref_ch, str) for ref_ch in ref_channels)
            or isinstance(ref_channels, str)):
        raise ValueError(
            "The new reference channel must be a list of strings "
            "or 'average' or 'REST'.")

    if ref_old is not None:
        mne.add_reference_channels(inst, ref_old, copy=False)

    mne.set_eeg_reference(inst, ref_channels, copy=False, **kwargs)
