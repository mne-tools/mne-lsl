"""
Export fif raw data to different format.

Supported:
    - EEGLAB ``'.set'``
"""
from scipy.io import savemat
from numpy.core.records import fromarrays


# ----------------------------- EEG LAB -----------------------------
def write_set(raw, fname):
    """
    Export `~mne.io.Raw` to EEGLAB ``.set`` file.

    Source: MNELAB
    https://github.com/cbrnr/mnelab/blob/main/mnelab/io/writers.py

    Parameters
    ----------
    raw : `~mne.io.Raw`
        MNE `~mne.io.Raw` instance.
    fname : `str` | `~pathlib.Path`
        Name/Path of the ``'.set'`` file created.
    """
    data = raw.get_data() * 1e6  # convert to microvolts
    sample_rate = raw.info["sfreq"]
    times = raw.times
    ch_names = raw.info["ch_names"]
    chanlocs = fromarrays([ch_names], names=["labels"])
    events = fromarrays([raw.annotations.description,
                         raw.annotations.onset * sample_rate + 1,
                         raw.annotations.duration * sample_rate],
                        names=["type", "latency", "duration"])
    savemat(fname,
            dict(EEG=dict(data=data,
                          setname=fname,
                          nbchan=data.shape[0],
                          pnts=data.shape[1],
                          trials=1,
                          srate=sample_rate,
                          xmin=times[0],
                          xmax=times[-1],
                          chanlocs=chanlocs,
                          event=events,
                          icawinv=[],
                          icasphere=[],
                          icaweights=[])),
            appendmat=False)
