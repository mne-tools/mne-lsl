"""Convert BSL file format to FIF."""

import os
import pickle
from pathlib import Path

import mne
import numpy as np

from . import find_event_channel
from ._logs import logger

mne.set_log_level("ERROR")


# ------------------------- Stream Recorder PCL -------------------------
def pcl2fif(
    fname,
    out_dir=None,
    external_event=None,
    precision="double",
    replace=False,
    overwrite=True,
):
    """Convert BSL pickle format to MNE Raw format.

    Parameters
    ----------
    fname : file-like
        Pickle file path to convert to ``.fif`` format.
    out_dir : path-like
        Saving directory. If ``None``, it will be the directory
        ``fname.parent/'fif'``.
    external_event : file-like
        Event file path in text format, following MNE event structure. Each row
        should be: ``index 0 event``.
    precision : str
        Data matrix format. ``[single|double|int|short]``, ``'single'``
        improves backward compatibility.
    replace : bool
        If `True`, previous events will be overwritten by the new ones from
        the external events file.
    overwrite : bool
        If ``True``, overwrite the previous file.
    """
    fname = Path(fname)
    if not fname.is_file():
        raise IOError("File %s not found." % fname)
    if fname.suffix != ".pcl":
        raise IOError("File type %s is not '.pcl'." % fname.suffix)

    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = fname.parent / "fif"
    os.makedirs(out_dir, exist_ok=True)

    fiffile = out_dir / str(fname.stem + ".fif")

    # Load from file
    with open(fname, "rb") as file:
        data = pickle.load(file)

    # MNE format
    raw = _format_pcl_to_mne_RawArray(data)

    # Add events from txt file
    if external_event is not None:
        events = _load_events_from_txt(raw.times, external_event, data["timestamps"][0])
        if 0 < len(events):
            raw.add_events(events, stim_channel="TRIGGER", replace=replace)

    # Save
    raw.save(fiffile, verbose=False, overwrite=overwrite, fmt=precision)
    logger.info("Data saved to: '%s'", fiffile)


def _format_pcl_to_mne_RawArray(data):
    """Format the raw data to the MNE RawArray structure.

    Data must be recorded with BSL StreamRecorder.

    Parameters
    ----------
    data : dict
        Data loaded from the .pcl file.

    Returns
    -------
    raw : Raw
        MNE raw structure.
    """
    if isinstance(data["signals"], list):
        signals_raw = np.array(data["signals"][0]).T  # to channels x samples
    else:
        signals_raw = data["signals"].T  # to channels x samples

    sample_rate = data["sample_rate"]

    # Look for channels name
    if "ch_names" not in data:
        ch_names = [f"CH{x+1}" for x in range(signals_raw.shape[0])]
    else:
        ch_names = data["ch_names"]

    # search for the trigger channel
    trig_ch = find_event_channel(signals_raw, ch_names)
    # TODO: patch to be improved for multi-trig channel recording
    if isinstance(trig_ch, list):
        trig_ch = trig_ch[0]

    # move trigger channel to index 0
    if trig_ch is None:
        # Add a event channel to index 0 for consistency.
        logger.warning(
            "Event channel was not found. " "Adding a blank event channel to index 0."
        )
        eventch = np.zeros([1, signals_raw.shape[1]])
        signals = np.concatenate((eventch, signals_raw), axis=0)
        # data['channels'] is not reliable any more
        num_eeg_channels = signals_raw.shape[0]
        trig_ch = 0
        ch_names = ["TRIGGER"] + ch_names

    elif trig_ch == 0:
        signals = signals_raw
        num_eeg_channels = data["channels"] - 1

    else:
        logger.info("Moving event channel %s to 0.", trig_ch)
        signals = np.concatenate(
            (
                signals_raw[[trig_ch]],
                signals_raw[:trig_ch],
                signals_raw[trig_ch + 1 :],
            ),
            axis=0,
        )
        assert signals_raw.shape == signals.shape
        num_eeg_channels = data["channels"] - 1
        ch_names.pop(trig_ch)
        trig_ch = 0
        ch_names.insert(trig_ch, "TRIGGER")
        logger.info("New channel list:")
        for channel in ch_names:
            logger.info("%s", channel)

    ch_info = ["stim"] + ["eeg"] * num_eeg_channels
    info = mne.create_info(ch_names, sample_rate, ch_info)

    # create Raw object
    raw = mne.io.RawArray(signals, info)

    return raw


def _load_events_from_txt(raw_times, eve_file, offset):  # noqa
    """Load events delivered by the software trigger from the event txt file,
    and convert LSL timestamps to indices.
    """

    ts_min = min(raw_times)
    ts_max = max(raw_times)
    events = []

    with open(eve_file, "r") as file:
        for line in file:
            data = line.strip().split("\t")
            event_ts = float(data[0]) - offset
            event_value = int(data[2])
            next_index = np.searchsorted(raw_times, event_ts)
            if next_index >= len(raw_times):
                logger.warning(
                    "Event %d at time %.3f is out of time range" " (%.3f - %.3f).",
                    event_value,
                    event_ts,
                    ts_min,
                    ts_max,
                )
            else:
                events.append([next_index, 0, event_value])

    return np.array(events)


def _add_events_from_txt(raw, events_index, stim_channel="TRIGGER", replace=False):
    """Merge the events extracted from a .txt file to the trigger channel.

    Parameters
    ----------
    raw : Raw
        MNE Raw instance.
    events_index : np.array
        MNE-compatible events [shape=(n_events, 3)].
        Used as input to raw.add_events.
    stim_channel : str
        Stim channel where the events are added.
    replace : bool
        If True, the old events on the stim channel are removed before
        adding the new ones.
    """
    if len(events_index) == 0:
        logger.warning("No events were found in the event file.")
    else:
        logger.info("Found %i events", len(events_index))
        raw.add_events(events_index, stim_channel=stim_channel, replace=replace)
