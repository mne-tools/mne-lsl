"""
Real-time cardiac R-peak detection
==================================

With a :class:`~mne_lsl.stream.StreamLSL` connected to an amplifier stream containing an
EEG bipolar channel, we can detect in real-time the R-peak within the ECG signal. The
objective of this example is to create a ``Detector`` object able to detect new R-peak
entering the buffer as fast as possible, with some robustness to external noise sources
(e.g. movements) and a simple design.

.. image:: ../../_static/tutorials/qrs.png
    :align: center
    :class: qrs-img

First let's have a look to a sample ECG signal and to how we could detect the R-peak
reliably with :func:`scipy.signal.find_peaks`.
"""

from mne.io import read_raw_fif
from mne_lsl.datasets import sample

raw = read_raw_fif(sample.data_path() / "sample-ecg-raw.fif", preload=True)
raw

# %%
# This sample recording contains a single channel with the ECG signal.

from time import sleep

import numpy as np
from mne_lsl.stream import StreamLSL
from numpy.typing import NDArray
from scipy.signal import find_peaks


class Detector:
    """Real-time single channel peak detector.

    Parameters
    ----------
    bufsize : float
        Size of the buffer in seconds. The buffer will be filled on instantiation, thus
        the program will hold during this duration.
    stream_name : str
        Name of the LSL stream to use for the respiration or cardiac detection. The
        stream should contain a respiration channel using a respiration belt or a
        thermistor and/or an ECG channel.
    ch_name : str
        Name of the ECG channel in the LSL stream. This channel should contain the ECG
        signal recorded with 2 bipolar electrodes.
    ecg_height : float | None
        The height of the ECG peaks as a percentage of the data range, between 0 and 1.
    ecg_distance : float | None
        The minimum distance between two ECG peaks in seconds.
    """

    def __init__(
        self,
        bufsize: float,
        stream_name: str,
        ch_name: str,
        ecg_height: float | None = None,
        ecg_distance: float | None = None,
    ) -> None:
        self._ecg_height = ecg_height
        self._ecg_distance = ecg_distance
        # create stream
        self._stream = StreamLSL(bufsize, stream_name).connect(processing_flags="all")
        self._stream.pick(ch_name)
        self._stream.set_channel_types({ch_name: "misc" }, on_unit_change="ignore")
        self._stream.notch_filter(50, picks=ch_name)
        self._stream.notch_filter(100, picks=ch_name)
        sleep(bufsize)  # prefill an entire buffer
        # peak detection settings
        self._last_peak = None
        self._peak_candidates = None
        self._peak_candidates_count = None

    def _detect_peaks(self) -> NDArray[np.float64]:
        """Detect all peaks in the buffer.

        Returns
        -------
        peaks : array of shape (n_peaks,)
            The timestamps of all detected peaks.
        """
        data, ts = self._stream.get_data()  # we have a single channel in the stream
        data = data.squeeze()
        peaks, _ = find_peaks(
            data,
            distance=self._ecg_distance * self._stream.info["sfreq"],
            height=np.percentile(data, self._ecg_height * 100),
        )
        return ts[peaks]

    def new_peak(self) -> float | None:
        """Detect new peak entering the buffer.

        Returns
        -------
        peak : float | None
            The timestamp of the newly detected peak. None if no new peak is detected.
        """
        ts_peaks = self._detect_peaks()
        if ts_peaks.size == 0:
            return None  # unlikely to happen, but let's exit early if we have nothing
        if (
            self._peak_candidates is None
            and self._peak_candidates_count is None
        ):
            self._peak_candidates = list(ts_peaks)
            self._peak_candidates_count = [1] * ts_peaks.size
            return None
        peaks2append = []
        for k, peak in enumerate(self._peak_candidates):
            if peak in ts_peaks:
                self._peak_candidates_count[k] += 1
            else:
                peaks2append.append(peak)
        # before going further, let's make sure we don't add too many false positives
        if int(self._stream._bufsize * (1 / self._ecg_distance)) < len(
            peaks2append
        ) + len(self._peak_candidates):
            self._peak_candidates = None
            self._peak_candidates_count = None
            return None
        self._peak_candidates.extend(peaks2append)
        self._peak_candidates_count.extend([1] * len(peaks2append))
        # now, all the detected peaks have been triage, let's see if we have a winner
        idx = [
            k
            for k, count in enumerate(self._peak_candidates_count)
            if 4 <= count
        ]
        if len(idx) == 0:
            return None
        peaks = sorted([self._peak_candidates[k] for k in idx])
        # compare the winner with the last known peak
        if self._last_peak is None:  # don't return the first peak detected
            new_peak = None
            self._last_peak = peaks[-1]
        if (
            self._last_peak is None
            or self._last_peak + self._ecg_distance <= peaks[-1]
        ):
            new_peak = peaks[-1]
            self._last_peak = peaks[-1]
        else:
            new_peak = None
        # reset the peak candidates
        self._peak_candidates = None
        self._peak_candidates_count = None
        return new_peak
