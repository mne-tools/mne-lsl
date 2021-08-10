"""
Sound loaded from a file.
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

from ._sound import _Sound
from ... import logger
from ...utils._docs import copy_doc

SUPPORTED = ('.wav')


class Sound(_Sound):
    """
    Sound loaded from file.

    Parameters
    ----------
    fname : str | Path
        Path to the supported audio file to load.
    """

    def __init__(self, fname):
        self._fname = Sound._check_file(fname)

        _original_sample_rate, _original_signal = wavfile.read(self._fname)
        self._original_sample_rate = _Sound._check_sample_rate(
            _original_sample_rate)
        self._original_signal = Sound._check_signal(_original_signal)
        self._original_duration = Sound._compute_duration(
            self._original_signal, self._original_sample_rate)

        _volume = Sound._compute_volume(self._original_signal)
        self._trim_samples = None
        super().__init__(
            _volume, self._original_sample_rate, self._original_duration)

        logger.info(
            f'Sound loaded from {str(self._fname)}.\n'
            f'Sample rate: {self._sample_rate} Hz, '
            f'Duration: {np.round(self._duration, 2)} seconds.')

    @copy_doc(_Sound._set_signal)
    def _set_signal(self):
        slc = (slice(None, self._trim_samples), slice(None))
        self._signal = self._original_signal[slc]

    def trim(self, duration):
        """
        Trim the original sound to the new duration.
        """
        if Sound._valid_trim_duration(duration, self._original_duration):
            self._duration = _Sound._check_duration(duration)
            self._trim_samples = int(self._duration * self._sample_rate)
            self._set_signal()

    def resample(self, sample_rate):
        """
        Resample the curent sound to the new sampling rate.
        """
        logger.warning('Resampling a loaded sound will distort the sound.')
        self._sample_rate = _Sound._check_sample_rate(sample_rate)
        self._signal = resample(
            self._signal, int(self._sample_rate * self._duration), axis=0)

    def reset(self):
        """
        Reset the current sound to the original loaded sound.
        """
        self._duration = self._original_duration
        self._trim_samples = None
        self._sample_rate = self._original_sample_rate
        self._set_signal()

        logger.info(
            'Sound reset to original. '
            f'Sample rate: {self._sample_rate} Hz, '
            f'Duration: {np.round(self._duration, 2)} seconds.')

    # --------------------------------------------------------------------
    @staticmethod
    def _check_file(fname):
        """
        Cheks if the file is supported and exists.
        """
        fname = Path(fname)
        if fname.suffix not in SUPPORTED:
            logger.error(
                "Sound file format not supported. Supported: {SUPPORTED}. "
                f"Provided '{fname.name}'.")
            raise IOError

        if not fname.exists():
            logger.error(
                'The file provided does not exist. Check the path..')
            raise IOError

        return fname

    @staticmethod
    def _check_signal(signal):
        """
        Checks that the sound is either mono or stereo.
        """
        if signal.shape[1] not in (1, 2):
            logger.error(
                'Supported sounds must have 1 (mono) or 2 (stereo) channels '
                f'Provided {signal.shape[1]}.')
            raise TypeError
        return signal

    @staticmethod
    def _compute_duration(signal, sample_rate):
        """
        Computes the sounds duration from the number of samples and the
        sampling rate.
        """
        return signal.shape[0] / sample_rate

    @staticmethod
    def _compute_volume(signal):
        """
        Volume modifications is not supported for loaded sounds.
        Returns ``[1] * number of channels``.
        """
        return [1] * signal.shape[1]

    @staticmethod
    def _valid_trim_duration(trim_duration, sound_duration):
        """
        Returns ``True`` if ``trim_duration`` is smaller than
        ``sound_duration``.
        """
        if sound_duration <= trim_duration:
            logger.warning(
                'Requested trimming duration is shorter than the '
                'loaded sound. Skipping.')
            return False
        return True

    # --------------------------------------------------------------------
    @_Sound.volume.setter
    def volume(self, volume):
        logger.warning(
            "The sound's volume cannot be changed for a loaded sound.")

    @_Sound.sample_rate.setter
    def sample_rate(self, sample_rate):
        self.resample(sample_rate)

    @_Sound.duration.setter
    def duration(self, duration):
        self.trim(duration)

    @property
    def fname(self):
        """
        The sound's original file name.
        """
        return self._fname
