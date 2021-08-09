"""
Base class for sound delivery.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.io import wavfile

from ... import logger
from ...utils._docs import fill_doc
from ...utils._imports import import_optional_dependency

sd = import_optional_dependency(
    "sounddevice", extra="Install sounddevice for audio/sound support.")


@fill_doc
class _Sound(ABC):
    """
    Base audio stimulus class.

    Parameters
    ----------
    %(audio_volume)s
    %(audio_sample_rate)s
    %(audio_duration)s
    """

    @abstractmethod
    def __init__(self, volume, sample_rate=44100, duration=1.0):
        self._volume = _Sound._check_volume(volume)
        self._sample_rate = _Sound._check_sample_rate(sample_rate)
        self._duration = _Sound._check_duration(duration)
        self._time_arr = np.linspace(
            0, duration, int(duration*sample_rate), endpoint=True)
        # [:, 0] for left and [:, 1] for right
        self._signal = np.zeros(shape=(self._time_arr.size, len(self._volume)))

        self._set_signal()

    @abstractmethod
    def _set_signal(self):
        """
        Sets the signal to output.
        """
        pass

    # --------------------------------------------------------------------
    def play(self, blocking=False):
        """
        Play the sound. This function creates and terminates an audio stream.
        """
        sd.play(self._signal, samplerate=self._sample_rate, mapping=[1, 2])
        if blocking:
            sd.wait()

    def stop(self):
        """
        Stops the sounds played in the background.
        """
        sd.stop()

    def write(self, fname):
        """
        Save a sound signal into a ``.wav`` file with
        ``scipy.io.wavfile.write()``.

        Parameters
        ----------
        fname : str, path
            Path to the file where the sound signal is saved. The extension
            should be ``'.wav'``.
        """
        wavfile.write(fname, self._sample_rate, self._signal)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_volume(volume):
        """
        Checks that the volume is either:
            - 1 number, 1-item iterable for mono.
            - 2 numbers in a 2-item iterable for stereo.
        Checks that the volume value is between ``[0, 100]``.
        """
        if isinstance(volume, (int, float)):
            volume = [volume]
        if not isinstance(volume, (list, tuple)):
            logger.error(
                'Volume must be either an (int | float) for mono OR '
                'a (list | tuple) for mono or stereo.'
                f'Provided {type(volume)}.')
        if len(volume) not in (1, 2):
            logger.error(
                'Volume must be a 1-length (mono) or a '
                '2-length (stereo) sequence. '
                f'Provided {len(volume)} channels.')
            raise ValueError
        if not all(0 <= v <= 100 for v in volume):
            logger.error(
                f'Volume must be set between 0 and 100. Provided {volume}.')
            raise ValueError

        return volume

    @staticmethod
    def _check_sample_rate(sample_rate):
        """
        Checks if the sample rate is a positive integer.
        """
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            logger.error(
                'The sample rate must be a positive integer. '
                f'Provided {sample_rate} Hz.')
            raise ValueError
        if sample_rate not in (44100, 48000):
            logger.warning(
                f'Sound sample rate {sample_rate} different from the usual '
                '44 100 Hz or 48 000 Hz.')

        return sample_rate

    @staticmethod
    def _check_duration(duration):
        """
        Checks if the duration is positive.
        """
        duration = float(duration)
        if duration <= 0:
            logger.error(
                'The duration must be positive. '
                f'Provided {duration} seconds.')
            raise ValueError

        return duration

    # --------------------------------------------------------------------
    @property
    def volume(self):
        """
        Sound's volume(s).
        """
        return self._volume

    @volume.setter
    def volume(self, volume):
        self._volume = _Sound._check_volume(volume)
        self._signal = np.zeros(shape=(self._time_arr.size, len(self._volume)))
        self._set_signal()

    @property
    def sample_rate(self):
        """
        Sound's sampling rate.
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self._sample_rate = _Sound._check_sample_rate(sample_rate)
        self._time_arr = np.linspace(
            0, self._duration,
            int(self._duration*self._sample_rate), endpoint=True)
        self._signal = np.zeros(shape=(self._time_arr.size, len(self._volume)))
        self._set_signal()

    @property
    def duration(self):
        """
        Sound's duration (seconds).
        """
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = _Sound._check_duration(duration)
        self._time_arr = np.linspace(
            0, self._duration,
            int(self._duration*self._sample_rate), endpoint=True)
        self._signal = np.zeros(shape=(self._time_arr.size, len(self._volume)))
        self._set_signal()

    @property
    def signal(self):
        """
        Sound's signal.
        """
        return self._signal
