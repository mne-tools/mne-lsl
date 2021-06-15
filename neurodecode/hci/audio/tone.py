"""
White Noise

@author: Mathieu Scheltienne
"""

import numpy as np

from ._sound import _Sound
from ... import logger


class Tone(_Sound):
    """
    Pure tone stimuli at the frequency f (Hz).
    The equation is sin(2*pi*f*time).

    Example: A 440 - La 440 - Tone(f=440)

    Parameters
    ----------
    volume : list | int | float, optional
        If an int or a float is provided, the sound will use only one channel
        (mono). If a 2-length sequence is provided, the sound will use 2
        channels (stereo).
        The volume of each channel is given between 0 and 100. For stereo, the
        volume is given as [L, R].
    frequency : int
        Pure tone frequency. The default is 440 Hz (La - A440).
    sample_rate : int, optional
        Sampling frequency of the sound. The default is 44100 kHz.
    duration : float, optional
        The duration of the sound. The default is 1.0 second.
    """

    def __init__(self, volume, frequency=440, sample_rate=44100, duration=1.0):
        self.name = 'tone'
        self._frequency = Tone._check_frequency(frequency)
        super().__init__(volume, sample_rate, duration)

    def _compute_signal(self):
        """
        Computes the signal to output.
        """
        tone_arr = np.sin(2*np.pi*self._frequency*self._time_arr)

        self._signal[:, 0] = tone_arr * self._volume[0] / 100
        if len(self._volume) == 2:
            self._signal[:, 1] = tone_arr * self._volume[1] / 100

    # --------------------------------------------------------------------
    @staticmethod
    def _check_frequency(frequency):
        """
        Checks if the frequency is positive.
        """
        frequency = float(frequency)
        if frequency <= 0:
            logger.error(
                'The frequency must be positive. '
                f'Provided {frequency} Hz.')
            raise ValueError

        return frequency

    # --------------------------------------------------------------------
    @property
    def frequency(self):
        """
        The sound's pure tone frequency.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._frequency = Tone._check_frequency(frequency)
        self._compute_signal()
