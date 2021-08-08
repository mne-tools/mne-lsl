"""
Pure tone sound.
"""

import numpy as np

from ._sound import _Sound
from ... import logger
from ...utils.docs import fill_doc


@fill_doc
class Tone(_Sound):
    """
    Pure tone stimuli at the frequency f (Hz).
    The equation is ``sin(2*pi*f*time)``.

    Example: A 440 - La 440 - Tone(f=440)

    Parameters
    ----------
    %(audio_volume)s
    frequency : int
        Pure tone frequency. The default is 440 Hz (La - A440).
    %(audio_sample_rate)s
    %(audio_duration)s
    """

    def __init__(self, volume, frequency=440, sample_rate=44100, duration=1.0):
        self.name = 'tone'
        self._frequency = Tone._check_frequency(frequency)
        super().__init__(volume, sample_rate, duration)

    def _set_signal(self):
        """
        Sets the signal to output.
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
        Sound's pure tone frequency.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._frequency = Tone._check_frequency(frequency)
        self._set_signal()
