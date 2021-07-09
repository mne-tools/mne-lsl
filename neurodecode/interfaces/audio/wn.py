"""
White Noise

@author: Mathieu Scheltienne
"""

import numpy as np

from ._sound import _Sound


class WhiteNoise(_Sound):
    """
    White noise stimuli.

    Parameters
    ----------
    volume : list | int | float, optional
        If an int or a float is provided, the sound will use only one channel
        (mono). If a 2-length sequence is provided, the sound will use 2
        channels (stereo).
        Volume of each channel, given between 0 and 100. For stereo, the
        volume is given as [L, R].
    sample_rate : int, optional
        Sampling frequency of the sound. The default is 44100 kHz.
    duration : float, optional
        Duration of the sound. The default is 1.0 second.
    """

    def __init__(self, volume, sample_rate=44100, duration=1.0):
        self.name = 'whitenoise'
        self._rng = np.random.default_rng()
        super().__init__(volume, sample_rate, duration)

    def _set_signal(self):
        """
        Sets the signal to output.
        """
        # mean: 0, sigma: 0.33
        wn_arr = self._rng.normal(loc=0, scale=1/3, size=self._time_arr.size)

        self._signal[:, 0] = wn_arr * 0.1 * self._volume[0] / 100
        if len(self._volume) == 2:
            self._signal[:, 1] = wn_arr * 0.1 * self._volume[1] / 100
