"""
White Noise sound.
"""

import numpy as np

from ._sound import _Sound
from ...utils.docs import fill_doc


@fill_doc
class WhiteNoise(_Sound):
    """
    White noise stimuli.

    Parameters
    ----------
    %(audio_volume)s
    %(audio_sample_rate)s
    %(audio_duration)s
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
