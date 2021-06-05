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
    f : int
        Pure tone frequency. The default is 440 Hz (La - A440).
    fs : int, optional
        Sampling frequency of the sound. The default is 44100 kHz.
    duration : float, optional
        The duration of the sound. The default is 1.0 second.
    """

    def __init__(self, volume, f=440, fs=44100, duration=1.0):
        if isinstance(volume, (int, float)):
            volume = [volume]
        if not all(0 <= v <= 100 for v in volume):
            logger.error(
                f'Volume must be set between 0 and 100. Provided {volume}.')
            raise ValueError
        if not len(volume) in (1, 2):
            logger.error(
                'Volume must be a 1-length (mono) or a '
                '2-length (stereo) sequence.')
            raise ValueError

        super().__init__(fs, duration, len(volume))
        self.name = 'tone'
        self.f = f
        self.volume = volume

        tone_arr = np.sin(2*np.pi*self.f*self.t)

        self.signal[:, 0] = tone_arr * volume[0] / 100
        if len(volume) == 2:
            self.signal[:, 1] = tone_arr * volume[1] / 100
