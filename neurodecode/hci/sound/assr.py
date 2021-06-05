"""
Auditory Steady State Response Stimuli

@author: Mathieu Scheltienne
"""

import numpy as np

from ._sound import _Sound
from ... import logger


class ASSR(_Sound):
    """
    Auditory Steady State Response Stimuli.
    Composed of a carrier frequency 'fc' which is amplitude modulated at 'fm'.

    By default, a 1000 Hz carrier frequency modulated at 40 Hz through
    conventional modulation.

    Parameters
    ----------
    volume : list | int | float, optional
        If an int or a float is provided, the sound will use only one channel
        (mono). If a 2-length sequence is provided, the sound will use 2
        channels (stereo).
        The volume of each channel is given between 0 and 100. For stereo, the
        volume is given as [L, R].
    fc : int
        Carrier frequency in Hz.
    fm : int
        Modulatiom frequency in Hz.
    fs : int, optional
        Sampling frequency of the sound.
        The default is 44100 kHz.
    method : str
        Either 'conventional' or 'dsbsc'.
        'conventional':
            Also called 'classical AM', the equation used is:
                signal = (1 - M(t)) * cos(2*pi*fc*t)
                M(t) = cos(2*pi*fm*t)
        'dsbsc':
            Also called 'double side band suppressed carrier', the equation
            used is:
                signal = M(t)*cos(2*pi*fc*t)
                M(t) = sin(2*pi*fm*t)
    duration : float, optional
        The duration of the sound. The default is 1.0 second.
    """

    def __init__(self, volume, fc=1000, fm=40, method='conventional',
                 fs=44100, duration=1.0):
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
        method = method.lower().strip()
        if method not in ('conventional', 'dsbsc'):
            logger.error(
                "Supported amplitude modulation methods are 'conventional' "
                f"and 'dsbsc'. Provided: '{method}'.")

        super().__init__(fs, duration, len(volume))
        self.name = f'assr {method}'
        self.fc = int(fc)
        self.fm = int(fm)
        self.volume = volume

        if method == 'conventional':
            self.assr_amplitude = (1-np.cos(2*np.pi*self.fm*self.t))
            assr_arr = self.assr_amplitude * np.cos(2*np.pi*self.fc*self.t)

            self.signal[:, 0] = assr_arr * self.volume[0] / 100
            if len(self.volume) == 2:
                self.signal[:, 1] = assr_arr * self.volume[1] / 100

        elif method == 'dsbsc':
            self.assr_amplitude = np.sin(2*np.pi*self.fm*self.t)
            assr_arr = self.assr_amplitude * np.sin(2*np.pi*self.fc*self.t)

            self.signal[:, 0] = assr_arr * self.volume[0] / 100
            if len(self.volume) == 2:
                self.signal[:, 1] = assr_arr * self.volume[1] / 100
