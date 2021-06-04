"""
Base class for sound delivery.

@author: Mathieu Scheltienne
"""

import numpy as np
import sounddevice as sd

from scipy.io import wavfile

from ... import logger


class _Sound:
    """
    Base audio stimulus class.

    Parameters
    ----------
    fs : int, optional
        Sampling frequency of the sound. The default is 44100 kHz.
    duration : float, optional
        The duration of the sound. The default is 1.0 second.
    channels : int, optional
        The number of audio channels. Supported:
            1: mono
            2: stereo
        The default is 1.
    """

    def __init__(self, fs=44100, duration=1.0, channels=1):
        self.fs = int(fs)
        self.duration = duration
        self.t = np.linspace(0, duration, int(duration*fs), endpoint=True)

        if channels not in (1, 2):
            logger.error(
                'A sound can use 1 (mono) or 2 (stereo) channels. '
                f'Provided {channels}.')
            raise ValueError

        # [:, 0] for left and [:, 1] for right
        self.signal = np.zeros(shape=(self.t.size, channels))

    def play(self, blocking=False):
        """
        Play the sound. This function creates and terminates an audio stream.
        """
        sd.play(self.signal, samplerate=self.fs, mapping=[1, 2])
        if blocking:
            sd.wait()

    def stop(self):
        """
        Stops the sounds played in the background.
        """
        sd.stop()

    def read(self, fname):
        """
        scipy.io.wavfile.read()
        """
        raise NotImplementedError

    def write(self, fname):
        """
        Save a sound signal into a wav file with scipy.io.wavfile.write().

        Parameters
        ----------
        fname : str, path
            Path to the file where the sound signal is saved. The extension
            should be '.wav'.
        """
        wavfile.write(fname, self.fs, self.signal)
