"""
Auditory Steady State Response Stimuli.
"""

import numpy as np

from ._sound import _Sound
from ... import logger
from ...utils._docs import fill_doc


@fill_doc
class ASSR(_Sound):
    """
    Auditory Steady State Response Stimuli.
    Composed of a carrier frequency 'fc' which is amplitude modulated at 'fm'.

    By default, a 1000 Hz carrier frequency modulated at 40 Hz through
    conventional modulation.

    Parameters
    ----------
    %(audio_volume)s
    frequency_carrier : int
        Carrier frequency in Hz.
    frequency_modulation : int
        Modulatiom frequency in Hz.
    method : str
        Either ``'conventional'`` or ``'dsbsc'``.
        ``'conventional'``:
            Also called 'classical AM', the equation used is:
                ``signal = (1 - M(t)) * cos(2*pi*fc*t)``
                ``M(t) = cos(2*pi*fm*t)``
        ``'dsbsc'``:
            Also called 'double side band suppressed carrier', the equation
            used is:
                ``signal = M(t)*cos(2*pi*fc*t)``
                ``M(t) = sin(2*pi*fm*t)``
    %(audio_sample_rate)s
    %(audio_duration)s
    """

    def __init__(self, volume, frequency_carrier=1000,
                 frequency_modulation=40, method='conventional',
                 sample_rate=44100, duration=1.0):
        self._method = ASSR._check_method(method)
        self.name = f'assr {self._method}'
        self._frequency_carrier = ASSR._check_frequency_carrier(
            frequency_carrier)
        self._frequency_modulation = ASSR._check_frequency_modulation(
            frequency_modulation)
        super().__init__(volume, sample_rate, duration)

    def _set_signal(self):
        """
        Sets the signal to output.
        """
        if self._method == 'conventional':
            assr_amplitude = (1-np.cos(
                2*np.pi*self._frequency_modulation*self._time_arr))
            assr_arr = assr_amplitude * np.cos(
                2*np.pi*self._frequency_carrier*self._time_arr)

            self._signal[:, 0] = assr_arr * self._volume[0] / 100
            if len(self._volume) == 2:
                self._signal[:, 1] = assr_arr * self._volume[1] / 100

        elif self._method == 'dsbsc':
            assr_amplitude = np.sin(
                2*np.pi*self._frequency_modulation*self._time_arr)
            assr_arr = assr_amplitude * np.sin(
                2*np.pi*self._frequency_carrier*self._time_arr)

            self._signal[:, 0] = assr_arr * self._volume[0] / 100
            if len(self._volume) == 2:
                self._signal[:, 1] = assr_arr * self._volume[1] / 1000

    # --------------------------------------------------------------------
    @staticmethod
    def _check_method(method):
        """
        Checks that the method is either ``'conventional'`` or ``'dsbsc'``.
        """
        method = method.lower().strip()
        if method not in ('conventional', 'dsbsc'):
            logger.error(
                "Supported amplitude modulation methods are 'conventional' "
                f"and 'dsbsc'. Provided: '{method}'.")

        return method

    @staticmethod
    def _check_frequency_carrier(frequency_carrier):
        """
        Checks if the carrier frequency is positive.
        """
        frequency_carrier = float(frequency_carrier)
        if frequency_carrier <= 0:
            logger.error(
                'The carrier frequency must be positive. '
                f'Provided {frequency_carrier} Hz.')
            raise ValueError

        return frequency_carrier

    @staticmethod
    def _check_frequency_modulation(frequency_modulation):
        """
        Checks if the modulation frequency is positive.
        """
        frequency_modulation = float(frequency_modulation)
        if frequency_modulation <= 0:
            logger.error(
                'The modulation frequency must be positive. '
                f'Provided {frequency_modulation} Hz.')
            raise ValueError

        return frequency_modulation

    # --------------------------------------------------------------------
    @property
    def frequency_carrier(self):
        """
        Sound's carrier frequency.
        """
        return self._frequency_carrier

    @frequency_carrier.setter
    def frequency_carrier(self, frequency_carrier):
        self._frequency_carrier = ASSR._check_frequency_carrier(
            frequency_carrier)
        self._set_signal()

    @property
    def frequency_modulation(self):
        """
        Sound's modulation frequency.
        """
        return self._frequency_modulation

    @frequency_modulation.setter
    def frequency_modulation(self, frequency_modulation):
        self._frequency_modulation = ASSR._check_frequency_modulation(
            frequency_modulation)
        self._set_signal()

    @property
    def method(self):
        """
        Sound's modulation method.
        """
        return self._method

    @method.setter
    def method(self, method):
        self._method = ASSR._check_method(method)
        self._set_signal()
