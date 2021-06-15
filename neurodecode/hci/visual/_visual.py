"""
Created on Tue Jun 15 16:14:39 2021

@author: scheltie
"""
from abc import ABC, abstractmethod

import cv2
import numpy as np
from screeninfo import get_monitors

from ... import logger


class _Visual(ABC):
    """
    Base visual class.
    """

    @abstractmethod
    def __init__(self, window_name='Visual', window_size=None):
        self.window_name = str(window_name)

        # Size attributes
        self._window_size = _Visual._check_window_size(window_size)
        self._window_width = self._window_size[0]
        self._window_height = self._window_size[1]
        self._window_center = (self._window_width//2, self._window_height//2)

        self.img = np.full((self._window_height, self._window_width, 3),
                            fill_value=(0, 0, 0), dtype=np.uint8)

    def show(self, wait=1):
        """
        Show the visual with cv2.imshow() and cv2.waitKey().

        Parameters
        ----------
        wait : int
            Wait timer passed to cv2.waitKey(). The timer is defined in ms.
        """
        cv2.imshow(self.window_name, self.img)
        cv2.waitKey(wait)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_window_size(window_size):
        if window_size is not None:
            window_size = tuple(int(size) for size in window_size)

            if len(window_size) != 2:
                logger.error(
                    'The window size must be either a 2-length sequence '
                    'OR None for automatic selection based on screen size. '
                    f'Provided {len(window_size)}.')
                raise ValueError
            if not all(0 < size for size in window_size):
                logger.error(
                    'The window size must be either a 2-length sequence of '
                    'positive integers OR None for automatic selection based '
                    f'on screen size. Provided {window_size}.')
                raise ValueError
        else:
            try:
                width = min(monitor.width for monitor in get_monitors())
                height = min(monitor.height for monitor in get_monitors())
            except ValueError:
                logger.error(
                    'Automatic window size selection does not work for '
                    'headless systems.')
                raise ValueError
            window_size = (width, height)

        return window_size
