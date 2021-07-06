from abc import ABC, abstractmethod

import cv2
import numpy as np
from matplotlib import colors
from screeninfo import get_monitors

from ... import logger


class _Visual(ABC):
    """
    Base visual class.

    Parameters
    ----------
    window_name : str
        The name of the window in which the visual is displayed.
    window_size : tuple | list | None
        Either None to automatically select a window size based on the
        available monitors, or a 2-length of positive integer sequence.
    """

    @abstractmethod
    def __init__(self, window_name='Visual', window_size=None):
        self._window_name = str(window_name)

        # Size attributes
        self._window_size = _Visual._check_window_size(window_size)
        self._window_width = self._window_size[0]
        self._window_height = self._window_size[1]
        self._window_center = (self._window_width//2, self._window_height//2)

        # Default black background
        self._img = np.full(
            (self._window_height, self._window_width, 3),
            fill_value=(0, 0, 0), dtype=np.uint8)

    def show(self, wait=1):
        """
        Show the visual with cv2.imshow() and cv2.waitKey().

        Parameters
        ----------
        wait : int
            Wait timer passed to cv2.waitKey(). The timer is defined in ms.
        """
        cv2.imshow(self._window_name, self._img)
        cv2.waitKey(wait)

    def close(self):
        """
        Close the visual.
        """
        cv2.destroyWindow(self._window_name)

    def draw_background_uniform(self, background_color):
        """
        Draw a uniform single color background.
        """
        background_color = _Visual._check_color(background_color)
        self._img = np.full(
            (self._window_height, self._window_width, 3),
            fill_value=background_color, dtype=np.uint8)

    def draw_background_stripes(self, stripes_colors, axis=0):
        """
        Draw a multi-color background composed of stripes along the given axis.
        """
        stripes_colors = _Visual._check_color(stripes_colors)
        if axis == 0:
            stripes_size = self._window_height // len(stripes_colors)
            extra = self._window_height % len(stripes_colors)
        elif axis == 1:
            stripes_size = self._window_width // len(stripes_colors)
            extra = self._window_width % len(stripes_colors)
        else:
            logger.error(
                'Axis must be 0 (horizontal stripes) or 1 (vertical stripes).')
            raise ValueError

        for k, color in enumerate(stripes_colors):
            slc = [slice(None)] * 2
            slc[axis] = slice(k*stripes_size, (k+1)*stripes_size)
            stripes_shape = [self._window_height, self._window_width, 3]
            stripes_shape[axis] = stripes_size

            self._img[tuple(slc)] = np.full(
                tuple(stripes_shape), fill_value=color, dtype=np.uint8)

        if extra != 0:
            slc[axis] = slice(len(stripes_colors)*stripes_size,
                              len(stripes_colors)*stripes_size + extra)
            extra_shape = [self._window_height, self._window_width, 3]
            extra_shape[axis] = extra
            self._img[tuple(slc)] = np.full(
                tuple(extra_shape),
                fill_value=stripes_colors[-1], dtype=np.uint8)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_window_size(window_size):
        """
        Checks if the window size is valid or set it as the minimum
        (width, height) supported by any connected monitor.
        """
        if window_size is not None:
            window_size = tuple(int(size) for size in window_size)

            if len(window_size) != 2:
                logger.error(
                    'The window size must be either a 2-length sequence '
                    'OR None for automatic selection based on screen size. '
                    f'Provided {len(window_size)}.')
                raise ValueError
            if not all(size > 0 for size in window_size):
                logger.error(
                    'The window size must be either a 2-length sequence of '
                    'positive integers OR None for automatic selection based '
                    f'on screen size. Provided {window_size}.')
                raise ValueError
        else:
            try:
                width = min(monitor.width for monitor in get_monitors())
                height = min(monitor.height for monitor in get_monitors())
            except ValueError as headless:
                logger.error(
                    'Automatic window size selection does not work for '
                    'headless systems.')
                raise ValueError from headless
            window_size = (width, height)

        return window_size

    @staticmethod
    def _check_color(color):
        """
        Checks if a color or a list of colors is valid and converts it to BGR.
        """
        if isinstance(color, str):
            r, g, b, _ = colors.to_rgba(color)
            bgr_color = [int(c*255) for c in (b, g, r)]

        elif isinstance(color, (tuple, list)):
            bgr_color = list()
            for col in color:
                if isinstance(col, str):
                    try:
                        r, g, b, _ = colors.to_rgba(col)
                    except ValueError:
                        logger.warning(
                            f"Color '{col}' is not supported. Skipping.")
                    bgr_color.append([int(c*255) for c in (b, g, r)])
                elif isinstance(col, (list, tuple)) and len(col) == 3 and \
                        all(0 <= c <= 255 for c in col):
                    bgr_color.append(col)
                else:
                    logger.warning(
                        f"Color '{col}' is not supported. Skipping.")

            if len(bgr_color) == 0:
                logger.error(
                    'None of the color provided was supported.')
                raise ValueError

        else:
            logger.error('Unrecognized color format.')
            raise TypeError

        return bgr_color

    # --------------------------------------------------------------------
    @property
    def window_name(self):
        """
        The window name.
        """
        return self._window_name

    @property
    def window_size(self):
        """
        The window size (width x height).
        """
        return self._window_size

    @property
    def img(self):
        """
        The image array.
        """
        return self._img
