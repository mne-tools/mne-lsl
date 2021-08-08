from abc import ABC, abstractmethod

import numpy as np

from ... import logger
from ...utils._imports import import_optional_dependency

colors = import_optional_dependency(
    "matplotlib.colors", extra="Install matplotlib for visual(s) support.")
cv2 = import_optional_dependency(
    "cv2", extra="Install matplotlib for visual(s) support.")
screeninfo = import_optional_dependency(
    "screeninfo", extra="Install screeninfo for visual(s) support.")


class _Visual(ABC):
    """
    Base visual class.

    Parameters
    ----------
    window_name : str
        Name of the window in which the visual is displayed.
    window_size : tuple | list | None
        Either None to automatically select a window size based on the
        available monitors, or a 2-length of positive integer sequence, as
        (width, height).
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
        self._background = [0, 0, 0]

    def show(self, wait=1):
        """
        Show the visual with cv2.imshow() and cv2.waitKey().

        Parameters
        ----------
        wait : int
            Wait timer passed to cv2.waitKey() [ms].
        """
        cv2.imshow(self._window_name, self._img)
        cv2.waitKey(wait)

    def close(self):
        """
        Close the visual.
        """
        cv2.destroyWindow(self._window_name)

    def draw_background(self, color):
        """
        Draw a uniform single color background.

        Parameters
        ----------
        color : str | tuple | list
            Color of the background as a matplotlib string or a (B, G, R)
            tuple.
        """
        color = _Visual._check_color(color)
        self._img = np.full(
            (self._window_height, self._window_width, 3),
            fill_value=color, dtype=np.uint8)
        self._background = color

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
                width = min(
                    monitor.width for monitor in screeninfo.get_monitors())
                height = min(
                    monitor.height for monitor in screeninfo.get_monitors())
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
        Checks if a color is valid and converts it to BGR.
        """
        if isinstance(color, str):
            r, g, b, _ = colors.to_rgba(color)
            bgr_color = [int(c*255) for c in (b, g, r)]

        elif isinstance(color, (tuple, list)) and len(color) == 3:
            if not all(isinstance(c, (float, int)) for c in color):
                logger.error(
                    'BGR color tuple must be provided as a 3-length sequence '
                    'of integers between 0 and 255.')
                raise TypeError
            if not all(0 <= c <= 255 for c in color):
                logger.error(
                    'BGR color tuple must be provided as a 3-length sequence '
                    'of integers between 0 and 255.')
                raise ValueError

            bgr_color = [int(c) for c in color]

        else:
            logger.error('Unrecognized color format.')
            raise TypeError

        return bgr_color

    @staticmethod
    def _check_axis(axis):
        """
        Checks that the axis is valid and converts it to integer (0, 1).
            0 - Vertical
            1 - Horizontal
        """
        if isinstance(axis, str):
            axis = axis.lower().strip()
            if axis not in ['horizontal', 'h', 'vertical', 'v']:
                logger.error(
                    "The attribute axis can be set as the string "
                    f"'vertical' or 'horizontal'. Provided '{axis}'.")
                raise ValueError
            if axis.startswith('v'):
                axis = 0
            elif axis.startswith('h'):
                axis = 1
        elif isinstance(axis, (bool, int, float)):
            axis = int(axis)
            if axis not in (0, 1):
                logger.error(
                    "The attribute axis can be set as an int, 0 for "
                    f"vertical and 1 for horizontal. Provided {axis}.")
                raise ValueError
        else:
            logger.error('Unrecognized axis.')
            raise TypeError

        return axis

    # --------------------------------------------------------------------
    @property
    def window_name(self):
        """
        Window's name.
        """
        return self._window_name

    @property
    def window_size(self):
        """
        Window's size (width x height).
        """
        return self._window_size

    @property
    def window_center(self):
        """
        Window's center position.
        """
        return self._window_center

    @property
    def img(self):
        """
        Image array.
        """
        return self._img

    @property
    def background(self):
        """
        Background color.
        """
        return self._background

    @background.setter
    def background(self, background):
        self.draw_background(background)
