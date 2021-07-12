import copy

import cv2

from ._visual import _Visual
from ... import logger


class FillingBar(_Visual):
    """
    Class to display a centered bar which can fill/unfill along a given axis.
    The filling process starts from the center of the bar and occurs on both
    side simultaneously.

    Parameters
    ----------
    window_name : str
        Name of the window in which the visual is displayed.
    window_size : tuple | list | None
        Either None to automatically select a window size based on the
        available monitors, or a 2-length of positive integer sequence.
    """

    def __init__(self, window_name='Visual', window_size=None):
        super().__init__(window_name, window_size)
        self._backup_img = None

    def putBar(self, length, width, margin, color, fill_color,
               fill_perc=0, axis=0):
        """
        Backup the visual and draw the bar on top.

        Parameters
        ----------
        length : int
            Number of pixels used to draw the length of the bar.
        width : int
            Number of pixels used to draw the width of the bar.
        margin : int
            Margin in pixel between the filling bar and the containing bar.
            The containing bar lengthxwidth is set as:
                (length+margin, width+margin)
        color : str | tuple
            Color used to draw the bar background. Either a matplotlib color
            string or a (Blue, Green, Red) tuple of int8 set between 0 and 255.
        fill_color : str | tuple
            Color used to fill the bar. Either a matplotlib color string or a
            (Blue, Green, Red) tuple of int8 set between 0 and 255.
        fill_perc : float
            Percentage between 0 and 1 of bar filling.
                0 - not filled
                1 - fully filled
            As the bar fills on both side simultaneously, the percentage filled
            is length//2 * fill_perc.
        axis : int | str
            Axis along which the bar is moving:
                0, 'vertical', 'v'      - vertical bar
                1, 'horizontal', 'h'    - horizontal bar
        """
        if self._backup_img is None:
            self._backup_img = copy.deepcopy(self._img)
        else:
            self._reset()

        self._axis = _Visual._check_axis(axis)
        self._length, margin = FillingBar._check_length_margin(
            length, margin, self._axis, self.window_size)
        self._width, margin = FillingBar._check_width_margin(
            width, margin, self._length, self._axis, self.window_size)
        self._margin = margin
        self._color = _Visual._check_color(color)
        self._fill_color = _Visual._check_color(fill_color)
        self._fill_perc = FillingBar._check_fill_perc(fill_perc)

        self._putBar()

    def _putBar(self):
        """
        Draw the bar rectangle and fill rectangle.

        - Axis = 1 - Horizontal bar
        P1 ---------------
        |                |
        --------------- P2

        - Axis = 0 - Vertical bar
        P1 ---
        |    |
        |    |
        |    |
        |    |
        |    |
        --- P2
        """
        # External rectangle to fill
        if self._axis == 0:
            xP1 = self.window_center[0] - self._width//2 - self._margin
            yP1 = self.window_center[1] - self._length//2 - self._margin
            xP2 = xP1 + self._width + 2*self._margin
            yP2 = yP1 + self._length + 2*self._margin
        elif self._axis == 1:
            xP1 = self.window_center[0] - self._length//2 - self._margin
            yP1 = self.window_center[1] - self._width//2 - self._margin
            xP2 = xP1 + self._length + 2*self._margin
            yP2 = yP1 + self._width + 2*self._margin

        cv2.rectangle(self._img, (xP1, yP1), (xP2, yP2), self._color, -1)

        # Internal smaller rectangle filling the external rectangle
        fill_perc = FillingBar._convert_fill_perc_to_pixel(
            self.fill_perc, self._length)
        if fill_perc != 0:
            if self._axis == 0:
                xP1 = self.window_center[0] - self._width//2
                yP1 = self.window_center[1] - fill_perc
                xP2 = xP1 + self._width
                yP2 = yP1 + 2*fill_perc
            elif self._axis == 1:
                xP1 = self.window_center[0] - fill_perc
                yP1 = self.window_center[1] - self._width//2
                xP2 = xP1 + 2*fill_perc
                yP2 = yP1 + self._width

            cv2.rectangle(
                self._img, (xP1, yP1), (xP2, yP2), self._fill_color, -1)

    def _reset(self):
        """
        Reset the visual with the backup, thus removing the bar.
        """
        self._img = copy.deepcopy(self._backup_img)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_length_margin(length, margin, axis, window_size):
        """
        Checks that the length and margin are strictly positive and add up to
        a shorter dimension than the window dimension along the relevant axis.
        """
        length = int(length)
        margin = int(margin)
        if length <= 0:
            logger.error('The length must be a strictly positive integer.')
            raise ValueError
        if margin <= 0:
            logger.error('The margin must be a strictly positive integer.')
            raise ValueError

        if length <= margin:
            logger.error(
                'The margin must be strictly smaller than the length.')
            raise ValueError

        if window_size[(axis + 1) % 2] < length + margin:
            logger.error(
                'The length+margin must be smaller than the window dimension.')
            raise ValueError

        return length, margin

    @staticmethod
    def _check_width_margin(width, margin, length, axis, window_size):
        """
        Checks that the width is strictly positive and shorter than the length,
        and shorter than the window dimension along the relevant axis.
        """
        width = int(width)
        margin = int(margin)
        if width <= 0:
            logger.error('The width must be a strictly positive integer.')
            raise ValueError
        if length < width:
            logger.error('The width must be larger than the length.')
            raise ValueError
        if width <= margin:
            logger.error(
                'The margin must be strictly smaller than the width.')
            raise ValueError
        if window_size[axis] < width + margin:
            logger.error(
                'The width+margin must be smaller than the window dimension.')
            raise ValueError

        return width, margin

    @staticmethod
    def _check_fill_perc(fill_perc):
        """
        Checks that the fill length is provided as percentage between 0 and 1.
        """
        if not isinstance(fill_perc, (float, int)):
            logger.error('The fill length must be provided between 0 and 1.')
            raise TypeError
        if not (0 <= fill_perc <= 1):
            logger.error('The fill length must be provided between 0 and 1.')
            raise ValueError

        return fill_perc

    @staticmethod
    def _convert_fill_perc_to_pixel(fill_perc, length):
        """
        Convert the fill length between 0 and 1 to the fill length in pixel.
            0 - Not filled
            1 - Fully filled
        Expresses the % of length//2 filled.
        """
        return int((length//2) * fill_perc)

    # --------------------------------------------------------------------
    @property
    def length(self):
        """
        Length of the bar in pixel.
        """
        return self._length

    @length.setter
    def length(self, length):
        self._length, _ = FillingBar._check_length_margin(
            length, self._margin, self._axis, self.window_size)
        self._reset()
        self._putBar()

    @property
    def width(self):
        """
        Width of the bar in pixel.
        """
        return self._width

    @width.setter
    def width(self, width):
        self._width, _ = FillingBar._check_width_margin(
            width, self._margin, self._length, self._axis, self.window_size)
        self._reset()
        self._putBar()

    @property
    def margin(self):
        """
        Margin in pixel between the bar and its filled content.
        """
        return self._margin

    @margin.setter
    def margin(self, margin):
        _, margin = FillingBar._check_length_margin(
            self._length, margin, self._axis, self.window_size)
        _, margin = FillingBar._check_width_margin(
            self._width, margin, self._length, self._axis, self.window_size)
        self._margin = margin
        self._reset()
        self._putBar()

    @property
    def color(self):
        """
        Color of the bar background.
        """
        return self._color

    @color.setter
    def color(self, color):
        self._color = _Visual._check_color(color)
        self._reset()
        self._putBar()

    @property
    def fill_color(self):
        """
        Color to fill the bar.
        """
        return self._fill_color

    @fill_color.setter
    def fill_color(self, fill_color):
        self._fill_color = _Visual._check_color(fill_color)
        self._reset()
        self._putBar()

    @property
    def fill_perc(self):
        """
        Length filled in percent between 0 and 1.
        """
        return self._fill_perc

    @fill_perc.setter
    def fill_perc(self, fill_perc):
        self._fill_perc = FillingBar._check_fill_perc(fill_perc)
        self._reset()
        self._putBar()

    @property
    def axis(self):
        """
        Axis on which the bar is moving.
            0 - Vertical bar filling along the vertical axis.
            1 - Horizontal bar filling along the horizontal axis.
        """
        return self._axis

    @axis.setter
    def axis(self, axis):
        self._axis = _Visual._check_axis(axis)
        self._reset()
        self._putBar()
