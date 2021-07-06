import copy

import cv2


from ._visual import _Visual
from ... import logger


class MovingBar(_Visual):
    """
    Class to display a centered moving bar along the vertical or horizontal
    axis.

    Parameters
    ----------
    window_name : str
        The name of the window in which the visual is displayed.
    window_size : tuple | list | None
        Either None to automatically select a window size based on the
        available monitors, or a 2-length of positive integer sequence.
    """

    def __init__(self, window_name='Visual', window_size=None):
        super().__init__(window_name, window_size)
        self._backup_img = None

    def putBar(self, axis, position, length, width, color):
        """
        Backup the visual and draw the bar on top.

        Parameters
        ----------
        axis : int | str
            The axis along which the bar is moving:
                0 - horizontal bar along vertical axis.
                1 - vertical bar along horizontal axis.
        position : int | float
            The relative position of the bar along the given axis.
            Along the vertical axis:
                -1 corresponds to the top of the window.
                1 corresponds to the bottom of the window.
            Along the horizontal axis:
                -1 corresponds to the left of the window.
                1 corresponds to the right of the window.
            0 corresponds to the center of the window.
        length : int
            The number of pixels used to draw the length of the bar.
        width : int
            The number of pixels used to draw the width of the bar.
        color : str | tuple | list
            The color used to fill the bar. Either a matplotlib color string
            or a (Blue, Green, Red) tuple of int8 set between 0 and 255.
        """
        if self._backup_img is None:
            self._backup_img = copy.deepcopy(self._img)
        else:
            self._reset()

        self._axis = MovingBar._check_axis(axis)
        self._position = MovingBar._check_position(position)
        self._length = MovingBar._check_length(
            length, self._axis, self.window_size)
        self._width = MovingBar._check_width(width, self._length)
        self._color = _Visual._check_color(color)

        self._putBar()

    def _putBar(self):
        """
        Draw the bar rectangle.

        - Horizontal bar
        P1 ---------------
        |                |
        --------------- P2

        - Vertical bar
        P1 ---
        |    |
        |    |
        |    |
        |    |
        |    |
        --- P2
        """
        position = MovingBar._convert_position_to_pixel(
            self._position, self._axis, self.window_size, self.window_center)

        if self._axis == 0:
            xP1 = self.window_center[0] - self._length//2
            yP1 = position - self._width//2
            xP2 = xP1 + self._length
            yP2 = yP1 + self._width
        elif self._axis == 1:
            xP1 = position - self._width//2
            yP1 = self.window_center[1] - self._length//2
            xP2 = xP1 + self._width
            yP2 = yP1 + self._length

        cv2.rectangle(self._img, (xP1, yP1), (xP2, yP2), self._color, -1)

    def _reset(self):
        """
        Reset the visual with the backup, thus removing the bar.
        """
        self._img = copy.deepcopy(self._backup_img)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_axis(axis):
        """
        Checks that the axis is valid and converts it to integer (0, 1).
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

    @staticmethod
    def _check_position(position):
        """
        Checks that the position given is between -1 and 1.
        """
        if isinstance(position, (int, float)):
            if -1 <= position <= 1:
                return position
            else:
                logger.error('The bar position must be between -1 and 1 inc.')
                raise ValueError
        else:
            logger.error(
                'The bar position must be a value between -1 and 1 inc.')
            raise TypeError

    @staticmethod
    def _check_length(length, axis, window_size):
        """
        Checks that the length is strictly positive and shorter than the
        window dimension along the given axis.
        """
        length = int(length)
        if length <= 0:
            logger.error('The length must be a strictly positive integer.')
            raise ValueError
        if window_size[axis] < length:
            logger.error(
                'The length must be smaller than the window dimension.')
            raise ValueError

        return length

    @staticmethod
    def _check_width(width, length):
        """
        Checks that the width is strictly positive and shorter than the length.
        """
        width = int(width)
        if width <= 0:
            logger.error('The width must be a strictly positive integer.')
            raise ValueError
        if length < width:
            logger.error('The width must be larger than the length.')
            raise ValueError

        return width

    @staticmethod
    def _convert_position_to_pixel(position, axis, window_size, window_center):
        """
        Convert the relative position between -1 and 1 to an absolute position
        based on the window_size and window_center.
        """
        # horizontal bar moving up and down
        if axis == 0:
            if position == 0:
                return window_center[1]
            elif -1 <= position < 0:
                # top to center
                return int(window_center[1] * (1-abs(position)))
            elif 0 < position <= 1:
                # center to bottom
                return int(window_center[1] +
                           (window_size[1]-window_center[1])*position)

        # vertical bar moving left and right
        elif axis == 1:
            if position == 0:
                return window_center[0]
            elif -1 <= position < 0:
                # left to center
                return int(window_center[0] * (1-abs(position)))
            elif 0 < position <= 1:
                # center to right
                return int(window_center[0] +
                           (window_size[0]-window_center[0])*position)

    # --------------------------------------------------------------------
    @property
    def axis(self):
        """
        The axis on which the bar is moving.
        0 - Horizontal bar along vertical axis.
        1 - Vertical bar along horizonal axis.
        """
        return self._axis

    @axis.setter
    def axis(self, axis):
        self._axis = MovingBar._check_axis(axis)
        self._reset()
        self._putBar()

    @property
    def position(self):
        """
        The position between -1 and 1 of the bar on the given axis.
        """
        return self._position

    @position.setter
    def position(self, position):
        self._position = MovingBar._check_position(position)
        self._reset()
        self._putBar()

    @property
    def length(self):
        """
        The length of the bar.
        """
        return self._length

    @length.setter
    def length(self, length):
        self._length = MovingBar._check_length(
            length, self._axis, self.window_size)
        self._reset()
        self._putBar()

    @property
    def width(self):
        """
        The width of the bar.
        """
        return self._width

    @width.setter
    def width(self, width):
        self._width = MovingBar._check_width(width, self._length)
        self._reset()
        self._putBar()

    @property
    def color(self):
        """
        The color of the bar.
        """
        return self._color

    @color.setter
    def color(self, color):
        self._color = _Visual._check_color(color)
        self._reset()
        self._putBar()
