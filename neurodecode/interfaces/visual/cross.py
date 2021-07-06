import cv2

from ._visual import _Visual
from ... import logger


class Cross(_Visual):
    """
    Class to display a cross, e.g. a fixation cross.

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

    def putCross(self, length, thickness, color, position):
        """
        Draw a cross composed of 2 rectangles defined by length and thickness.
        The rectangles are positionned to form a cross by overlapping.

        - Horizontal rectangle
        P1 ---------------
        |                |
        --------------- P2

        - Vertical rectangle
        P1 ---
        |    |
        |    |
        |    |
        |    |
        |    |
        --- P2

        Parameters
        ----------
        length : int
            The number of pixels used to draw the length of the cross.
        thickness : int
            The number of pixels used to draw the thickness of the cross.
        color : str | tuple | list
            The color used to fill the cross. Either a matplotlib color string
            or a (Blue, Green, Red) tuple of int8 set between 0 and 255.
        position : str | tuple | list
            The position of the center of the cross.
            Either the string 'center' or 'centered' to position the cross in
            the center of the window; or a 2-length of positive integer
            sequence defining the position of the center of the cross in the
            window. The position is defined in cv2 coordinates, with (0, 0)
            being the top left corner of the window.
        """
        length = Cross._check_length(length, self.window_size)
        thickness = Cross._check_thickness(thickness, length)
        color = _Visual._check_color(color)
        position = Cross._check_position(
            position, length, self.window_size, self.window_center)

        # Horizontal rectangle
        xP1 = position[0] - length//2
        yP1 = position[1] - thickness//2
        xP2 = xP1 + length
        yP2 = yP1 + thickness
        cv2.rectangle(self._img, (xP1, yP1), (xP2, yP2), color, -1)

        # Vertical rectangle
        xP1 = position[0] - thickness//2
        yP1 = position[1] - length//2
        xP2 = xP1 + thickness
        yP2 = yP1 + length
        cv2.rectangle(self._img, (xP1, yP1), (xP2, yP2), color, -1)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_length(length, window_size):
        """
        Checks that the length is a strictly positive integer shorter than the
        width or the height of the window.
        """
        length = int(length)
        if length <= 0:
            logger.error(
                'The cross length must be a strictly positive integer.')
            raise ValueError
        if any(size < length for size in window_size):
            logger.error(
                'The cross length must be shorter than the width or the height'
                'of the window.')
            raise ValueError

        return length

    @staticmethod
    def _check_thickness(thickness, length):
        """
        Checks that the thickness is a strictly positive integer shorter than
        length.
        """
        thickness = int(thickness)
        if thickness <= 0:
            logger.error(
                'The cross thickness must be a strictly positive integer.')
            raise ValueError

        if length <= thickness:
            logger.error(
                'The cross thickness must be strictly smaller than the cross '
                'length.')
            raise ValueError

        return thickness

    @staticmethod
    def _check_position(position, length, window_size, window_center):
        """
        Checks that the inputted position of the center of the cross allows
        the cross to fit in the window.
        The position is given as (X, Y) in cv2 coordinates, with (0, 0) being
        the top left corner of the window.
        """
        if isinstance(position, str):
            position = position.lower().strip()
            if position not in ['centered', 'center']:
                logger.error(
                    "The attribute position can be set as the string 'center' "
                    f"or 'centered'. Provided '{position}'.")
                raise ValueError

            position = window_center

        position = tuple(position)
        if len(position) != 2:
            logger.error(
                'The cross position must be a 2-length sequence (x, y).')
            raise ValueError

        if position[0] - length//2 < 0:
            logger.error(
                'The cross position does not allow the cross '
                'to fit on the window. Crossing left border.')
            raise ValueError
        elif window_size[0] < position[0] - length//2 + length:
            logger.error(
                'The cross position does not allow the cross '
                'to fit on the window. Crossing right border.')
            raise ValueError
        if position[1] - length//2 < 0:
            logger.error(
                'The cross position does not allow the cross '
                'to fit on the window. Crossing top border.')
            raise ValueError
        elif window_size[1] < position[1] - length//2 + length:
            logger.error(
                'The cross position does not allow the cross '
                'to fit on the window. Crossing bottom border.')
            raise ValueError

        return position
