import cv2

from ._visual import _Visual
from ... import logger


class FillingBar(_Visual):
    def __init__(self, window_name='Visual', window_size=None):
        super().__init__(window_name, window_size)

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

    # --------------------------------------------------------------------
    @property
    def axis(self):
        """
        The axis on which the bar is moving.
        0 - Vertical bar filling along the vertical axis.
        1 - Horizontal bar filling along the horizontal axis.
        """
        return self._axis
