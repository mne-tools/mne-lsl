from ._visual import _Visual
from ... import logger
from ...utils._docs import fill_doc

import cv2


@fill_doc
class Text(_Visual):
    """
    Class to display a text.

    Parameters
    ----------
    %(visual_window_name)s
    %(visual_window_size)s
    """

    def __init__(self, window_name='Visual', window_size=None):
        super().__init__(window_name, window_size)

    @fill_doc
    def putText(self, text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2,
                color='white', thickness=2, position='centered'):
        """
        Method adding text to the visual.

        Parameters
        ----------
        text : str
            Text to display.
        fontFace : cv2 font
            Font to use to write the text.
        fontScale : int
            Scale of the font.
        %(visual_color_text)s
        thickness : int
            Text line thickness in pixel.
        %(visual_position_text)s
        """
        if text != '':
            textWidth, textHeight = cv2.getTextSize(
                text, fontFace, fontScale, thickness)[0]
            position = Text._check_position(
                position, textWidth, textHeight,
                self.window_size, self.window_center)
            color = _Visual._check_color(color)

            cv2.putText(self._img, text, position, fontFace, fontScale, color,
                        thickness=thickness, lineType=cv2.LINE_AA)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_position(position, textWidth, textHeight,
                        window_size, window_center):
        """
        Checks that the inputted position of the bottom left corner of the
        text allows the text to fit in the window.
        The position is given as ``(X, Y)`` in opencv coordinates, with
        ``(0, 0)`` being the top left corner of the window.
        """
        if isinstance(position, str):
            position = position.lower().strip()
            if position not in ['centered', 'center']:
                logger.error(
                    "The attribute position can be set as the string 'center' "
                    f"or 'centered' only. Provided '{position}'.")
                raise ValueError

            position = (window_center[0] - textWidth//2,
                        window_center[1] + textHeight//2)

        position = tuple(position)
        if len(position) != 2:
            logger.error(
                'The text position must be a 2-length sequence (x, y).')
            raise ValueError

        if position[0] < 0:
            logger.error(
                'The text position along the x-axis must be positive.')
            raise ValueError

        elif window_size[0] < position[0] + textWidth:
            logger.error(
                'The text does not fit in the window. Crossing right border.')
            raise ValueError

        if position[1] - textHeight < 0:
            logger.error(
                'The text does not fit in the window. Crossing top border.')
            raise ValueError

        elif window_size[1] < position[1]:
            logger.error(
                'The text position along thje y-axis must be smaller or equal '
                'to the window height.')
            raise ValueError

        return position
