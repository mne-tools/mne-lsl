from collections.abc import Callable

from ..utils.logs import logger as logger
from ..utils.logs import warn as warn
from .base import BaseStream as BaseStream

CH_NAMES: dict[str, list[str]]

def check_hpi_ch_names(ch_names: list[str], format: str) -> None:
    """Check if the channel names match the MEGIN HPI format.

    Parameters
    ----------
    ch_names : list of str
        The list of channel names to check.
    format : str
        The format of the HPI data, e.g., "megin".
    """

def create_hpi_callback_megin(main_stream: BaseStream) -> Callable:
    """Create a callback function for processing MEGIN HPI data.

    The callback processes HPI data from a ``neuromag2lsl`` HPI stream and updates
    the main stream's ``dev_head_t`` transformation matrix in real-time.

    Parameters
    ----------
    main_stream : BaseStream
        The main MEG stream whose ``dev_head_t`` will be updated.

    Returns
    -------
    callback : Callable
        A callback function that can be added to an HPI stream to automatically
        update the main stream's head position transformation.

    Notes
    -----
    The MEGIN format expects HPI data as a vector of shape (12,) containing
    the 4x4 transformation matrix::

        R11 R12 R13 T1
        R21 R22 R23 T2
        R31 R32 R33 T3
        0   0   0   1
    """
