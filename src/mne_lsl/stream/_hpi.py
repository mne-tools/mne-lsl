from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mne import Transform

from ..utils.logs import logger, warn

if TYPE_CHECKING:
    from collections.abc import Callable

    from mne import Info
    from numpy.typing import NDArray

    from .base import BaseStream


def check_hpi_ch_names_megin(ch_names: list[str]) -> None:
    """Check if the channel names match the MEGIN HPI format.

    Parameters
    ----------
    ch_names : list of str
        The list of channel names to check.
    """
    expected_names = [
        "R11",
        "R12",
        "R13",
        "R21",
        "R22",
        "R23",
        "R31",
        "R32",
        "R33",
        "T1",
        "T2",
        "T3",
    ]
    if ch_names != expected_names:
        raise RuntimeError(
            f"Expected HPI channel names for MEGIN format, got {ch_names}."
        )


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

    def hpi_callback(
        data: NDArray[np.floating], timestamps: NDArray[np.float64], info: Info
    ) -> tuple[NDArray[np.floating], NDArray[np.float64]]:
        """Process HPI data and update main stream's dev_head_t.

        Parameters
        ----------
        data : array of shape (n_times, n_channels)
            HPI data from the stream buffer.
        timestamps : array of shape (n_times,)
            Timestamps corresponding to the HPI data.
        info : Info
            Info object from the HPI stream.

        Returns
        -------
        data : array of shape (n_times, n_channels)
            Unmodified HPI data.
        timestamps : array of shape (n_times,)
            Unmodified timestamps.
        """
        assert data.size != 0  # sanity-check

        # Get the latest HPI measurement (most recent sample)
        hpi_data = data[-1, :].squeeze()  # Shape: (12,)

        if hpi_data.shape[0] != 12:
            warn(
                f"Expected 12 HPI values for MEGIN format, got {hpi_data.shape[0]}. "
                "Skipping dev_head_t update."
            )
            return data, timestamps

        # Reconstruct 4x4 transformation matrix from 12-element vector
        # Format: R11, R12, R13, R21, R22, R23, R31, R32, R33, T1, T2, T3
        r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz = hpi_data

        trans_matrix = np.array(
            [
                [r11, r12, r13, tx],
                [r21, r22, r23, ty],
                [r31, r32, r33, tz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Create MNE Transform object (MEG device to head coordinate transformation)
        transform = Transform("meg", "head", trans_matrix)

        # Update main stream's dev_head_t safely
        try:
            with main_stream._interrupt_acquisition():
                with main_stream._info._unlock(
                    update_redundant=False, check_after=False
                ):
                    main_stream._info["dev_head_t"] = transform

            logger.debug(
                "Updated dev_head_t from HPI stream at timestamp %.3f", timestamps[-1]
            )

        except Exception as exc:
            logger.error("Failed to update dev_head_t from HPI data: %s.", exc)

        return data, timestamps

    return hpi_callback
