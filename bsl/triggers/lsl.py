"""Trigger using an LSL outlet."""

import numpy as np

from ..lsl import StreamInfo, StreamOutlet
from ..utils._checks import _check_type
from ..utils._docs import copy_doc, fill_doc
from ._base import BaseTrigger


@fill_doc
class LSLTrigger(BaseTrigger):
    """Trigger sending values on an LSL outlet.

    Make sure you are recording the stream created by the
    `~bsl.triggers.LSLTrigger` alongside your data. e.g. if you use
    LabRecorder, update the stream list after creating the
    `~bsl.triggers.LSLTrigger`.

    .. warning::

        Make sure to close the `~bsl.lsl.StreamOutlet` by calling the
        `~bsl.triggers.LSLTrigger.close` method or by deleting the trigger
        after use.

    Parameters
    ----------
    name : str
        Name of the trigger displayed on the LSL network.

    Notes
    -----
    The `~bsl.lsl.StreamOutlet` created has the following properties:
    * Type: ``"Markers"``
    * Number of channels: 1
    * Sampling rate: Irregular
    * Data type: ``np.int8``
    * Source ID: ``"BSL-name"``

    The values sent must be in the range of strictly positive integers defined
    by ``np.int8``, 1 to 127 included.
    """

    def __init__(self, name: str):
        _check_type(name, (str,), "name")
        self._name = name
        # create outlet
        self._sinfo = StreamInfo(
            name=name,
            stype="Markers",
            n_channels=1,
            sfreq=0.0,
            dtype="int8",
            source_id=f"BSL-{name}",
        )
        self._outlet = StreamOutlet(self._sinfo, max_buffered=1)

    @copy_doc(BaseTrigger.signal)
    def signal(self, value: int) -> None:
        super().signal(value)
        self._outlet.push_sample(np.int8(value))

    def close(self) -> None:
        """Close the LSL outlet."""
        try:
            del self._outlet
        except Exception:
            pass

    def __del__(self):  # noqa: D105
        self.close()

    # --------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Name of the trigger displayed on the LSL network.

        :type: str
        """
        return self._name

    @property
    def sinfo(self) -> StreamInfo:
        """Description of the trigger outlet.

        :type: `~bsl.lsl.StreamInfo`
        """
        return self._sinfo

    @property
    def outlet(self) -> StreamOutlet:
        """Trigger outlet.

        :type: `~bsl.lsl.StreamOutlet`
        """
        return self._outlet
