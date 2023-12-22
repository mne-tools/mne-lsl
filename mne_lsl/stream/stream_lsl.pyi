from typing import Optional, Sequence, Union

from _typeshed import Incomplete

from mne_lsl.lsl.stream_info import _BaseStreamInfo as _BaseStreamInfo

from ..lsl import StreamInlet as StreamInlet
from ..lsl import resolve_streams as resolve_streams
from ..lsl.constants import fmt2numpy as fmt2numpy
from ..utils._checks import check_type as check_type
from ..utils._docs import fill_doc as fill_doc
from ..utils.logs import logger as logger
from ._base import BaseStream as BaseStream

class StreamLSL(BaseStream):
    """Stream object representing a single LSL stream.

    Parameters
    ----------
    bufsize : float | int
        Size of the buffer keeping track of the data received from the stream. If
        the stream sampling rate ``sfreq`` is regular, ``bufsize`` is expressed in
        seconds. The buffer will hold the last ``bufsize * sfreq`` samples (ceiled).
        If the stream sampling rate ``sfreq`` is irregular, ``bufsize`` is
        expressed in samples. The buffer will hold the last ``bufsize`` samples.
    name : str
        Name of the LSL stream.
    stype : str
        Type of the LSL stream.
    source_id : str
        ID of the source of the LSL stream.

    Notes
    -----
    The 3 arguments ``name``, ``stype``, and ``source_id`` must uniquely identify an
    LSL stream. If this is not possible, please resolve the available LSL streams
    with :func:`mne_lsl.lsl.resolve_streams` and create an inlet with
    :class:`~mne_lsl.lsl.StreamInlet`.
    """

    _name: Incomplete
    _stype: Incomplete
    _source_id: Incomplete

    def __init__(
        self,
        bufsize: float,
        name: Optional[str] = None,
        stype: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> None: ...
    def __repr__(self) -> str:
        """Representation of the instance."""
    _inlet: Incomplete
    _sinfo: Incomplete
    _info: Incomplete
    _buffer: Incomplete
    _timestamps: Incomplete
    _picks_inlet: Incomplete

    def connect(
        self,
        acquisition_delay: float = 0.001,
        processing_flags: Optional[Union[str, Sequence[str]]] = None,
        timeout: Optional[float] = 2,
    ) -> None:
        """Connect to the LSL stream and initiate data collection in the buffer.

        Parameters
        ----------
        acquisition_delay : float
            Delay in seconds between 2 acquisition during which chunks of data are
            pulled from the :class:`~mne_lsl.lsl.StreamInlet`.
        processing_flags : list of str | ``'all'`` | None
            Set the post-processing options. By default, post-processing is disabled.
            Any combination of the processing flags is valid. The available flags are:

            * ``'clocksync'``: Automatic clock synchronization, equivalent to
              manually adding the estimated
              :meth:`~mne_lsl.lsl.StreamInlet.time_correction`.
            * ``'dejitter'``: Remove jitter on the received timestamps with a
              smoothing algorithm.
            * ``'monotize'``: Force the timestamps to be monotically ascending.
              This option should not be enable if ``'dejitter'`` is not enabled.
        timeout : float | None
            Optional timeout (in seconds) of the operation. ``None`` disables the
            timeout. The timeout value is applied once to every operation supporting it.

        Notes
        -----
        If all 3 stream identifiers ``name``, ``stype`` and ``source_id`` are left to
        ``None``, resolution of the available streams will require a full ``timeout``,
        blocking the execution until this function returns. If at least one of the 3
        stream identifiers is specified, resolution will stop as soon as one stream
        matching the identifier is found.
        """

    def disconnect(self) -> None:
        """Disconnect from the LSL stream and interrupt data collection."""

    def _acquire(self) -> None:
        """Update function pulling new samples in the buffer at a regular interval."""

    def _reset_variables(self) -> None:
        """Reset variables define after connection."""

    @property
    def compensation_grade(self) -> Optional[int]:
        """The current gradient compensation grade.

        :type: :class:`int` | None
        """

    @property
    def connected(self) -> bool:
        """Connection status of the stream.

        :type: :class:`bool`
        """

    @property
    def name(self) -> Optional[str]:
        """Name of the LSL stream.

        :type: :class:`str` | None
        """

    @property
    def sinfo(self) -> Optional[_BaseStreamInfo]:
        """StreamInfo of the connected stream.

        :type: :class:`~mne_lsl.lsl.StreamInfo` | None
        """

    @property
    def stype(self) -> Optional[str]:
        """Type of the LSL stream.

        :type: :class:`str` | None
        """

    @property
    def source_id(self) -> Optional[str]:
        """ID of the source of the LSL stream.

        :type: :class:`str` | None
        """
