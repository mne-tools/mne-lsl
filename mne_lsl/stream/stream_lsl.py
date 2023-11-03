from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from mne.utils import check_version

if check_version("mne", "1.5"):
    from mne.io.constants import FIFF
elif check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF
else:
    from mne.io.constants import FIFF

from ..lsl import StreamInlet, resolve_streams
from ..lsl.constants import fmt2numpy
from ..utils._checks import check_type
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ._base import BaseStream

if TYPE_CHECKING:
    from typing import Optional, Sequence, Union

    from mne_lsl.lsl.stream_info import _BaseStreamInfo


@fill_doc
class StreamLSL(BaseStream):
    """Stream object representing a single LSL stream.

    Parameters
    ----------
    %(stream_bufsize)s
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

    def __init__(
        self,
        bufsize: float,
        name: Optional[str] = None,
        stype: Optional[str] = None,
        source_id: Optional[str] = None,
    ):
        super().__init__(bufsize)
        check_type(name, (str, None), "name")
        check_type(stype, (str, None), "stype")
        check_type(source_id, (str, None), "source_id")
        self._name = name
        self._stype = stype
        self._source_id = source_id
        self._reset_variables()

    def __repr__(self):
        """Representation of the instance."""
        try:
            conn = self.connected
        except AssertionError:  # can raise on `assert`, e.g., in mid-disconnect or del
            conn = False
        if conn:
            status = "ON"
            desc = f"{self._name} (source: {self._source_id or 'unknown'})"
        else:
            status = "OFF"
            _name = getattr(self, "_name", None)  # could be del'ed
            _source_id = getattr(self, "_source_id", "")
            if _name is not None:
                desc = f"{_name} (source: {_source_id or 'unknown'})"
            elif _source_id:
                desc = f"(source: {self._source_id}"
            else:
                desc = None
        if desc is None:
            return f"<Stream: {status}>"
        else:
            return f"<Stream: {status} | {desc}>"

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
        super().connect(acquisition_delay)
        # The threadsafe processing flag should not be needed for this class. If it is
        # provided, then it means the user is retrieving and doing something with the
        # inlet in a different thread. This use-case is not supported, and users which
        # needs this level of control should create the inlet themselves.
        if processing_flags is not None and (
            processing_flags == "threadsafe" or "threadsafe" in processing_flags
        ):
            raise ValueError(
                "The 'threadsafe' processing flag should not be provided for an "
                "MNE-LSL Stream. If you require access to the underlying StreamInlet "
                "in a separate thread, please instantiate the StreamInlet directly "
                "from mne_lsl.lsl.StreamInlet."
            )
        if processing_flags == "all":
            processing_flags = ("clocksync", "dejitter", "monotize")
        # resolve and connect to available streams
        sinfos = resolve_streams(timeout, self._name, self._stype, self._source_id)
        if len(sinfos) != 1:
            raise RuntimeError(
                "The provided arguments 'name', 'stype', and 'source_id' do not "
                f"uniquely identify an LSL stream. {len(sinfos)} were found: "
                f"{[(sinfo.name, sinfo.stype, sinfo.source_id) for sinfo in sinfos]}."
            )
        if sinfos[0].dtype == "string":
            raise RuntimeError(
                "The Stream class is designed for numerical types. It does not support "
                "string LSL streams. Please use a mne_lsl.lsl.StreamInlet directly to "
                "interact with this stream."
            )
        # create inlet and retrieve stream info
        self._inlet = StreamInlet(
            sinfos[0],
            max_buffered=ceil(self._bufsize),
            processing_flags=processing_flags,
        )
        self._inlet.open_stream(timeout=timeout)
        self._sinfo = self._inlet.get_sinfo()
        self._name = self._sinfo.name
        self._stype = self._sinfo.stype
        self._source_id = self._sinfo.source_id
        # create MNE info from the LSL stream info returned by an open stream inlet
        self._info = self._sinfo.get_channel_info()
        # initiate time-correction
        tc = self._inlet.time_correction(timeout=timeout)
        logger.info("The estimated timestamp offset is %.2f ms.", tc * 1000)
        # create buffer of shape (n_samples, n_channels) and (n_samples,)
        if self._inlet.sfreq == 0:
            self._buffer = np.zeros(
                (self._bufsize, self._inlet.n_channels),
                dtype=fmt2numpy[self._inlet._dtype],
            )
            self._timestamps = np.zeros(self._bufsize, dtype=np.float64)
        else:
            self._buffer = np.zeros(
                (ceil(self._bufsize * self._inlet.sfreq), self._inlet.n_channels),
                dtype=fmt2numpy[self._inlet._dtype],
            )
            self._timestamps = np.zeros(
                ceil(self._bufsize * self._inlet.sfreq), dtype=np.float64
            )
        self._picks_inlet = np.arange(0, self._inlet.n_channels)
        # define the acquisition thread
        self._create_acquisition_thread(0)

    def disconnect(self) -> None:
        """Disconnect from the LSL stream and interrupt data collection."""
        super().disconnect()
        inlet = self._inlet
        logger.debug("Calling inlet.close_stream() for %s", self)
        self._inlet = None  # prevent _acquire from being called
        del inlet
        self._reset_variables()  # also sets self._inlet = None

    def _acquire(self) -> None:
        """Update function pulling new samples in the buffer at a regular interval."""
        if not getattr(self, "_inlet", None):
            return  # stream disconnected
        if getattr(self, "_interrupt", False):
            return  # stream interrupted (don't continue)
        try:
            # pull data
            data, timestamps = self._inlet.pull_chunk(timeout=0.0)
            if timestamps.size == 0:
                if not self._interrupt:
                    self._create_acquisition_thread(self._acquisition_delay)
                return  # interrupt early

            # process acquisition window
            assert data.ndim == 2 and data.shape[-1] == self._inlet.n_channels, (
                data.shape,
                self._inlet.n_channels,
            )
            data = data[:, self._picks_inlet]  # subselect channels
            if len(self._added_channels) != 0:
                refs = np.zeros(
                    (timestamps.size, len(self._added_channels)), dtype=self.dtype
                )
                data = np.hstack((data, refs), dtype=self.dtype)

            if self.info["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_ON:
                data_ref = data[:, self._ref_channels].mean(axis=1, keepdims=True)
                data[:, self._ref_from] -= data_ref

            # roll and update buffers
            self._buffer = np.roll(self._buffer, -timestamps.size, axis=0)
            self._timestamps = np.roll(self._timestamps, -timestamps.size, axis=0)
            assert self._buffer.ndim == 2
            assert self._buffer.shape[1] == data.shape[1], (
                self._buffer.shape,
                data.shape,
                self._inlet.n_channels,
                self._picks_inlet.size,
            )
            # select the last self._timestamps.size samples from data and timestamps in
            # case more samples than the buffer can hold were retrieved.
            self._buffer[-timestamps.size :, :] = data[-self._timestamps.size :, :]
            self._timestamps[-timestamps.size :] = timestamps[-self._timestamps.size :]
            # update the number of new samples available
            self._n_new_samples += min(timestamps.size, self._timestamps.size)
            if (
                self._timestamps.size < self._n_new_samples
                or self._timestamps.size < timestamps.size
            ):
                logger.info(
                    "The number of new samples exceeds the buffer size. Consider using "
                    "a larger buffer by creating a Stream with a larger 'bufsize' "
                    "argument or consider retrieving new samples more often with "
                    "Stream.get_data()."
                )
        except Exception as error:
            logger.exception(error)
            self._reset_variables()  # disconnects from the stream
            if os.getenv("MNE_LSL_RAISE_STREAM_ERRORS", "false").lower() == "true":
                raise
        else:
            if not self._interrupt:
                self._create_acquisition_thread(self._acquisition_delay)

    def _reset_variables(self) -> None:
        """Reset variables define after connection."""
        super()._reset_variables()
        self._sinfo = None
        self._inlet = None

    # ----------------------------------------------------------------------------------
    @property
    def compensation_grade(self) -> Optional[int]:
        """The current gradient compensation grade.

        :type: :class:`int` | None
        """
        self._check_connected(name="compensation_grade")
        return super().compensation_grade

    # ----------------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        """Connection status of the stream.

        :type: :class:`bool`
        """
        attributes = (
            "_sinfo",
            "_inlet",
        )
        if super().connected:
            # sanity-check
            assert not any(getattr(self, attr, None) is None for attr in attributes)
            return True
        else:
            # sanity-check
            assert all(getattr(self, attr, None) is None for attr in attributes)
            return False

    @property
    def name(self) -> Optional[str]:
        """Name of the LSL stream.

        :type: :class:`str` | None
        """
        return self._name

    @property
    def sinfo(self) -> Optional[_BaseStreamInfo]:
        """StreamInfo of the connected stream.

        :type: :class:`~mne_lsl.lsl.StreamInfo` | None
        """
        return self._sinfo

    @property
    def stype(self) -> Optional[str]:
        """Type of the LSL stream.

        :type: :class:`str` | None
        """
        return self._stype

    @property
    def source_id(self) -> Optional[str]:
        """ID of the source of the LSL stream.

        :type: :class:`str` | None
        """
        return self._source_id
