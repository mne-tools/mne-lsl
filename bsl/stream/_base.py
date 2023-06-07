# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from math import ceil
from threading import Timer
from typing import TYPE_CHECKING

import numpy as np

from .. import logger
from ..lsl import StreamInlet, resolve_streams
from ..lsl.constants import fmt2numpy
from ..utils._checks import check_type
from ..utils.meas_info import create_info

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union

    from numpy.typing import NDarray


class BaseStream:
    def __init__(
        self,
        bufsize: float,
        name: Optional[str] = None,
        stype: Optional[str] = None,
        source_id: Optional[str] = None,
    ):
        check_type(bufsize, ("numeric",), "bufsize")
        if 0 <= bufsize:
            raise ValueError(
                "The buffer size 'bufsize' must be a strictly positive number. "
                f"{bufsize} is invalid."
            )
        check_type(name, (str, None), "name")
        check_type(stype, (str, None), "stype")
        check_type(source_id, (str, None), "source_id")
        self._name = name
        self._stype = stype
        self._source_id = source_id
        self._bufsize = bufsize

        # -- variables defined after resolution ----------------------------------------
        self._sinfo = None
        self._info = None

        # -- variables defined after connection ----------------------------------------
        self._inlet = None
        # The buffer shape is similar to a pull_sample/pull_chunk from an inlet:
        # (n_samples, n_channels). New samples are added to the right of the buffer
        # while old samples are removed from the left of the buffer.
        self._buffer = None
        # picks defines the selected channels and their order
        self._picks = None

    def resolve(self, timeout: float = 10) -> None:
        sinfos = resolve_streams(timeout, self._name, self._stype, self._source_id)
        if len(sinfos) != 1:
            raise RuntimeError(
                "The provided arguments 'name', 'stype', and 'source_id' do not "
                f"uniquely identify an LSL stream. {len(sinfos)} were found: "
                f"{[(sinfo.name, sinfo.stype, sinfo.source_id) for sinfo in sinfos]}."
            )
        self._sinfo = sinfos[0]

        # create MNE info from the LSL stream info
        self._info = create_info(
            self._sinfo.n_channels,
            self._sinfo.sfreq,
            self._sinfo.stype,
            self._sinfo,
        )

    def connect(
        self,
        processing_flags: Optional[Union[str, List[str]]] = None,
        timeout: Optional[float] = 10,
        ufreq: float = 5,
    ) -> None:
        """Connect to the LSL stream and initiate data collection in the buffer.

        Parameters
        ----------
        processing_flags : list of str | ``'all'`` | None
            Set the post-processing options. By default, post-processing is disabled.
            Any combination of the processing flags is valid. The available flags are:

            * ``'clocksync'``: Automatic clock synchronization, equivalent to
              manually adding the estimated `~bsl.lsl.StreamInlet.time_correction`.
            * ``'dejitter'``: Remove jitter on the received timestamps with a
              smoothing algorithm.
            * ``'monotize'``: Force the timestamps to be monotically ascending.
              This option should not be enable if ``'dejitter'`` is not enabled.
        timeout : float | None
            Optional timeout (in seconds) of the operation. ``None`` disables the
            timeout.
        ufreq : float
            Update frequency (Hz) at which chunks of data are pulled from the
            `~bsl.lsl.StreamInlet`.
        """
        # The threadsafe processing flag should not be needed for this class. If it is
        # provided, then it means the user is retrieving and doing something with the
        # inlet in a different thread. This use-case is not supported, and users which
        # needs this level of control should create the inlet themselves.
        if processing_flags == "threadsafe" or "threadsafe" in processing_flags:
            raise ValueError(
                "The 'threadsafe' processing flag should not be provided for a BSL "
                "Stream. If you require access to the underlying StreamInlet in a "
                "separate thread, please instantiate the StreamInlet directly from "
                "bsl.lsl.StreamInlet."
            )
        check_type(ufreq, ("numeric",), "ufreq")
        if ufreq <= 0:
            raise ValueError(
                "The update frequency 'ufreq' must be a strictly positive number "
                "defining the frequency at which new samples are acquired in Hz. For "
                "instance, 5 Hz corresponds to a pull every 200 ms. The provided "
                f"{ufreq} is invalid."
            )
        self._inlet = StreamInlet(self._sinfo, processing_flags=processing_flags)
        self._inlet.open_stream(timeout=timeout)
        # initiate time-correction
        tc = self._inlet.time_correction(timeout=timeout)
        logger.info("The estimated timestamp offset is %.2f seconds.", tc)

        # create buffer of shape (n_samples, n_channels) and (n_samples,)
        self._buffer = np.zeros(
            ceil(self._bufsize * self._inlet.sfreq),
            self._inlet.n_channels,
            dtype=fmt2numpy[self._inlet._dtype],
        )
        self._timestamps = np.zeros(
            ceil(self._bufsize * self._inlet.sfreq), dtype=np.float64
        )
        self._picks = np.arange(0, self._inlet.n_channels)

        # define the acquisition thread
        self._update_delay = 1. / ufreq
        self._update_thread = Timer(1 / self._update_delay, self._update)
        self._update_thread.start()

    def disconnect(self):
        pass

    def _update(self) -> None:
        """Update function pulling new samples in the buffer at a regular interval."""
        data, timestamps = self._inlet.pull_chunk(timeout=0.0)
        if timestamps.size != 0:
            self._buffer = np.roll(self._buffer, -data.shape[0], axis=0)
            self._timestamps = np.roll(self._timestamps, -timestamps.size, axis=0)
            self._buffer[-data.shape[0] :, :] = data
            self._timestamps[-timestamps.size:] = timestamps

        # recreate the timer thread as it is one-call only
        self._update_thread = Timer(self._update_delay, self._update)
        self._update_thread.start()

    def get_data(self, winsize: float) -> Tuple[NDarray[float], NDarray[float]]:
        assert 0 <= winsize, "The window size must be a strictly positive number."
        n_samples = ceil(winsize * self._inlet.sfreq)
        return self._buffer[-n_samples:, self._picks], self._timestamps[-n_samples:]

    def set_channel_types(self):
        pass

    def set_channel_units(self):
        pass

    def rename_channels(self):
        pass

    def reorder_channels(self):
        pass

    def set_montage(self):
        pass

    def pick(self):
        pass

    def drop_channels(self):
        pass

    # ----------------------------------------------------------------------------------
    @property
    def name(self) -> Optional[str]:
        """Name of the LSL stream.

        :type: `str` | None
        """
        return self._name

    @property
    def stype(self) -> Optional[str]:
        """Type of the LSL stream.

        :type: `str` | None
        """
        return self._stype

    @property
    def source_id(self) -> Optional[str]:
        """ID of the source of the LSL stream.

        :type: `str` | None
        """
        return self._source_id
