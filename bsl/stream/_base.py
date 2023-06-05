from typing import List, Optional, Union

from mne import Info

from .. import logger
from ..lsl import StreamInlet, resolve_streams
from ..utils._checks import check_type
from ..utils.meas_info import create_info


class BaseStream:
    def __init__(
        self,
        bufsize: float,
        name: Optional[str] = None,
        stype: Optional[str] = None,
        source_id: Optional[str] = None,
    ):
        check_type(name, (str, None), "name")
        check_type(stype, (str, None), "stype")
        check_type(source_id, (str, None), "source_id")
        self._name = name
        self._stype = stype
        self._source_id = source_id

        # variables defined after resolution
        self._sinfo = None
        self._info = None

        # variables defined after connection
        self._inlet = None
        self._connected = False

    def resolve(self, timeout: float = 1.0):
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
            self._sinfo.as_xml,
        )

    def connect(
        self,
        processing_flags: Optional[Union[str, List[str]]] = None,
        timeout: Optional[float] = 10,
    ):
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
        self._inlet = StreamInlet(self._sinfo, processing_flags=processing_flags)
        self._inlet.open_stream(timeout=timeout)
        # initiate time-correction
        tc = self._inlet.time_correction(timeout=timeout)
        logger.info("The estimated timestamp offset is %.2f seconds.", tc)

    def disconnect(self):
        pass

    def get_data(self, winsize: float):
        pass

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
