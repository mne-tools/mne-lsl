from typing import Optional

from mne import Info

from ..lsl import resolve_streams
from .. import logger
from ..utils._checks import check_type


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
        # TODO

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_data(self, winsize: float):
        pass

    def set_channel_types(self):
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
    def name(self) -> Optional[str]:
        """ID of the source of the LSL stream.

        :type: `str` | None
        """
        return self._source_id
