"""Trigger using an LSL outlet."""

from ..externals import pylsl
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

    Make sure to close the LSL outlet by calling the
    `~bsl.triggers.LSLTrigger.close` method or by deleting the trigger after
    use.

    Parameters
    ----------
    name : str
        Name of the LSL outlet.
    %(trigger_verbose)s
    """

    def __init__(self, name: str, *, verbose: bool = True):
        super().__init__(verbose)
        _check_type(name, (str,), "name")
        self._name = name
        # create outlet
        self._sinfo = pylsl.StreamInfo(
            name=name,
            type="Markers",
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_int16,
            source_id=name,
        )
        self._outlet = pylsl.StreamOutlet(self._sinfo)

    @copy_doc(BaseTrigger.signal)
    def signal(self, value: int) -> None:
        _check_type(value, ("int",), item_name="value")
        self._set_data(value)
        super().signal(value)

    @copy_doc(BaseTrigger._set_data)
    def _set_data(self, value: int) -> None:
        super()._set_data(value)
        self._outlet.push_sample([value])

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
        """LSL outlet name.

        :type: str
        """
        return self._name

    @property
    def sinfo(self) -> pylsl.StreamInfo:
        """LSL stream info.

        :type: ``pylsl.StreamInfo``
        """
        return self._sinfo

    @property
    def outlet(self) -> pylsl.StreamOutlet:
        """LSL stream outlet.

        :type: ``pylsl.StreamOutlet``
        """
        return self._outlet
