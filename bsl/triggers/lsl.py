"""Trigger using an LSL outlet."""

from ._trigger import _Trigger


@fill_doc
class LSLTrigger(_Trigger):
    """Trigger sending values on an LSL outlet.

    Make sure you are recording the stream created by the
    `~bsl.triggers.LSLTrigger` alongside your data. e.g. if you use
    LabRecorder, update the stream list after creating the
    `~bsl.triggers.LSLTrigger`.

    Parameters
    ----------
    name : str
        Name of the LSL outlet.
    %(trigger_verbose)s
    """

    def __init__(self, name: str, *, verbose: bool = True):
        super().__init__(verbose)
