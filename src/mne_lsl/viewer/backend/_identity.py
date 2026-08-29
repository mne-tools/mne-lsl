"""Stream identity and the descriptor returned by discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._config import identity_text

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


@dataclass(frozen=True, slots=True)
class StreamIdentity:
    """Exact identity of a stream, ``(name, stype, source_id)``.

    A stream is identified by the full tuple and never by its name alone: two streams
    may share a name and differ by type or source ID.

    Parameters
    ----------
    name : str
        Stream name.
    stype : str
        Stream type, e.g. ``'eeg'``.
    source_id : str
        Source ID of the stream.
    """

    name: str
    stype: str
    source_id: str

    def as_tuple(self) -> tuple[str, str, str]:
        """Return the identity as a plain ``(name, stype, source_id)`` tuple."""
        return (self.name, self.stype, self.source_id)


@dataclass(frozen=True, slots=True)
class StreamDescriptor:
    """Description of a stream present on the network, as returned by discovery.

    Built from a :class:`~mne_lsl.lsl.StreamInfo` without opening an inlet, thus the
    channel names are not part of it: they require a connection, see
    :func:`~mne_lsl.viewer.backend.probe_channels`.

    Parameters
    ----------
    identity : StreamIdentity
        Exact identity of the stream.
    n_channels : int
        Number of channels.
    sfreq : float
        Nominal sampling rate, ``0`` for an irregularly sampled (event) stream.
    hostname : str
        Host on which the outlet runs.
    dtype : str | DTypeLike
        Channel format of the stream.
    uid : str
        Identifier of the *outlet instance*, not of the stream: re-instantiating an
        outlet under the same identity yields a different ``uid``. A channel set cached
        against ``(identity, uid)`` is therefore still valid, while a changed ``uid``
        proves nothing about the channels and forces a fresh probe.

    Notes
    -----
    ``uid`` is deliberately not part of :class:`StreamIdentity`: an identity is the
    ``(name, stype, source_id)`` triple, and folding the instance identifier into it
    would make every reconnection of the same stream a different stream.

    Frozen, and built exclusively of plain Python and numpy scalar types, because a
    descriptor is what crosses the worker/GUI thread boundary in place of the
    :class:`~mne_lsl.lsl.StreamInfo` it was read from: that object's ``__del__``
    destroys the native stream info, so it must never be stored nor handed to another
    thread. Freezing makes that ownership rule structural instead of conventional, and
    makes a descriptor hashable, which is what the identity de-duplication of
    :meth:`~mne_lsl.viewer._window.ViewerWindow.open_streams` relies on.
    """

    identity: StreamIdentity
    n_channels: int
    sfreq: float
    hostname: str
    dtype: str | DTypeLike
    uid: str


@dataclass(frozen=True, slots=True)
class StreamSignature:
    """Everything an automatic resume must find unchanged on a stream which came back.

    Parameters
    ----------
    identity : StreamIdentity
        Exact identity of the stream.
    sfreq : float
        Nominal sampling rate, ``0`` for an irregularly sampled (event) stream.
    dtype : str
        Channel format, as a string, and both sides must come from
        ``str(BaseStream.dtype)``: that is ``'float32'``, while ``str()`` of the numpy
        *type object* a :class:`~mne_lsl.viewer.backend.StreamDescriptor` carries is
        ``"<class 'numpy.float32'>"``. Building one side from a descriptor therefore
        makes the comparison unequal forever, i.e. makes the document permanently refuse
        its own stream; comparing a type object against a :class:`numpy.dtype` without
        the ``str`` is silently always unequal for the same reason.
    ch_names : tuple of str
        Channel names in acquisition order, as they were recorded on the wire.

    Notes
    -----
    ``uid`` and ``hostname`` are deliberately absent, unlike in
    :class:`~mne_lsl.viewer.backend.StreamDescriptor`. ``uid`` identifies the outlet
    *instance*, so a restarted source always publishes a new one and comparing it would
    make every recovery a refusal.
    """

    identity: StreamIdentity
    sfreq: float
    dtype: str
    ch_names: tuple[str, ...]


def signature_mismatch(
    expected: StreamSignature, actual: StreamSignature
) -> str | None:
    """Return why ``actual`` may not silently replace ``expected``, or ``None``.

    Parameters
    ----------
    expected : StreamSignature
        Signature recorded while the document was live.
    actual : StreamSignature
        Signature of the stream which answered the same identity.

    Returns
    -------
    reason : str | None
        A one-line reason, shown to the user, or ``None`` when the stream may be resumed
        into the existing display.

    Notes
    -----
    This is not the rule which decides whether a *saved configuration* may be opened,
    :func:`~mne_lsl.viewer.backend.evaluate_state`, and the two must never be unified.
    That one answers "may the user open this workspace at all" and is deliberately
    friendly: it matches channel names as a subset, tolerates extra channels and ignores
    order, the sampling rate and the format, because a configuration describes a
    *desired* workspace whose extra channels are appended in acquisition order. This one
    answers "may I keep drawing into an existing, already index-bound display", where
    any tolerance is a silent mis-mapping: every piece of stream-side state the viewer
    sets is an integer index, so one extra channel shifts all the following ones and the
    operator then watches one channel's samples under another channel's label, with
    nothing on screen to say so.

    The sampling rate is compared with ``!=`` on floats deliberately: both sides are
    read from the same XML field by the same reader, so equality is exact. It is
    rendered with ``:.10g`` and not with ``:g``, whose 6 significant digits turn a real
    change into the refusal *"the sampling rate changed from 1000 Hz to 1000 Hz"*.

    The impostor defence is the **ordered channel-name comparison**, which runs whatever
    the identity looks like. Nothing here requires a stream to publish a non-empty
    ``name``, ``stype`` or ``source_id``, and requiring one would buy nothing: measured,
    :func:`~mne_lsl.lsl.resolve_streams` short-circuits on the first answer, so two
    outlets publishing an identical full triple hand the connection to whichever answers
    first -- a non-empty ``source_id`` does not make ``connect()`` refuse the duplicate.
    It would, on the other hand, make a document refuse **itself**: a stream publishing
    ``source_id=''`` is legal LSL, discoverable, connectable and drawable, and
    :class:`~mne_lsl.player.PlayerLSL` publishes an empty ``stype`` for any file whose
    channels are not all of one type, i.e. for most real recordings.
    """
    if actual.identity != expected.identity:
        return (
            "another stream answered the identity "
            f"{identity_text(expected.identity.as_tuple())}"
        )
    if actual.sfreq != expected.sfreq:
        return (
            f"the sampling rate changed from {expected.sfreq:.10g} Hz to "
            f"{actual.sfreq:.10g} Hz"
        )
    if actual.dtype != expected.dtype:
        return f"the channel format changed from {expected.dtype} to {actual.dtype}"
    if len(actual.ch_names) != len(expected.ch_names):
        return (
            f"the channel count changed from {len(expected.ch_names)} to "
            f"{len(actual.ch_names)}"
        )
    names = zip(expected.ch_names, actual.ch_names, strict=True)  # lengths just checked
    for k, (name_e, name_a) in enumerate(names):
        if name_e != name_a:
            return f"the channels changed: {k} is now {name_a} and was {name_e}"
    return None
