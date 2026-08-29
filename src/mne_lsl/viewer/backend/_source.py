"""Discovery of the streams on the network and construction of a stream object.

This module is the only place in the viewer which names an LSL-specific object:
everything downstream is typed to :class:`~mne_lsl.stream.BaseStream`, so that a future
protocol gains a source module here and nothing else.

Notes
-----
Every :class:`~mne_lsl.lsl.StreamInfo` this module creates is created, read and
destroyed inside a single function body. One is never stored on an attribute, never
placed in a container which outlives the call, never returned, and never named outside
this module: its ``__del__`` destroys the native stream info, so handing one to another
thread or caching it invites a hard crash. What crosses out is
:class:`~mne_lsl.viewer.backend.StreamDescriptor`, a frozen dataclass of plain values,
and :class:`~mne_lsl.stream.BaseStream`, which has no Qt thread affinity.
"""

from __future__ import annotations

from math import ceil, isfinite
from typing import TYPE_CHECKING

from ...lsl import StreamInlet, resolve_streams
from ...lsl._utils import LostError
from ...stream import StreamLSL
from ._identity import StreamDescriptor, StreamIdentity, StreamSignature

if TYPE_CHECKING:
    from ...lsl.stream_info import _BaseStreamInfo
    from ...stream import BaseStream

# Discovery pass: long enough for the outlets of a local subnet to answer, short enough
# to be re-run on demand. 'resolve_streams' blocks the full timeout when unrestricted,
# thus this is also the cost of one pass.
_RESOLVE_TIMEOUT = 1.0

# Channel probe: liblsl's own timeouts, applied once per operation. No watchdog, as a
# blocking liblsl call cannot be interrupted and a watchdog would be theatre.
_PROBE_TIMEOUT = 2.0

# Buffer headroom over the widest window the display can show, and the floor below which
# a buffer is not worth having. The buffer must hold a full window plus the samples a
# future processing pipeline needs to warm up. No shipped caller reaches the floor: the
# only one passes the widest selectable window, 20 s, and the floor binds below 2.67 s.
_BUFSIZE_HEADROOM = 1.5
_BUFSIZE_MIN = 4.0


def _descriptor(sinfo: _BaseStreamInfo) -> StreamDescriptor:
    """Return the descriptor of ``sinfo``, reading every property once.

    Parameters
    ----------
    sinfo : StreamInfo
        Stream info to read, owned by the caller and destroyed by it.

    Returns
    -------
    descriptor : StreamDescriptor
        A frozen descriptor holding no reference to ``sinfo``.
    """
    return StreamDescriptor(
        identity=StreamIdentity(
            name=sinfo.name, stype=sinfo.stype, source_id=sinfo.source_id
        ),
        n_channels=sinfo.n_channels,
        sfreq=sinfo.sfreq,
        hostname=sinfo.hostname,
        dtype=sinfo.dtype,
        uid=sinfo.uid,
    )


def resolve_descriptors(timeout: float = _RESOLVE_TIMEOUT) -> list[StreamDescriptor]:
    """Return a descriptor for every stream currently present on the network.

    Parameters
    ----------
    timeout : float
        Timeout of the resolution, in seconds. An unrestricted resolution blocks for the
        full duration, thus this must never be called on the GUI thread.

    Returns
    -------
    descriptors : list of StreamDescriptor
        Descriptors of the streams found, regular and irregular alike, sorted by
        identity.

    Notes
    -----
    The sort is not cosmetic: :func:`~mne_lsl.lsl.resolve_streams` de-duplicates through
    a :class:`set` and therefore returns an unordered list, which would reshuffle the
    launcher table between two passes of an unchanged network.

    The resolved stream infos are local to this call and die when it returns, on the
    thread which called it. That is the ownership rule of this module.
    """
    sinfos = resolve_streams(timeout)
    descriptors = [_descriptor(sinfo) for sinfo in sinfos]
    return sorted(descriptors, key=lambda descriptor: descriptor.identity.as_tuple())


def derive_bufsize(window: float) -> float:
    """Return the buffer size covering a display window of ``window`` seconds.

    Parameters
    ----------
    window : float
        Widest time window the display may show, in seconds.

    Returns
    -------
    bufsize : float
        Buffer size in seconds, a whole number, at least ``window``.

    Raises
    ------
    OverflowError
        If ``window`` is finite but above ~1.198e308, where the headroom multiplication
        overflows to infinity and the rounding then refuses it. Not reachable from the
        interface, and recorded because the check below does not cover it.
    TypeError
        If ``window`` is not a real number, e.g. ``'5'`` or ``None``, raised by the
        finiteness check. A :class:`bool` is a real number and is therefore *accepted*,
        unlike in :meth:`~mne_lsl.viewer.display.DisplayControls.set_state`, which
        refuses one explicitly because ``True`` would clamp to a plausible value.
    ValueError
        If ``window`` is not a finite, strictly positive number: a non-positive one
        allocates an empty buffer whose every read returns nothing, and a non-finite one
        raises from inside the buffer allocation of
        :meth:`~mne_lsl.stream.StreamLSL.connect`, after the inlet is already open.

    Notes
    -----
    The buffer must cover the window: a window wider than the buffer is not an error --
    ``get_data`` silently returns the shorter buffer -- and the display then draws over
    part of its time axis for the rest of the session, with nothing in the interface to
    explain it. The size is therefore derived from the widest window the control bar can
    select, not from the current one.

    The result is a whole number of seconds so that the same value is legal for an
    irregularly sampled stream, whose buffer size is a sample count:
    :func:`create_stream` refuses a fractional value there, and one batch of connections
    carries a single buffer size for regular and event streams alike.
    """
    if not isfinite(window) or window <= 0:
        raise ValueError(
            f"The time window must be a finite, strictly positive number of seconds, "
            f"got {window}."
        )
    # ponytail: derived from the widest selectable window, thus every stream pays for a
    # window nobody may ever select -- 31.5 MB per stream at 256 channels and 1024 Hz,
    # plus the same duration again in liblsl's own ring. The upgrade is to teach the
    # display control bar a runtime maximum, then derive from the window actually shown.
    return float(ceil(max(_BUFSIZE_MIN, window * _BUFSIZE_HEADROOM)))


def create_stream(descriptor: StreamDescriptor, bufsize: float) -> BaseStream:
    """Create a disconnected stream for ``descriptor``.

    Parameters
    ----------
    descriptor : StreamDescriptor
        Descriptor of the stream to create, providing the exact identity tuple.
    bufsize : float
        Size of the stream buffer, in seconds.

    Returns
    -------
    stream : BaseStream
        The stream object, not connected yet.

    Raises
    ------
    ValueError
        If ``bufsize`` is not integral while the stream is irregularly sampled.

    Notes
    -----
    The ``bufsize`` check is this module's, not the library's.
    :meth:`~mne_lsl.stream.StreamLSL.connect` validates ``bufsize`` against
    ``sfreq == 0`` only *after* it has created and opened the inlet, and it does not
    reset the stream on the way out: the :class:`ValueError` leaves an object holding a
    live, subscribed inlet while reading as disconnected. It is now recoverable --
    :attr:`~mne_lsl.stream.BaseStream.connected` reads a partial state as ``False``,
    :meth:`~mne_lsl.stream.BaseStream.disconnect` is idempotent and destroys the inlet
    unconditionally, and ``__del__`` therefore closes it -- but only for a caller who
    thinks to disconnect an object whose connection raised. A descriptor already carries
    :attr:`~mne_lsl.viewer.backend.StreamDescriptor.sfreq`, thus the viewer refuses the
    value up front. That matters because
    :meth:`~mne_lsl.viewer.backend.Connector.open` passes one ``bufsize`` to a batch
    holding regular and event streams alike, where a single non-integral value would
    otherwise half-connect every event stream of the configuration.
    """
    identity = descriptor.identity
    if descriptor.sfreq == 0 and bufsize != int(bufsize):
        raise ValueError(
            f"The buffer size of the irregularly sampled stream {identity.as_tuple()} "
            f"must be a whole number of samples, got {bufsize}."
        )
    return StreamLSL(
        bufsize,
        name=identity.name,
        stype=identity.stype,
        source_id=identity.source_id,
    )


def connect_stream(descriptor: StreamDescriptor, bufsize: float) -> BaseStream:
    """Create and connect a stream for ``descriptor``, blocking until it is ready.

    Parameters
    ----------
    descriptor : StreamDescriptor
        Descriptor of the stream to connect to, providing the exact identity tuple.
    bufsize : float
        Size of the stream buffer, in seconds.

    Returns
    -------
    stream : BaseStream
        The connected stream.

    Raises
    ------
    RuntimeError
        If the identity no longer matches exactly one stream, or if the stream is a
        string stream, which :class:`~mne_lsl.stream.StreamLSL` refuses.
    ValueError
        If ``bufsize`` is not integral while the stream is irregularly sampled, raised
        by :func:`create_stream` before anything is opened.

    Notes
    -----
    This and :func:`reconnect_stream` are the two places in the viewer where ``recover``
    is written, and both write ``False``. With ``recover=True`` liblsl re-resolves a
    lost stream forever at 500 ms intervals, matching on the identity, the channel count
    and the format but not on the sampling rate, and Python observes nothing at all --
    no error, no state change, just an indefinitely empty pull. The per-document
    disconnection notice the viewer must show cannot exist on top of that, and neither
    can the check that the stream which came back is the stream which left. The library
    default stays ``True``, as flipping it would change the behaviour of every existing
    consumer, so the viewer passes ``False`` explicitly.

    Both live in this module so that the discovery transport needs no LSL knowledge:
    ``recover`` is a :class:`~mne_lsl.stream.StreamLSL` keyword absent from
    :meth:`~mne_lsl.stream.BaseStream.connect`, thus passing it from a
    ``BaseStream``-typed call site would break on the first non-LSL protocol.

    ``acquisition_delay`` and ``timeout`` keep the library defaults. A generous timeout
    would directly lengthen the failure path, as a missing identity burns most of it
    before raising.
    """
    return create_stream(descriptor, bufsize).connect(recover=False)


def probe_channels(descriptor: StreamDescriptor) -> list[str]:
    """Return the channel names of a stream, by briefly connecting to it.

    Discovery reports the channel count but not the channel names, which are only
    available once an inlet is opened. This is used to evaluate the availability of a
    saved configuration against the channels it expects.

    Parameters
    ----------
    descriptor : StreamDescriptor
        Descriptor of the stream to probe.

    Returns
    -------
    ch_names : list of str
        The channel names, in acquisition order.

    Raises
    ------
    RuntimeError
        If the identity does not match exactly one stream on the network, or if the
        inlet could not be opened or read within the probe timeout. The message is the
        reason the interface shows, thus this raises rather than returning an empty
        list, which would be indistinguishable from a stream publishing no channels.

    Notes
    -----
    The identity is re-resolved here, because a descriptor deliberately carries no
    stream info of its own -- see this module's ownership rule. A fully specified
    identity resolves instantly on a local network, and the repeated probe cost is meant
    to be avoided one level up, by caching the *names* against ``(identity, uid)``: a
    changed ``uid`` proves the channel set may have changed, and an unchanged one proves
    it cannot have.

    The names come from :meth:`~mne_lsl.lsl.StreamInfo.get_channel_info`, never from
    :meth:`~mne_lsl.lsl.StreamInfo.get_channel_names`: a stream publishing duplicates
    advertises ``['Cz', 'Cz', 'Fp1']`` on the wire while the viewer sees MNE's
    deterministic ``['Cz-0', 'Cz-1', 'Fp1']``, and a stream publishing no names at all
    advertises ``None`` while the viewer sees ``['0', '1', ...]``. Comparing the raw
    names would make both kinds of stream permanently unavailable.

    Event and marker streams are matched on their identity only and are never probed: a
    string marker stream has no meaningful channel set. That is the caller's rule, not a
    guard here.

    The probe inlet is destroyed with :meth:`~mne_lsl.lsl.StreamInlet._del` and never
    closed first: ``close_stream`` is broken in liblsl (sccn/liblsl#180) and closing
    before destruction engages the recovery machinery, whose cancellation races with the
    destruction and can abort the process (sccn/liblsl#220). ``recover=False`` keeps
    that machinery out of a throwaway inlet altogether.
    """
    sinfos = resolve_streams(_PROBE_TIMEOUT, *descriptor.identity.as_tuple())
    if len(sinfos) != 1:
        # mirrors 'StreamLSL.connect''s own message, as both are shown to the user.
        raise RuntimeError(
            "The identity 'name', 'stype' and 'source_id' does not uniquely identify "
            f"an LSL stream. {len(sinfos)} were found: "
            f"{[(sinfo.name, sinfo.stype, sinfo.source_id) for sinfo in sinfos]}."
        )
    inlet = StreamInlet(sinfos[0], max_buffered=1, recover=False)
    try:
        inlet.open_stream(timeout=_PROBE_TIMEOUT)
        sinfo = inlet.get_sinfo(timeout=_PROBE_TIMEOUT)
        # copied out of an 'Info' which is about to be destroyed with its stream info.
        return list(sinfo.get_channel_info()["ch_names"])
    finally:
        inlet._del()


def stream_identity(stream: BaseStream) -> StreamIdentity:
    """Return the exact identity of a connected stream.

    Parameters
    ----------
    stream : BaseStream
        A connected stream.

    Returns
    -------
    identity : StreamIdentity
        The ``(name, stype, source_id)`` triple, as the opened inlet reports it.

    Raises
    ------
    RuntimeError
        If the stream is not connected. The three fields are only guaranteed to be set
        once :meth:`~mne_lsl.stream.StreamLSL.connect` has back-filled them from the
        inlet it opened.
    TypeError
        If the stream is not an LSL stream, i.e. if it carries no LSL identity.

    Notes
    -----
    This lives here because the identity is LSL-specific: it is read from attributes
    which :class:`~mne_lsl.stream.BaseStream` does not define, and this module is the
    only one allowed to name :class:`~mne_lsl.stream.StreamLSL`. It exists for the
    borrowed-stream path, where the viewer is handed a stream instead of a descriptor.
    """
    if not isinstance(stream, StreamLSL):
        raise TypeError(
            "The viewer can only open a document for an LSL stream, which carries an "
            f"identity, got {type(stream).__name__}."
        )
    if not stream.connected:
        raise RuntimeError(
            "The identity of a stream is only known once it is connected: the name, "
            "the type and the source ID are back-filled from the inlet it opened."
        )
    return StreamIdentity(
        name=stream.name, stype=stream.stype, source_id=stream.source_id
    )


def reconnect_stream(stream: BaseStream) -> None:
    """Reconnect an existing stream in place, blocking until it is ready.

    Parameters
    ----------
    stream : BaseStream
        The stream to reconnect, connected or not.

    Raises
    ------
    RuntimeError
        If the identity no longer matches exactly one stream on the network.

    Notes
    -----
    The identity triple, the buffer size and every other constructor argument survive
    the reset the library performs on a disconnection, so the same object reconnects to
    the same identity with an identical buffer. What does *not* survive is the buffer
    content: a reconnection allocates a fresh one, so the samples acquired before the
    outage are gone from the stream and the display refills from the right edge.
    Stitching the two sides of an outage is not attempted.

    Both halves of the body are load-bearing. The disconnection first is required
    because :meth:`~mne_lsl.stream.BaseStream.connect` warns and returns unchanged on an
    already connected stream: without it a source which recovered on its own would
    silently never be reconnected, and a warning is an error under the test suite.
    ``recover=False`` is required for the reason :func:`connect_stream` records.
    """
    if stream.connected:
        stream.disconnect()
    stream.connect(recover=False)


def stream_signature(stream: BaseStream) -> StreamSignature:
    """Return the resume signature of a connected stream.

    Parameters
    ----------
    stream : BaseStream
        A connected stream.

    Returns
    -------
    signature : StreamSignature
        Everything :func:`~mne_lsl.viewer.backend.signature_mismatch` compares.

    Raises
    ------
    RuntimeError
        If the stream is not connected, raised by the property reads.
    TypeError
        If the stream is not an LSL stream, i.e. if it carries no LSL identity.

    Notes
    -----
    The channel names come from :attr:`~mne_lsl.stream.BaseStream.info`, never from
    :meth:`~mne_lsl.lsl.StreamInfo.get_channel_info`: the two return the same
    de-duplicated list, but the latter re-emits the duplicate-name warning of the
    initial connection every time it is called, and costs a full XML parse to do it.
    """
    return StreamSignature(
        identity=stream_identity(stream),
        sfreq=float(stream.info["sfreq"]),
        dtype=str(stream.dtype),
        ch_names=tuple(stream.info["ch_names"]),
    )


def disconnect_text(reason: BaseException | None) -> str:
    """Return the one-line reason a document shows for a disconnection.

    Parameters
    ----------
    reason : BaseException | None
        :attr:`~mne_lsl.stream.BaseStream.disconnect_reason` of the stream, ``None``
        when the stream was disconnected cleanly rather than by its acquisition thread.

    Returns
    -------
    text : str
        A short reason, without trailing punctuation, so that a caller may append to it.

    Notes
    -----
    This is the only place in the viewer which names ``LostError``, which is why it
    lives in this module: a lost source and a stream someone else disconnected are the
    same state with different wording, and the document must not import from
    :mod:`mne_lsl.lsl` to tell them apart.
    """
    if isinstance(reason, LostError):
        return "Stream lost"
    if reason is None:
        return "Stream disconnected"
    return f"Stream error: {type(reason).__name__}"
