from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from mne_lsl.viewer.backend import (
    StreamIdentity,
    StreamSignature,
    signature_mismatch,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mne_lsl.viewer.backend import StreamDescriptor


def _signature(**kwargs) -> StreamSignature:
    """Return a signature built of plain values, with every field defaulted."""
    fields = dict(
        identity=StreamIdentity(name="Polar", stype="eeg", source_id="src-1"),
        sfreq=100.0,
        dtype="float32",
        ch_names=("Fp1", "Fpz", "ECG", "TRIGGER"),
    )
    fields.update(kwargs)
    return StreamSignature(**fields)


def test_as_tuple(descriptor: Callable[..., StreamDescriptor]) -> None:
    """Test that the identity round-trips through its tuple form."""
    identity = descriptor(name="Polar", source_id="src-1").identity
    assert identity.as_tuple() == ("Polar", "eeg", "src-1")
    assert StreamIdentity(*identity.as_tuple()) == identity


def test_descriptor_identity(descriptor: Callable[..., StreamDescriptor]) -> None:
    """Test that a descriptor exposes the identity object, not a bare tuple."""
    obj = descriptor(name="Polar", source_id="src-1")
    assert isinstance(obj.identity, StreamIdentity)
    assert obj.identity.as_tuple() == ("Polar", "eeg", "src-1")


def test_frozen(descriptor: Callable[..., StreamDescriptor]) -> None:
    """Test that neither dataclass can be mutated after construction."""
    obj = descriptor()
    for target, attribute in ((obj.identity, "name"), (obj, "n_channels")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, attribute, "other")


def test_slots_and_hashable(descriptor: Callable[..., StreamDescriptor]) -> None:
    """Test that both dataclasses carry no instance dictionary and are hashable.

    Hashability is what the identity de-duplication of 'ViewerWindow.open_streams'
    relies on, and slots are what makes the ownership rule of a descriptor structural.
    """
    obj = descriptor()
    for target in (obj.identity, obj):
        assert not hasattr(target, "__dict__")
        assert len({target, target}) == 1


def test_descriptor_uid_is_not_part_of_the_identity(
    descriptor: Callable[..., StreamDescriptor],
) -> None:
    """Test that two instances of a stream share an identity and differ as descriptors.

    The uid identifies the outlet *instance*, so it must stay out of 'StreamIdentity':
    folding it in would make every reconnection of one stream a different stream and
    break the 3-tuple identity rule the whole availability check rests on. It must still
    take part in the descriptor's own equality, since that is the probe-cache key.
    """
    first = descriptor(source_id="unit-001", uid="uid-a")
    second = descriptor(source_id="unit-001", uid="uid-b")
    assert first.identity == second.identity
    assert first != second
    assert len({first, second}) == 2


def test_descriptor_still_frozen_and_hashable(
    descriptor: Callable[..., StreamDescriptor],
) -> None:
    """Test that the uid field left the descriptor frozen, slotted and hashable.

    Pinned separately from 'test_frozen' and 'test_slots_and_hashable' because those two
    read fields which predate the uid: dropping 'frozen' or 'slots' while appending a
    field would silently stop the identity de-duplication of 'open_streams' from
    de-duplicating at all.
    """
    obj = descriptor(uid="uid-a")
    assert not hasattr(obj, "__dict__")
    assert isinstance(hash(obj), int)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.uid = "uid-b"


def test_same_name_different_source_id(
    descriptor: Callable[..., StreamDescriptor],
) -> None:
    """Test that two streams differing only by 'source_id' stay distinct.

    Two same-name streams are an explicitly supported case: the identity is the full
    3-tuple, thus the data layer must keep them apart before any interface does. Only
    inequality is asserted, never a hash inequality: distinct hashes are not a property
    the language promises, and the set below is what the de-duplication actually needs.
    """
    first, second = descriptor(source_id="unit-001"), descriptor(source_id="unit-002")
    assert first.identity != second.identity
    assert len({first, second}) == 2


@pytest.mark.parametrize("stype", ["eeg", ""])
def test_signature_mismatch_equal(stype: str) -> None:
    """Test that two equal signatures may resume into one another.

    The empty ``stype`` is the load-bearing half. An empty type is normal -- LSL allows
    it and 'PlayerLSL' publishes exactly that for any file whose channels are not all of
    one type -- so a non-empty check over the whole triple refuses to resume most real
    recordings. Kills a body replaced by an unconditional refusal, and kills requiring
    ``stype`` to be non-empty.
    """
    identity = StreamIdentity(name="Polar", stype=stype, source_id="src-1")
    signature = _signature(identity=identity)
    assert signature_mismatch(signature, _signature(identity=identity)) is None


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        pytest.param(
            dict(identity=StreamIdentity("Other", "eeg", "src-1")),
            "another stream answered the identity",
            id="identity",
        ),
        pytest.param(
            dict(sfreq=200.0), "sampling rate changed from 100 Hz", id="sfreq"
        ),
        pytest.param(
            dict(dtype="int16"), "format changed from float32 to int16", id="dtype"
        ),
        pytest.param(
            dict(ch_names=("Fp1", "Fpz", "ECG")),
            "channel count changed from 4 to 3",
            id="count",
        ),
        pytest.param(
            dict(ch_names=("Fp1", "Cz", "ECG", "TRIGGER")),
            "channels changed: 1 is now Cz and was Fpz",
            id="names",
        ),
    ],
)
def test_signature_mismatch_refuses(changed: dict, match: str) -> None:
    """Test one refusal per check, each naming the thing which changed.

    Kills deleting any single check, and kills swapping the count check with the ordered
    name check, which would report a name difference for a plain count change.
    """
    reason = signature_mismatch(_signature(), _signature(**changed))
    assert reason is not None
    assert match in reason


@pytest.mark.parametrize("triple", [("", "eeg", "src-1"), ("Polar", "eeg", "")])
def test_signature_mismatch_accepts_a_degenerate_identity(
    triple: tuple[str, str, str],
) -> None:
    """Test that a stream which barely identifies itself may still resume into itself.

    An empty ``name`` or ``source_id`` is legal LSL, and such a stream is discoverable,
    connectable and drawable. A non-empty check here reads the stream which answered,
    but the equality check above has already forced the two identities equal -- so it is
    really a test on the *document's own* identity, and a document over a stream
    publishing ``source_id=''`` refuses **itself** on its first resume: disconnected and
    left terminally refused for being what it always was.

    It buys nothing either: measured, 'resolve_streams' short-circuits on the first
    answer, so two outlets publishing an identical full triple both connect and a
    non-empty ``source_id`` refuses no impostor. The defence is the ordered name
    comparison, which runs regardless -- see the test below.
    """
    signature = _signature(identity=StreamIdentity(*triple))
    assert signature_mismatch(signature, signature) is None


def test_refusal_reasons_are_shown_to_a_human() -> None:
    """Test that a refusal renders an identity and a rate the way the banner needs them.

    Both reasons are shown verbatim on a notice strip and in the status bar. A raw
    Python tuple there reads as a traceback fragment, and the rendering exists once for
    exactly this. ``:g`` gives 6 significant digits, which turns a real change into
    *"the sampling rate changed from 1000 Hz to 1000 Hz"* -- a refusal naming nothing.
    """
    other = StreamIdentity(name="Other", stype="eeg", source_id="src-2")
    reason = signature_mismatch(_signature(), _signature(identity=other))
    assert reason is not None
    assert "Polar (eeg/src-1)" in reason
    assert "('Polar'" not in reason
    reason = signature_mismatch(
        _signature(sfreq=1000.0001), _signature(sfreq=1000.0002)
    )
    assert reason is not None
    assert "from 1000.0001 Hz to 1000.0002 Hz" in reason


def test_signature_mismatch_is_ordered() -> None:
    """Test that a reordered but identical name set is a refusal.

    Kills comparing 'set(ch_names)', which is the silent mis-mapping this rule exists to
    prevent: every piece of stream-side state the viewer sets is an integer index.
    """
    names = ("Fpz", "Fp1", "ECG", "TRIGGER")
    reason = signature_mismatch(_signature(), _signature(ch_names=names))
    assert reason is not None
    assert "channels changed: 0 is now Fpz and was Fp1" in reason


def test_signature_has_no_uid_nor_hostname() -> None:
    """Test that the outlet instance and the host are not part of the signature.

    Kills adding either field: a restarted source always publishes a new uid, so
    comparing it would make every recovery a refusal.
    """
    fields = {field.name for field in dataclasses.fields(StreamSignature)}
    assert fields == {"identity", "sfreq", "dtype", "ch_names"}
