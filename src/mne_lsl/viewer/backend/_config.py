"""Named configurations: the envelope and its JSON persistence.

Configurations are application-managed: the storage location and the file format are
deliberately hidden from the interface, which only ever asks for a human-readable name.
This module is deliberately free of Qt and of LSL imports.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...utils._checks import check_type
from ...utils.logs import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# Bumped whenever the serialized layout changes; a configuration carrying an unknown
# version is reported as 'invalid' rather than migrated silently.
SCHEMA_VERSION = 1

# Availability states of a configuration card. The 4 settled states are distinct on
# purpose: the reason shown to the user differs.
STATE_CHECKING = "checking"  # the channel probe of an identity match is in flight
STATE_AVAILABLE = "available"  # every identity is present and the channels match
STATE_UNAVAILABLE_NO_MATCH = "unavailable-no-match"  # an identity tuple is missing
STATE_UNAVAILABLE_CHANNELS = "unavailable-channels"  # identities match, channels do not
STATE_INVALID = "invalid"  # schema/version mismatch or corrupt JSON
STATE_LOADING = "loading"  # the configuration is being opened

# Reasons carried by an invalid configuration, i.e. by one whose envelope could not be
# interpreted. Two distinct strings, because the two situations call for different user
# actions: delete the file, or upgrade MNE-LSL.
_REASON_UNREADABLE = "unreadable configuration file"
_REASON_NEWER = "written by a newer version of MNE-LSL"

# Case-insensitive on every Windows filesystem, and unusable as a file name whatever the
# extension. mne-lsl ships Windows wheels, thus the check is not academic.
_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{k}" for k in range(1, 10)),
        *(f"lpt{k}" for k in range(1, 10)),
    }
)

# Bound on the slug collision walk. A user with 999 configurations sharing a slug has a
# different problem; the bound exists so a bug cannot spin forever.
_MAX_COLLISIONS = 1000

# Names carried by an unavailability reason before the list is cut short. The count
# carries the magnitude, the names carry the identification: a 256-channel mismatch must
# not render a 4 kB label under a card title.
_REASON_NAMES = 3


def config_dir() -> Path:
    """Return the directory holding the saved configurations.

    Returns
    -------
    directory : Path
        ``~/.mne-lsl/viewer/configurations``, which may not exist yet.

    Notes
    -----
    Computed on every call rather than resolved once into a module constant, for two
    reasons. The home directory is read at call time instead of at import time, which is
    what lets the tests point it elsewhere with
    ``monkeypatch.setattr(Path, "home", ...)``, and it keeps the import of this module
    free of filesystem access. The ``configurations`` subdirectory matters as well: the
    global theme preference lives in ``~/.mne-lsl/viewer/settings.json``, and a
    ``*.json`` glob over the parent would list it as a corrupt configuration.

    A persistent home dot-directory, not a cache which the OS may purge, and not a
    :mod:`platformdirs` location: a single visible path, identical on every platform, is
    the deliberate choice. There is no environment override.
    """
    return Path.home() / ".mne-lsl" / "viewer" / "configurations"


def channel_key(identity: tuple[str, str, str]) -> str:
    r"""Return the key of ``identity`` in the ``channels`` mapping of a configuration.

    Parameters
    ----------
    identity : tuple of str
        Exact identity tuple ``(name, stype, source_id)``.

    Returns
    -------
    key : str
        The 3 elements as a JSON array, e.g. ``'["Polar", "eeg", "unit-001"]'``.

    Notes
    -----
    Spelled once here because it is the contract between the writer of a configuration
    and the availability check which reads it back: two independent spellings of the
    same encoding is how the two silently diverge.

    A JSON array rather than a join on a separator character. An LSL stream name is free
    text and a tabulation *is* accepted in one, so joining on a tabulation would give
    ``('A\tB', 'C', 'D')`` and ``('A', 'B\tC', 'D')`` the same key -- contrived, but an
    ambiguous key is not a property worth relying on. :mod:`json` escapes its own
    separators, thus the encoding is unambiguous whatever the identifiers hold.
    """
    return json.dumps(identity)


@dataclass
class ViewerConfig:
    r"""One named configuration.

    Only the envelope is defined: the exact content of :attr:`presentation` is still an
    open question, thus it is carried as an opaque mapping which the viewer writes and
    reads back as a whole and which nothing else interprets.

    Parameters
    ----------
    name : str
        Human-readable configuration name, also its file name.
    schema_version : int
        Version of the serialized layout, see :data:`SCHEMA_VERSION`.
    streams : list of tuple of str
        Exact identity tuples ``(name, stype, source_id)`` required by the
        configuration, regular and event streams alike. An identity which is missing
        from the network makes the configuration unavailable.
    channels : dict
        Channel names expected per stream, keyed by :func:`channel_key`. Compared
        against the probed channel set to distinguish
        :data:`STATE_UNAVAILABLE_CHANNELS` from :data:`STATE_UNAVAILABLE_NO_MATCH`. An
        identity of :attr:`streams` with no entry here is an event source: it takes part
        in the identity match and is never probed.
    presentation : dict
        Opaque payload: per-document presentation state (channel edits, visibility and
        order, display and processing settings, event mappings), the Qt-ADS layout
        returned by ``CDockManager.saveState()`` and the main-window geometry.
    invalid_reason : str | None
        ``None`` for a configuration whose envelope was interpreted. Otherwise the
        reason the interface shows on the invalid card, every other field being empty.
    """

    name: str
    schema_version: int = SCHEMA_VERSION
    streams: list[tuple[str, str, str]] = field(default_factory=list)
    channels: dict[str, list[str]] = field(default_factory=dict)
    presentation: dict[str, Any] = field(default_factory=dict)
    invalid_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigurationState:
    """Rendered availability of one configuration, i.e. what one card shows.

    Parameters
    ----------
    name : str
        Name of the configuration, as its card is titled.
    state : str
        One of the ``STATE_*`` constants of this module.
    reason : str
        One-line explanation shown under the title, ``''`` when the state needs none.
    n_streams : int
        Number of streams the configuration requires, ``0`` unless it is available: it
        is the 'N streams' line of an available card and nothing else.

    Notes
    -----
    Everything the launcher needs and nothing more: no path, no
    :class:`ViewerConfig`, no identity. That is what keeps the card region a passive
    renderer of values the window computed.
    """

    name: str
    state: str
    reason: str
    n_streams: int


def identity_text(identity: tuple[str, str, str]) -> str:
    """Return the user-visible rendering of ``identity``, e.g. ``'A (eeg/s1)'``.

    Parameters
    ----------
    identity : tuple of str
        Exact identity tuple ``(name, stype, source_id)``.

    Returns
    -------
    text : str
        The three fields on one line.

    Notes
    -----
    Spelled once because the same rendering appears in three unavailability reasons and
    in the load-failure dialog, and because the type and the source ID are what keep two
    same-name streams distinguishable: dropping them would render a name collision
    identically twice.
    """
    name, stype, source_id = identity
    return f"{name} ({stype}/{source_id})"


def _capped(items: list[str]) -> str:
    """Return ``items`` joined, cut to :data:`_REASON_NAMES` entries.

    Parameters
    ----------
    items : list of str
        The already-rendered entries, in the order they are shown.

    Returns
    -------
    text : str
        The entries joined by a comma, followed by an ellipsis when some were cut.
    """
    if len(items) <= _REASON_NAMES:
        return ", ".join(items)
    return ", ".join(items[:_REASON_NAMES]) + ", …"


def _missing_reason(missing: list[tuple[str, str, str]], total: int) -> str:
    """Return the reason of a configuration whose identities are not all present.

    Parameters
    ----------
    missing : list of tuple of str
        The identities absent from the last discovery pass.
    total : int
        Number of identities the configuration requires, for the ``'2 of 3'`` form.

    Returns
    -------
    reason : str
        The one-line reason.
    """
    text = _capped([identity_text(identity) for identity in missing])
    if len(missing) == 1:
        return f"No matching stream: {text}."
    return (
        f"No matching stream: {len(missing)} of {total} required streams are "
        f"missing — {text}."
    )


def _unreachable_reason(identity: tuple[str, str, str], message: str) -> str:
    """Return the reason of a stream which is present but could not be probed.

    Parameters
    ----------
    identity : tuple of str
        Identity of the stream which could not be reached.
    message : str
        Text of the exception the probe raised.

    Returns
    -------
    reason : str
        The one-line reason.
    """
    return f"Could not reach {identity_text(identity)}: {message}"


def missing_channels(expected: Iterable[str], present: Iterable[str]) -> list[str]:
    """Return the ``expected`` names which ``present`` lacks, in saved order.

    Parameters
    ----------
    expected : iterable of str
        Channel names the configuration was saved with.
    present : iterable of str
        Channel names the stream reports now, probed or connected.

    Returns
    -------
    missing : list of str
        The absent names, in the order they were saved.

    Notes
    -----
    Shared by the availability check and by the load path, which read their inputs from
    different places -- a probed description before connecting, the measurement info
    afterwards -- but must answer the identical question. Two spellings of one
    comparison is the shape that produces this feature's worst failure: a card reading
    available and a load that then refuses, or the reverse.

    ``present`` is materialised once. Rebuilding the set per name is quadratic, which at
    a few hundred channels across a few dozen configurations turns one republish into
    hundreds of milliseconds of work on the thread that paints.
    """
    have = set(present)
    return [name for name in expected if name not in have]


def channels_reason(identity: tuple[str, str, str], missing: list[str]) -> str:
    """Return the reason of a stream which no longer provides its saved channels.

    Parameters
    ----------
    identity : tuple of str
        Identity of the stream whose channel set shrank.
    missing : list of str
        Names the configuration expects and the stream no longer publishes.

    Returns
    -------
    reason : str
        The one-line reason, carrying the count and at most :data:`_REASON_NAMES` names.

    Notes
    -----
    Public, unlike the two reason builders next to it, because the same sentence is the
    one the load path shows when the *connected* metadata no longer covers the saved
    channel set: two spellings of one sentence would let the card and the dialog
    describe the same situation differently.
    """
    return (
        f"{identity_text(identity)} no longer provides {len(missing)} of its saved "
        f"channels ({_capped(missing)})."
    )


def evaluate_state(
    cfg: ViewerConfig,
    present: frozenset[tuple[str, str, str]] | None,
    probed: Mapping[tuple[str, str, str], list[str] | str],
) -> ConfigurationState:
    """Return the availability of ``cfg`` against a discovery pass and its probes.

    Parameters
    ----------
    cfg : ViewerConfig
        One configuration, as :func:`list_configurations` returned it.
    present : frozenset of tuple of str | None
        The identities the last discovery pass found, or ``None`` when no pass has
        completed yet.
    probed : dict
        Per identity, either the probed channel names or the message of the exception
        the probe raised. An identity absent from the mapping has a probe in flight or
        not submitted yet.

    Returns
    -------
    state : ConfigurationState
        The rendered card state.

    Notes
    -----
    Pure: no I/O, no Qt, no mutation of either argument -- an in-place ``pop`` or
    ``sort`` here would corrupt the caller's probe cache. It also does not know
    :data:`STATE_LOADING`, which is imposed by the caller on the one configuration it is
    opening, and it does not sort or group: presentation order belongs to the launcher.

    The evaluation order is load-bearing. An unreadable envelope is terminal for the
    session and is therefore reported *before* the identity check, because such a
    configuration has an empty ``streams`` list and would otherwise be reported as
    missing every stream -- telling the user to plug a device in to repair a corrupt
    file. ``present is None`` gets a reason of its own rather than a state of its own:
    before the first pass, 'no matching stream' is simply a false statement.

    A channel set matches when the saved names are a **subset** of the probed ones, by
    name: extras are tolerated and neither the order, the count, the types nor the
    sampling rate are compared. A channel type is viewer-editable, so gating on it would
    make a configuration unavailable *because* the user had edited a type; and the
    degenerate descriptions are what make subset-by-name right, since a stream
    publishing no names reduces the check to 'the channel count did not shrink', which
    is the best answer available for one.

    A failure which is already final wins over a sibling whose probe has not landed:
    reporting the precise reason as soon as it is known is the whole point of probing
    eagerly, and no pending check can change the verdict.
    """
    if cfg.invalid_reason is not None:
        return ConfigurationState(cfg.name, STATE_INVALID, cfg.invalid_reason, 0)
    if present is None:
        return ConfigurationState(
            cfg.name, STATE_UNAVAILABLE_NO_MATCH, "Waiting for discovery…", 0
        )
    missing = [identity for identity in cfg.streams if identity not in present]
    if missing:
        return ConfigurationState(
            cfg.name,
            STATE_UNAVAILABLE_NO_MATCH,
            _missing_reason(missing, len(cfg.streams)),
            0,
        )
    checking = False
    for identity in cfg.streams:
        expected = cfg.channels.get(channel_key(identity))
        if expected is None:
            continue  # an event source: matched on its identity and never probed
        if identity not in probed:
            checking = True
            continue
        result = probed[identity]
        if isinstance(result, str):
            # the probe's own message, never a name sequence: 'set' over a string
            # compares *characters* and would report a channel mismatch for a stream
            # which could not be reached at all.
            return ConfigurationState(
                cfg.name,
                STATE_UNAVAILABLE_NO_MATCH,
                _unreachable_reason(identity, result),
                0,
            )
        absent = missing_channels(expected, result)
        if absent:
            return ConfigurationState(
                cfg.name,
                STATE_UNAVAILABLE_CHANNELS,
                channels_reason(identity, absent),
                0,
            )
    if checking:
        return ConfigurationState(cfg.name, STATE_CHECKING, "Checking availability…", 0)
    return ConfigurationState(cfg.name, STATE_AVAILABLE, "", len(cfg.streams))


def _slug(name: str) -> str:
    """Return the file stem of a configuration named ``name``.

    Parameters
    ----------
    name : str
        Human-readable configuration name.

    Returns
    -------
    slug : str
        A stem which is safe on every supported filesystem.

    Notes
    -----
    ``casefold()`` runs *before* the substitution, which is what makes APFS, HFS+, ext4
    and NTFS behave identically: the stem can then never differ from another one by case
    alone. The character class strips path separators, ``..``, spaces and every
    non-ASCII character, so the human name stays free-form -- a purely non-Latin name
    slugs to ``'configuration'`` and collides its way to ``configuration-2``.

    The trailing hyphen is stripped **twice** on purpose: truncating to 64 characters
    can re-introduce one which the first strip had removed.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:64].strip("-")
    slug = slug or "configuration"
    if slug in _WINDOWS_RESERVED:
        slug += "-cfg"
    return slug


def _unreadable(title: str) -> ViewerConfig:
    """Return the invalid card of a configuration whose envelope did not interpret.

    Parameters
    ----------
    title : str
        Title the interface shows on the card, see :func:`_parse`.

    Returns
    -------
    cfg : ViewerConfig
        An empty configuration carrying :data:`_REASON_UNREADABLE`.
    """
    return ViewerConfig(name=title, invalid_reason=_REASON_UNREADABLE)


def _read(path: Path) -> dict[str, Any] | None:
    """Return the JSON object stored in ``path``, or ``None`` if unreadable.

    Parameters
    ----------
    path : Path
        Path of the file to read.

    Returns
    -------
    data : dict | None
        The decoded mapping, or ``None`` if the file could not be read, does not hold
        valid JSON, or does not hold a JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_streams(data: dict[str, Any]) -> list[tuple[str, str, str]] | None:
    """Return the identity tuples of ``data``, or ``None`` if the list is malformed.

    Parameters
    ----------
    data : dict
        The decoded configuration mapping.

    Returns
    -------
    streams : list of tuple of str | None
        The identity tuples, de-duplicated in saved order, or ``None`` if the field is
        absent, is not a non-empty list, or holds an entry which is not 3 strings of
        which the first is non-empty.

    Notes
    -----
    Only the name is required to be non-empty. A source identifier is **optional** in
    the protocol and real devices do omit it, and such a stream is discovered, probed
    and connected perfectly well -- so refusing it here would make the viewer write a
    file its own reader destroys at the next launch, which is exactly the disagreement
    the writer's non-empty check exists to prevent.

    Duplicates are dropped rather than rejected. A repeated identity would otherwise be
    counted twice on the card, opened twice by the connector, and built into two
    documents sharing one stream -- two channel models writing one measurement info,
    each taking the other's edits as its baseline, so the next save records the wrong
    deltas.
    """
    streams = data.get("streams")
    if not isinstance(streams, list) or len(streams) == 0:
        return None
    identities = []
    for entry in streams:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            return None
        if not all(isinstance(item, str) for item in entry) or not entry[0]:
            return None
        identities.append(tuple(entry))
    return list(dict.fromkeys(identities))


def _parse_channels(data: dict[str, Any], path: Path) -> dict[str, list[str]] | None:
    """Return the channel mapping of ``data``, or ``None`` if it is not a mapping.

    Parameters
    ----------
    data : dict
        The decoded configuration mapping.
    path : Path
        Path of the file, for the log messages.

    Returns
    -------
    channels : dict | None
        The channel names per identity key. ``None`` only if the field is present and is
        not a mapping, which is an envelope failure; an unusable *entry* is dropped with
        a log line instead, as leaf tolerance is what lets an additive schema change
        stay version-compatible.
    """
    channels = data.get("channels", {})
    if not isinstance(channels, dict):
        return None
    parsed: dict[str, list[str]] = {}
    for key, value in channels.items():
        if not isinstance(key, str) or not isinstance(value, list):
            logger.warning("Dropping a malformed 'channels' entry of %s.", path)
            continue
        if not all(isinstance(item, str) for item in value):
            logger.warning(
                "Dropping the 'channels' entry of %s which does not hold channel "
                "names.",
                path,
            )
            continue
        parsed[key] = list(value)
    return parsed


def _parse(path: Path) -> ViewerConfig:
    """Return the configuration stored in ``path``, invalid if the envelope is not.

    Parameters
    ----------
    path : Path
        Path of the file to parse.

    Returns
    -------
    cfg : ViewerConfig
        The parsed configuration, or one carrying an :attr:`ViewerConfig.invalid_reason`
        and an empty payload.

    Notes
    -----
    The envelope is validated strictly and everything beyond it is tolerated, which is
    the line that lets an additive field arrive without bumping
    :data:`SCHEMA_VERSION`: a strict leaf check would make every future viewer change a
    breaking change, while a strict envelope check is what actually protects the load
    path. An invalid card is titled by the file's ``name`` when that much is readable,
    else by the file stem, so that it can be identified and deleted.
    """
    data = _read(path)
    if data is None:
        return _unreadable(path.stem)
    # the title of an invalid card, once the file itself was decoded.
    name = data.get("name")
    title = name.strip() if isinstance(name, str) and name.strip() else path.stem
    version = data.get("schema_version")
    # a bool passes 'isinstance(True, int)', and a JSON float is not an int: both are
    # rejected, as neither can be a version this reader knows.
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        return _unreadable(title)
    if version > SCHEMA_VERSION:
        return ViewerConfig(name=title, invalid_reason=_REASON_NEWER)
    # 'version < SCHEMA_VERSION' is unreachable while SCHEMA_VERSION == 1. A version 2
    # gains one '_upgrade_v1_to_v2(data)' call here, not a migration framework.
    if not isinstance(name, str) or not name.strip():
        return _unreadable(title)
    streams = _parse_streams(data)
    if streams is None:
        return _unreadable(title)
    channels = _parse_channels(data, path)
    if channels is None:
        return _unreadable(title)
    presentation = data.get("presentation", {})
    if not isinstance(presentation, dict):
        logger.warning("Ignoring the malformed 'presentation' payload of %s.", path)
        presentation = {}
    # unknown top-level keys are ignored, thus dropped by the next save. That is the
    # documented consequence of tolerating them rather than refusing the file.
    return ViewerConfig(
        name=name,
        schema_version=version,
        streams=streams,
        channels=channels,
        presentation=presentation,
    )


def _find(name: str) -> Path | None:
    """Return the file holding the configuration named ``name``, if any.

    Parameters
    ----------
    name : str
        Name of the configuration to look up, matched case-insensitively.

    Returns
    -------
    path : Path | None
        Path of the file, or ``None`` if no file matches.

    Notes
    -----
    The authoritative name lives *inside* the file, thus every configuration is parsed
    and it is that name which is compared. An invalid configuration matches too, on the
    title its card carries -- the inner name when that much was readable, the file stem
    otherwise -- because :func:`delete_configuration` is the only way to clear a corrupt
    file from the interface and can only be asked for what the card shows. A valid match
    wins over an invalid one.

    The file stem of a *valid* configuration is deliberately never compared: it is
    derived from the name, it is not the name. Comparing it would let
    ``save_configuration(ViewerConfig(name='my-config'))`` silently overwrite the
    unrelated configuration named ``'My Config'``, which :func:`_slug` had already
    put in ``my-config.json``.
    """
    target = name.casefold()
    invalid: Path | None = None
    for path in sorted(config_dir().glob("*.json")):
        cfg = _parse(path)
        if cfg.name.casefold() != target:
            continue
        if cfg.invalid_reason is None:
            return path
        if invalid is None:
            invalid = path
    return invalid


def _new_path(name: str) -> Path:
    """Return a free path for a configuration named ``name``.

    Parameters
    ----------
    name : str
        Name of the configuration to write.

    Returns
    -------
    path : Path
        ``<slug>.json``, or ``<slug>-2.json``, ``<slug>-3.json``, ... if taken.

    Raises
    ------
    RuntimeError
        If no free path was found.
    """
    directory = config_dir()
    slug = _slug(name)
    for k in range(1, _MAX_COLLISIONS):
        path = directory / (f"{slug}.json" if k == 1 else f"{slug}-{k}.json")
        if not path.exists():
            return path
    raise RuntimeError(
        f"Could not find a free file name for the configuration '{name}': "
        f"'{slug}.json' and its {_MAX_COLLISIONS - 2} numbered variants are all taken."
    )


def _to_dict(cfg: ViewerConfig) -> dict[str, Any]:
    """Return the JSON-serializable mapping of ``cfg``.

    Parameters
    ----------
    cfg : ViewerConfig
        The configuration to serialize.

    Returns
    -------
    data : dict
        The mapping written to disk. :attr:`ViewerConfig.invalid_reason` is not part of
        it: it describes a file which could not be read, never one being written.

    Notes
    -----
    The payload is referenced, not copied: :func:`json.dumps` serializes a tuple as a
    JSON array just as it does a list, and :func:`_atomic_write` consumes the mapping
    synchronously, so there is no window in which a caller could mutate it.
    """
    return {
        "schema_version": cfg.schema_version,
        "name": cfg.name,
        "streams": cfg.streams,
        "channels": cfg.channels,
        "presentation": cfg.presentation,
    }


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` as JSON, atomically.

    Parameters
    ----------
    path : Path
        Path of the file to write.
    data : dict
        The mapping to serialize.

    Notes
    -----
    :func:`os.replace` on a closed handle guarantees that a reader sees either the old
    or the new file, never a torn one, which is the only property that matters here. No
    :func:`os.fsync`: this is a preferences store, trivially recreatable, not a
    durability-critical one. ``indent=2`` because inspectability is the reason JSON was
    chosen in the first place.

    A failure between the write and the replace leaves a ``.json.tmp`` file behind. It
    is deliberately never cleaned up: ``glob('*.json')`` does not match it, so it is
    invisible to the interface and harmless, and the next save of the same
    configuration overwrites it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def list_configurations() -> list[ViewerConfig]:
    """Return the saved configurations, ordered by name.

    A file which does not parse or which carries an unknown schema version is returned
    with an empty payload so that the interface can show it as invalid instead of
    hiding it.

    Returns
    -------
    configurations : list of ViewerConfig
        The saved configurations found in :func:`config_dir`.

    Notes
    -----
    Never creates the directory: nothing in the package writes to the filesystem at
    import or on first use. A missing directory simply yields an empty list, as
    :meth:`~pathlib.Path.glob` on a missing directory is empty rather than an error.

    There is no index file. Listing is one glob plus a parse over a handful of files,
    whereas an index would add a second write target and a consistency problem to
    repair whenever a file appears or disappears out of band.
    """
    configs = [_parse(path) for path in sorted(config_dir().glob("*.json"))]
    return sorted(configs, key=lambda cfg: cfg.name.casefold())


def save_configuration(cfg: ViewerConfig) -> Path:
    """Write ``cfg`` to :func:`config_dir`, replacing an existing file of that name.

    Parameters
    ----------
    cfg : ViewerConfig
        The configuration to save.

    Returns
    -------
    fname : Path
        Path of the written file.

    Raises
    ------
    TypeError
        If ``cfg`` is not a :class:`ViewerConfig`.
    ValueError
        If :attr:`ViewerConfig.name` is empty or blank, or if
        :attr:`ViewerConfig.streams` is empty.

    Notes
    -----
    The name is only checked for being non-blank. The 120-character cap is interface
    policy, applied by the Save-as prompt, and no character is rejected because
    :func:`_slug` neutralizes them all.

    An empty :attr:`ViewerConfig.streams` is refused because :func:`_parse_streams`
    refuses it on the way back in: writing one would return a path to a file the next
    :func:`list_configurations` reports as invalid. The writer and the reader have to
    agree on that, and a configuration referencing no stream has nothing to show anyway.

    A :class:`TypeError` raised by :func:`json.dumps` on a non-serializable
    :attr:`ViewerConfig.presentation` propagates: that payload is opaque to this module
    and a value it cannot serialize is a caller bug, not a corrupt file.
    """
    check_type(cfg, (ViewerConfig,), "cfg")
    if not cfg.name.strip():
        raise ValueError("The name of a configuration cannot be empty.")
    if len(cfg.streams) == 0:
        raise ValueError("A configuration must reference at least one stream.")
    path = _find(cfg.name) or _new_path(cfg.name)
    _atomic_write(path, _to_dict(cfg))
    logger.debug("Saved the configuration '%s' to %s.", cfg.name, path)
    return path


def rename_configuration(name: str, new_name: str) -> Path:
    """Rename the configuration named ``name`` to ``new_name``.

    Parameters
    ----------
    name : str
        Current name of the configuration.
    new_name : str
        New name, stripped of its surrounding whitespace.

    Returns
    -------
    fname : Path
        Path of the file holding the renamed configuration.

    Raises
    ------
    TypeError
        If either name is not a :class:`str`.
    ValueError
        If ``new_name`` is blank, if no configuration is named ``name``, if ``new_name``
        is already taken by a *different* configuration, or if the file could not be
        read -- a corrupt configuration has no payload to carry over and is cleared with
        :func:`delete_configuration` instead.

    Notes
    -----
    Deliberately not routed through :func:`save_configuration`, which looks the name up
    and would find the *old* file: it would overwrite that one and leave the rename a
    silent no-op. Only the ``name`` field is rewritten -- the presentation payload and
    the channel sets are carried over untouched.

    The new file is written **before** the old one is unlinked, so an interruption
    between the two leaves a duplicate rather than nothing, and a failing unlink is
    logged and swallowed for the same reason: the payload is already safe by then.
    Renaming to a name which differs only by case rewrites in place, because
    :func:`_slug` casefolds and the target path would otherwise collide with the source
    and walk to a numbered variant, leaving two files for one configuration.
    """
    check_type(name, (str,), "name")
    check_type(new_name, (str,), "new_name")
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("The name of a configuration cannot be empty.")
    path = _find(name)
    if path is None:
        raise ValueError(f"There is no saved configuration named '{name}'.")
    taken = _find(new_name)
    if taken is not None and taken != path:
        raise ValueError(f"A configuration named '{new_name}' already exists.")
    data = _read(path)
    if data is None:
        raise ValueError(
            f"The configuration '{name}' could not be read and cannot be renamed; "
            "delete it instead."
        )
    data["name"] = new_name
    if taken == path:
        _atomic_write(path, data)
        return path
    target = _new_path(new_name)
    _atomic_write(target, data)
    try:
        path.unlink(missing_ok=True)
    except OSError as error:  # see the note above: a duplicate beats a loss
        logger.warning(
            "Renamed the configuration '%s' to '%s' but could not remove %s: %s.",
            name,
            new_name,
            path,
            error,
        )
    logger.debug("Renamed the configuration '%s' to '%s'.", name, new_name)
    return target


def delete_configuration(name: str) -> None:
    """Delete the configuration named ``name``; missing files are ignored.

    Parameters
    ----------
    name : str
        Name of the configuration to delete.
    """
    check_type(name, (str,), "name")
    path = _find(name)
    if path is None:
        return
    path.unlink(missing_ok=True)
    logger.debug("Deleted the configuration '%s' at %s.", name, path)
