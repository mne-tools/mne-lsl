from ctypes import c_char_p, c_double, c_void_p
from typing import Union

from ..utils._checks import _check_type, _check_value
from . import lib
from .constants import fmt2string, string2fmt


class _BaseStreamInfo:
    def __init__(self, obj):
        self.obj = c_void_p(obj)
        if not self.obj:
            raise RuntimeError(
                "The StreamInfo could not be created from the description."
            )

    def __del__(self):
        """Destroy a SreamInfo."""
        try:
            lib.lsl_destroy_streaminfo(self.obj)
        except Exception:
            pass

    # -- Core information, assigned at construction ---------------------------
    @property
    def name(self) -> str:
        """Name of the stream.

        The name of the stream is defined by the application creating the LSL
        outlet. Streams with identical names can coexist, at the cost of
        ambiguity for the recording application and/or the experimenter.
        """
        return lib.lsl_get_name(self.obj).decode("utf-8")

    @property
    def stype(self) -> str:
        """Type of the stream.

        The content type is a short string, such as "EEG", "Gaze", ... which
        describes the content carried by the channel. If a stream contains
        mixed content, this value should be an empty string and the type should
        be stored in the description of individual channels.
        """
        return lib.lsl_get_type(self.obj).decode("utf-8")

    @property
    def n_channels(self) -> int:
        """Number of channels.

        A stream must have at least one channel. The number of channels remains
        constant for all samples.
        """
        return lib.lsl_get_channel_count(self.obj)

    @property
    def sfreq(self) -> float:
        """Sampling rate of the stream, according to the source (in Hz).

        If a stream is irregularly sampled, the sampling rate is set to 0.
        """
        return lib.lsl_get_nominal_srate(self.obj)

    @property
    def channel_format(self) -> str:
        """Channel format of a stream.

        All channels in a stream have the same format.
        """
        return fmt2string[lib.lsl_get_channel_format(self.obj)]

    @property
    def source_id(self) -> str:
        """Unique identifier of the stream's source.

        The unique source (or device) identifier is an optional piece of
        information that, if available, allows endpoints (such as the recording
        program) to re-acquire a stream automatically once if it came back
        online.
        """
        return lib.lsl_get_source_id(self.obj).decode("utf-8")

    # -- Hosting information, assigned when bound to an outlet/inlet ----------
    @property
    def version(self) -> int:
        """Version of the binary LSL library.

        The major version is version // 100.
        The minor version is version % 100.
        """
        return lib.lsl_get_version(self.obj)

    @property
    def created_at(self) -> float:
        """Creation time stamp of the stream.

        This is the time stamps at which the stream was first craeted, as
        determined by local_clock() on the providing machine.
        """
        return lib.lsl_get_created_at(self.obj)

    @property
    def uid(self) -> str:
        """Unique ID of the stream outlet instance.

        This ID is guaranteed to be different across multiple instantiations of
        the same outlet, e.g. after a re-start.
        """
        return lib.lsl_get_uid(self.obj).decode("utf-8")

    @property
    def session_id(self):
        return lib.lsl_get_session_id(self.obj).decode('utf-8')

    @property
    def hostname(self):
        return lib.lsl_get_hostname(self.obj).decode('utf-8')

    # -- Data description -----------------------------------------------------
    @property
    def desc(self):
        pass

    @property
    def as_xml(self):
        pass


class StreamInfo(_BaseStreamInfo):
    def __init__(
        self,
        name: str,
        stype: str,
        n_channels: int,
        sfreq: float,
        channel_format: int,
        source_id: str,
    ):
        _check_type(name, (str,), "name")
        _check_type(stype, (str,), "stype")
        _check_type(n_channels, ("int",), "n_channels")
        if n_channels <= 0:
            raise ValueError(
                "The number of channels 'n_channels' must be a strictly "
                f"positive integer. {n_channels} is invalid."
            )
        _check_type(sfreq, ("numeric",), "sfreq")
        _check_type(source_id, (str,), "source_id")

        obj = lib.lsl_create_streaminfo(
            c_char_p(str.encode(name)),
            c_char_p(str.encode(stype)),
            n_channels,
            c_double(sfreq),
            StreamInfo._string2fmt(channel_format),
            c_char_p(str.encode(source_id)),
        )
        super().__init__(obj)

    # -------------------------------------------------------------------------
    @staticmethod
    def _string2fmt(channel_format: Union[int, str]) -> int:
        """Convert a string format to its integer value."""
        _check_type(channel_format, (str, "int"), "channel_format")
        if isinstance(channel_format, str):
            channel_format = channel_format.lower()
            _check_value(channel_format, string2fmt, "channel_format")
            channel_format = string2fmt[channel_format]
        else:
            _check_value(channel_format, string2fmt.values(), "channel_format")
        return channel_format
