from ctypes import c_char_p, c_double, c_void_p
from typing import Union

from ..utils._checks import _check_type, _check_value
from .constants import fmt2string, string2fmt
from .load_liblsl import lib
from .utils import XMLElement


class _BaseStreamInfo:
    """Base Stream information object, storing the declaration of a stream.

    A StreamInfo contains the following information:
    - Core information (name, number of channels, sampling frequency, channel
      format, ...)
    - Optional metadata about the stream content (channel labels, measurement
      units, ...)
    - Hosting information (uID, hostname, ...) if bound to an inlet or outlet
    """

    def __init__(self, obj):
        self.obj = c_void_p(obj)
        if not self.obj:
            raise RuntimeError(
                "The StreamInfo could not be created from the description."
            )
        self._channel_format = lib.lsl_get_channel_format(self.obj)

    def __del__(self):
        """Destroy a `~bsl.lsl.StreamInfo`."""
        try:
            lib.lsl_destroy_streaminfo(self.obj)
        except Exception:
            pass

    # -- Core information, assigned at construction ---------------------------
    @property
    def channel_format(self) -> str:
        """Channel format of a stream.

        All channels in a stream have the same format.
        """
        return fmt2string[self._channel_format]

    @property
    def name(self) -> str:
        """Name of the stream.

        The name of the stream is defined by the application creating the LSL
        outlet. Streams with identical names can coexist, at the cost of
        ambiguity for the recording application and/or the experimenter.
        """
        return lib.lsl_get_name(self.obj).decode("utf-8")

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

        If a stream is irregularly sampled, the sampling rate is set to ``0``.
        """
        return lib.lsl_get_nominal_srate(self.obj)

    @property
    def source_id(self) -> str:
        """Unique identifier of the stream's source.

        The unique source (or device) identifier is an optional piece of
        information that, if available, allows endpoints (such as the recording
        program) to re-acquire a stream automatically once if it came back
        online.
        """
        return lib.lsl_get_source_id(self.obj).decode("utf-8")

    @property
    def stype(self) -> str:
        """Type of the stream.

        The content type is a short string, such as ``"EEG"``, ``"Gaze"``, ...
        which describes the content carried by the channel. If a stream
        contains mixed content, this value should be an empty string and the
        type should be stored in the description of individual channels.
        """
        return lib.lsl_get_type(self.obj).decode("utf-8")

    # -- Hosting information, assigned when bound to an outlet/inlet ----------
    @property
    def created_at(self) -> float:
        """Timestamp at which the stream was created.

        This is the time stamps at which the stream was first created, as
        determined by `~bsl.lsl.local_clock` on the providing machine.
        """
        return lib.lsl_get_created_at(self.obj)

    @property
    def hostname(self) -> str:
        """Hostname of the providing machine."""
        return lib.lsl_get_hostname(self.obj).decode("utf-8")

    @property
    def session_id(self) -> str:
        """Session ID for the given stream.

        The session ID is an optional human-assigned identifier of the
        recording session. While it is rarely used, it can be used to prevent
        concurrent recording activities on the same sub-network (e.g., in
        multiple experiment areas) from seeing each other's streams
        (can be assigned in a configuration file read by liblsl, see also
        Network Connectivity in the LSL wiki).
        """
        return lib.lsl_get_session_id(self.obj).decode("utf-8")

    @property
    def uid(self) -> str:
        """Unique ID of the `~bsl.lsl.StreamOutlet` instance.

        This ID is guaranteed to be different across multiple instantiations of
        the same ~bsl.lsl.StreamOutlet`, e.g. after a re-start.
        """
        return lib.lsl_get_uid(self.obj).decode("utf-8")

    @property
    def version(self) -> int:
        """Version of the binary LSL library.

        The major version is version // 100.
        The minor version is version % 100.
        """
        # TODO: Check why this is not returning the same version as the lib..
        return lib.lsl_get_version(self.obj)

    # -- Data description -----------------------------------------------------
    @property
    def as_xml(self) -> str:
        """Retrieve the entire stream_info in XML format.

        This yields an XML document (in string form) whose top-level element is
        <info>. The info element contains one element for each field of the
        `~bsl.lsl.StreamInfo` class, including:
        - the core elements <name>, <type>, <channel_count>, <nominal_srate>,
          <channel_format>, <source_id>
        - the misc elements <version>, <created_at>, <uid>, <session_id>,
          <v4address>, <v4data_port>, <v4service_port>, <v6address>,
          <v6data_port>, <v6service_port>
        - the extended description element <desc> with user-defined
          sub-elements.
        """
        return lib.lsl_get_xml(self.obj).decode("utf-8")

    @property
    def desc(self) -> XMLElement:
        """ "Extended description of the stream.

        It is highly recommended that at least the channel labels are described
        here. See code examples on the LSL wiki. Other information, such
        as amplifier settings, measurement units if deviating from defaults,
        setup information, subject information, etc.. can be specified here, as
        well. Meta-data recommendations follow the XDF file format project
        (github.com/sccn/xdf/wiki/Meta-Data or web search for: XDF meta-data).

        Important: if you use a stream content type for which meta-data
        recommendations exist, please try to lay out your meta-data in
        agreement with these recommendations for compatibility with other
        applications.
        """
        return XMLElement(lib.lsl_get_desc(self.obj))


class StreamInfo(_BaseStreamInfo):
    """Base Stream information object, storing the declaration of a stream.

    A StreamInfo contains the following information:
    - Core information (name, number of channels, sampling frequency, channel
      format, ...)
    - Optional metadata about the stream content (channel labels, measurement
      units, ...)
    - Hosting information (uID, hostname, ...) if bound to an
      `~bsl.lsl.StreamInlet` or `~bsl.lsl.StreamOutlet`

    Parameters
    ----------
    name : str
        Name of the stream. This field can not be empty.
    stype : str
        Content type of the stream, e.g. ``"EEG"`` or ``"Gaze"``. If a stream
        contains mixed content, this value should be empty and the description
        of each channel should include its type.
    n_channels : int ``≥ 1``
        Also called ``channel_count``, represents the number of channels per
        sample. This number stays constant for the lifetime of the stream.
    sfreq : float ``≥ 0``
        Also called ``nominal_srate``, represents the sampling rate (in Hz) as
        advertised by the data source. If the sampling rate is irregular (e.g.
        for a trigger stream), the sampling rate is set to ``0``.
    channel_format : str
        Format of each channel. If your channels have different formats,
        consider supplying multiple streams or use the largest type that can
        hold them all.
        One of ``('string', 'float32', 'float64', 'int8', 'int16', 'int32')``.
        ``'int64'`` is partially supported.
    source_id : str
        A unique identifier of the device or source of the data. If not empty,
        this information improves the system robustness since it allows
        recipients to recover from failure by finding a stream with the same
        ``source_id`` on the network.
    """

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
