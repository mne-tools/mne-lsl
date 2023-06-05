# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from ctypes import c_char_p, c_double, c_void_p
from typing import TYPE_CHECKING

from ..utils._checks import check_type, check_value, ensure_int
from .constants import fmt2idx, fmt2string, idx2fmt, string2fmt
from .load_liblsl import lib
from .utils import XMLElement

if TYPE_CHECKING:
    from typing import Any, List, Optional, Tuple, Union


class _BaseStreamInfo:
    """Base Stream information object, storing the declaration of a stream.

    A StreamInfo contains the following information:

    * Core information (name, number of channels, sampling frequency, channel format,
      ...)
    * Optional metadata about the stream content (channel labels, measurement units,
      ...)
    * Hosting information (uID, hostname, ...) if bound to an inlet or outlet
    """

    def __init__(self, obj):
        self._obj = c_void_p(obj)
        if not self._obj:
            raise RuntimeError(
                "The StreamInfo could not be created from the description."
            )
        self._dtype = idx2fmt[lib.lsl_get_channel_format(self._obj)]

    def __del__(self):
        """Destroy a `~bsl.lsl.StreamInfo`."""
        try:
            lib.lsl_destroy_streaminfo(self._obj)
        except Exception:
            pass

    def __eq__(self, other: Any) -> bool:
        """Equality == method.

        Test one by one to stop ASAP.
        """
        # TODO: check if this is not better
        # return isinstance(other, _BaseStreamInfo) and self.__dict__ == other.__dict__
        if self.name != other.name:
            return False
        if self.source_id != other.source_id:
            return False
        if self.stype != other.stype:
            return False
        if self.sfreq != other.sfreq:
            return False
        if self.n_channels != other.n_channels:
            return False
        if self.dtype != other.dtype:
            return False
        return True

    def __ne__(self, other: Any) -> bool:
        """Inequality != method."""
        return not self.__eq__(other)

    def __hash__(self) -> int:
        """Determine a hash from the properties."""
        return hash(
            (
                self.dtype,
                self.name,
                self.n_channels,
                self.sfreq,
                self.source_id,
                self.stype,
            )
        )

    def __repr__(self) -> str:
        """Representation of the Info."""
        str_ = f"< sInfo '{self.name}' >\n"
        stype = self.stype

        if len(stype) != 0:
            add = f"| Type: {stype}\n"
            str_ += add.rjust(2 + len(add))

        sfreq = self.sfreq
        if sfreq == 0:
            add = "| Sampling: Irregular\n"
            str_ += add.rjust(2 + len(add))
        else:
            add = f"| Sampling: {sfreq} Hz\n"
            str_ += add.rjust(2 + len(add))

        add = f"| Number of channels: {self.n_channels}\n"
        str_ += add.rjust(2 + len(add))

        add = f"| Data type: {self.dtype}\n"
        str_ += add.rjust(2 + len(add))

        source_id = self.source_id
        if len(source_id) != 0:
            add = f"| Source: {source_id}\n"
            str_ += add.rjust(2 + len(add))

        return str_

    # -- Core information, assigned at construction ------------------------------------
    @property
    def dtype(self) -> str:
        """Channel format of a stream.

        All channels in a stream have the same format.
        """
        return fmt2string[self._dtype]

    @property
    def name(self) -> str:
        """Name of the stream.

        The name of the stream is defined by the application creating the LSL outlet.
        Streams with identical names can coexist, at the cost of ambiguity for the
        recording application and/or the experimenter.
        """
        return lib.lsl_get_name(self._obj).decode("utf-8")

    @property
    def n_channels(self) -> int:
        """Number of channels.

        A stream must have at least one channel. The number of channels remains constant
        for all samples.
        """
        return lib.lsl_get_channel_count(self._obj)

    @property
    def sfreq(self) -> float:
        """Sampling rate of the stream, according to the source (in Hz).

        If a stream is irregularly sampled, the sampling rate is set to ``0``.
        """
        return lib.lsl_get_nominal_srate(self._obj)

    @property
    def source_id(self) -> str:
        """Unique identifier of the stream's source.

        The unique source (or device) identifier is an optional piece of information
        that, if available, allows endpoints (such as the recording program) to
        re-acquire a stream automatically once if it came back online.
        """
        return lib.lsl_get_source_id(self._obj).decode("utf-8")

    @property
    def stype(self) -> str:
        """Type of the stream.

        The content type is a short string, such as ``"EEG"``, ``"Gaze"``, ... which
        describes the content carried by the channel. If a stream contains mixed
        content, this value should be an empty string and the type should be stored in
        the description of individual channels.
        """
        return lib.lsl_get_type(self._obj).decode("utf-8")

    # -- Hosting information, assigned when bound to an outlet/inlet -------------------
    @property
    def created_at(self) -> float:
        """Timestamp at which the stream was created.

        This is the time stamps at which the stream was first created, as determined by
        `~bsl.lsl.local_clock` on the providing machine.
        """
        return lib.lsl_get_created_at(self._obj)

    @property
    def hostname(self) -> str:
        """Hostname of the providing machine."""
        return lib.lsl_get_hostname(self._obj).decode("utf-8")

    @property
    def session_id(self) -> str:
        """Session ID for the given stream.

        The session ID is an optional human-assigned identifier of the recording
        session. While it is rarely used, it can be used to prevent concurrent recording
        activities on the same sub-network (e.g., in multiple experiment areas) from
        seeing each other's streams (can be assigned in a configuration file read by
        liblsl, see also Network Connectivity in the LSL wiki).
        """
        return lib.lsl_get_session_id(self._obj).decode("utf-8")

    @property
    def uid(self) -> str:
        """Unique ID of the `~bsl.lsl.StreamOutlet` instance.

        This ID is guaranteed to be different across multiple instantiations of the same
        `~bsl.lsl.StreamOutlet`, e.g. after a re-start.
        """
        return lib.lsl_get_uid(self._obj).decode("utf-8")

    @property
    def protocol_version(self) -> int:
        """Version of the LSL protocol.

        The major version is ``version // 100``.
        The minor version is ``version % 100``.
        """
        return lib.lsl_get_version(self._obj)

    # -- Data description --------------------------------------------------------------
    @property
    def as_xml(self) -> str:
        """Retrieve the entire stream_info in XML format.

        This yields an XML document (in string form) whose top-level element is <info>.
        The info element contains one element for each field of the
        `~bsl.lsl.StreamInfo` class, including:

        * the core elements ``name``, ``type`` (eq. ``stype``), ``channel_count``
          (eq. ``n_channels``), ``nominal_srate`` (eq. ``sfreq``), ``channel_format``
          (eq. ``dtype``), ``source_id``
        * the misc elements ``version``, ``created_at``, ``uid``, ``session_id``,
          ``v4address``, ``v4data_port``, ``v4service_port``, ``v6address``,
          ``v6data_port``, ``v6service_port``
        * the extended description element ``desc`` with user-defined sub-elements.
        """
        return lib.lsl_get_xml(self._obj).decode("utf-8")

    @property
    def desc(self) -> XMLElement:
        """Extended description of the stream.

        It is highly recommended that at least the channel labels are described here.
        See code examples on the LSL wiki. Other information, such as amplifier
        settings, measurement units if deviating from defaults, setup information,
        subject information, etc.. can be specified here, as well. Meta-data
        recommendations follow the `XDF file format project`_.

        Important: if you use a stream content type for which meta-data recommendations
        exist, please try to lay out your meta-data in agreement with these
        recommendations for compatibility with other applications.

        .. _XDF file format project: https://github.com/sccn/xdf/wiki/Meta-Data
        """
        return XMLElement(lib.lsl_get_desc(self._obj))

    # -- Getter and setters for data description ---------------------------------------
    def get_channel_names(self) -> Optional[List[str]]:
        pass

    def get_channel_types(self) -> Optional[List[str]]:
        pass

    def get_channel_units(self) -> Optional[List[str]]:
        pass

    def set_channel_names(self, ch_names: Union[List[str], Tuple[str]]):
        """Set the channel names in the description. Existing labels are overwritten.

        Parameters
        ----------
        ch_names : sequence of str
            List of channel names, matching the number of total channels.
        """
        check_type(ch_names, (list, tuple), "ch_names")
        for ch_name in ch_names:
            check_type(ch_name, (str,), "ch_name")
        if len(ch_names) != self.n_channels:
            raise ValueError(
                f"The number of provided channel names {len(ch_names)} must match the "
                f"number of channels {self.n_channels}."
            )

        channels = self.desc.append_child("channels")
        for ch_name in ch_names:
            ch = channels.append_child("channel")
            ch.append_child_value("label", ch_name)

    def set_channel_types(self, ch_types: List[str]):
        """Set the channel types in the description. Existing types are overwritten.

        Parameters
        ----------
        ch_types : sequence of str
            List of channel types, matching the number of total channels.
        """
        pass

    def set_channel_units(self, ch_units: List[str]):
        """Set the channel units in the description. Existing units are overwritten.

        Parameters
        ----------
        ch_units : sequence of str
            List of channel units, matching the number of total channels.
        """
        pass


class StreamInfo(_BaseStreamInfo):
    """Base Stream information object, storing the declaration of a stream.

    A StreamInfo contains the following information:

    * Core information (name, number of channels, sampling frequency, channel format,
      ...)
    * Optional metadata about the stream content (channel labels, measurement units,
      ...)
    * Hosting information (uID, hostname, ...) if bound to an `~bsl.lsl.StreamInlet` or
      `~bsl.lsl.StreamOutlet`

    Parameters
    ----------
    name : str
        Name of the stream. This field can not be empty.
    stype : str
        Content type of the stream, e.g. ``"EEG"`` or ``"Gaze"``. If a stream contains
        mixed content, this value should be empty and the description of each channel
        should include its type.
    n_channels : int ``≥ 1``
        Also called ``channel_count``, represents the number of channels per sample.
        This number stays constant for the lifetime of the stream.
    sfreq : float ``≥ 0``
        Also called ``nominal_srate``, represents the sampling rate (in Hz) as
        advertised by the data source. If the sampling rate is irregular (e.g. for a
        trigger stream), the sampling rate is set to ``0``.
    dtype : str
        Format of each channel. If your channels have different formats, consider
        supplying multiple streams or use the largest type that can hold them all.
        One of ``('string', 'float32', 'float64', 'int8', 'int16', 'int32')``.
        ``'int64'`` is partially supported.
    source_id : str
        A unique identifier of the device or source of the data. If not empty, this
        information improves the system robustness since it allows recipients to recover
        from failure by finding a stream with the same ``source_id`` on the network.
    """

    def __init__(
        self,
        name: str,
        stype: str,
        n_channels: int,
        sfreq: float,
        dtype: str,
        source_id: str,
    ):
        check_type(name, (str,), "name")
        check_type(stype, (str,), "stype")
        n_channels = ensure_int(n_channels, "n_channels")
        if n_channels <= 0:
            raise ValueError(
                "The number of channels 'n_channels' must be a strictly positive "
                f"integer. {n_channels} is invalid."
            )
        check_type(sfreq, ("numeric",), "sfreq")
        check_type(source_id, (str,), "source_id")

        obj = lib.lsl_create_streaminfo(
            c_char_p(str.encode(name)),
            c_char_p(str.encode(stype)),
            n_channels,
            c_double(sfreq),
            StreamInfo._string2idxfmt(dtype),
            c_char_p(str.encode(source_id)),
        )
        super().__init__(obj)

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _string2idxfmt(dtype) -> int:
        """Convert a string format to its LSL integer value."""
        if dtype in fmt2idx:
            return fmt2idx[dtype]
        check_type(dtype, (str, "int-like"), "dtype")
        if isinstance(dtype, str):
            dtype = dtype.lower()
            check_value(dtype, string2fmt, "dtype")
            dtype = fmt2idx[string2fmt[dtype]]
        else:
            dtype = ensure_int(dtype)
            check_value(dtype, idx2fmt, "dtype")
        return dtype
