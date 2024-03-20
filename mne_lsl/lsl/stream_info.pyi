from typing import Any, Optional, Union

from _typeshed import Incomplete
from mne import Info, Projection
from mne.io._digitization import DigPoint
from numpy.typing import DTypeLike as DTypeLike

from .._typing import ScalarIntArray as ScalarIntArray
from ..utils._checks import check_type as check_type
from ..utils._checks import check_value as check_value
from ..utils._checks import ensure_int as ensure_int
from ..utils.logs import logger as logger
from ..utils.meas_info import create_info as create_info
from ._utils import XMLElement as XMLElement
from .constants import fmt2idx as fmt2idx
from .constants import fmt2numpy as fmt2numpy
from .constants import idx2fmt as idx2fmt
from .constants import numpy2fmt as numpy2fmt
from .constants import string2fmt as string2fmt
from .load_liblsl import lib as lib

_MAPPING_LSL: Incomplete
_LOC_NAMES: Incomplete

class _BaseStreamInfo:
    """Base Stream information object, storing the declaration of a stream.

    A StreamInfo contains the following information:

    * Core information (name, number of channels, sampling frequency, channel format,
      ...)
    * Optional metadata about the stream content (channel labels, measurement units,
      ...)
    * Hosting information (uID, hostname, ...) if bound to an inlet or outlet
    """

    _obj: Incomplete
    _dtype: Incomplete

    def __init__(self, obj) -> None: ...
    def __del__(self) -> None:
        """Destroy a `~mne_lsl.lsl.StreamInfo`."""

    def __eq__(self, other: Any) -> bool:
        """Equality == method."""

    def __ne__(self, other: Any) -> bool:
        """Inequality != method."""

    def __hash__(self) -> int:
        """Determine a hash from the properties."""

    def __repr__(self) -> str:
        """Representation of the Info."""

    @property
    def dtype(self) -> Union[str, DTypeLike]:
        """Channel format of a stream.

        All channels in a stream have the same format.

        :type: :class:`~numpy.dtype` | ``"string"``
        """

    @property
    def name(self) -> str:
        """Name of the stream.

        The name of the stream is defined by the application creating the LSL outlet.
        Streams with identical names can coexist, at the cost of ambiguity for the
        recording application and/or the experimenter.

        :type: :class:`str`
        """

    @property
    def n_channels(self) -> int:
        """Number of channels.

        A stream must have at least one channel. The number of channels remains constant
        for all samples.

        :type: :class:`int`
        """

    @property
    def sfreq(self) -> float:
        """Sampling rate of the stream, according to the source (in Hz).

        If a stream is irregularly sampled, the sampling rate is set to ``0``.

        :type: :class:`float`
        """

    @property
    def source_id(self) -> str:
        """Unique identifier of the stream's source.

        The unique source (or device) identifier is an optional piece of information
        that, if available, allows endpoints (such as the recording program) to
        re-acquire a stream automatically once if it came back online.

        :type: :class:`str`
        """

    @property
    def stype(self) -> str:
        """Type of the stream.

        The content type is a short string, such as ``"EEG"``, ``"Gaze"``, ... which
        describes the content carried by the channel. If a stream contains mixed
        content, this value should be an empty string and the type should be stored in
        the description of individual channels.

        :type: :class:`str`
        """

    @property
    def created_at(self) -> float:
        """Timestamp at which the stream was created.

        This is the time stamps at which the stream was first created, as determined by
        :func:`mne_lsl.lsl.local_clock` on the providing machine.

        :type: :class:`float`
        """

    @property
    def hostname(self) -> str:
        """Hostname of the providing machine.

        :type: :class:`str`
        """

    @property
    def session_id(self) -> str:
        """Session ID for the given stream.

        The session ID is an optional human-assigned identifier of the recording
        session. While it is rarely used, it can be used to prevent concurrent recording
        activities on the same sub-network (e.g., in multiple experiment areas) from
        seeing each other's streams (can be assigned in a configuration file read by
        liblsl, see also Network Connectivity in the LSL wiki).

        :type: :class:`str`
        """

    @property
    def uid(self) -> str:
        """Unique ID of the :class:`~mne_lsl.lsl.StreamOutlet` instance.

        This ID is guaranteed to be different across multiple instantiations of the same
        :class:`~mne_lsl.lsl.StreamOutlet`, e.g. after a re-start.

        :type: :class:`str`
        """

    @property
    def protocol_version(self) -> int:
        """Version of the LSL protocol.

        The major version is ``version // 100``.
        The minor version is ``version % 100``.

        :type: :class:`int`
        """

    @property
    def as_xml(self) -> str:
        """Retrieve the entire stream_info in XML format.

        This yields an XML document (in string form) whose top-level element is
        ``<info>``. The info element contains one element for each field of the
        :class:`~mne_lsl.lsl.StreamInfo` class, including:

        * the core elements ``name``, ``type`` (eq. ``stype``), ``channel_count``
          (eq. ``n_channels``), ``nominal_srate`` (eq. ``sfreq``), ``channel_format``
          (eq. ``dtype``), ``source_id``
        * the misc elements ``version``, ``created_at``, ``uid``, ``session_id``,
          ``v4address``, ``v4data_port``, ``v4service_port``, ``v6address``,
          ``v6data_port``, ``v6service_port``
        * the extended description element ``desc`` with user-defined sub-elements.

        :type: :class:`str`
        """

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

    def get_channel_info(self) -> Info:
        """Get the FIFF measurement :class:`~mne.Info` in the description.

        Returns
        -------
        info : Info
            :class:`~mne.Info` containing the measurement information.
        """

    def get_channel_names(self) -> Optional[list[str]]:
        """Get the channel names in the description.

        Returns
        -------
        ch_names : list of str or ``None`` | None
            List of channel names, matching the number of total channels.
            If ``None``, the channel names are not set.

            .. warning::

                If a list of str and ``None`` are returned, some of the channel names
                are missing. This is not expected and could occur if the XML tree in
                the ``desc`` property is tempered with outside of the defined getter and
                setter.
        """

    def get_channel_types(self) -> Optional[list[str]]:
        """Get the channel types in the description.

        Returns
        -------
        ch_types : list of str or ``None`` | None
            List of channel types, matching the number of total channels.
            If ``None``, the channel types are not set.

            .. warning::

                If a list of str and ``None`` are returned, some of the channel types
                are missing. This is not expected and could occur if the XML tree in
                the ``desc`` property is tempered with outside of the defined getter and
                setter.
        """

    def get_channel_units(self) -> Optional[list[str]]:
        """Get the channel units in the description.

        Returns
        -------
        ch_units : list of str or ``None`` | None
            List of channel units, matching the number of total channels.
            If ``None``, the channel units are not set.

            .. warning::

                If a list of str and ``None`` are returned, some of the channel units
                are missing. This is not expected and could occur if the XML tree in
                the ``desc`` property is tempered with outside of the defined getter and
                setter.
        """

    def _get_channel_info(self, name: str) -> Optional[list[str]]:
        """Get the 'channel/name' element in the XML tree."""

    def _get_channel_projectors(self) -> list[Projection]:
        """Get the SSP vectors in the XML tree."""

    def _get_digitization(self) -> list[DigPoint]:
        """Get the digitization in the XML tree."""

    def set_channel_info(self, info: Info) -> None:
        """Set the channel info from a FIFF measurement :class:`~mne.Info`.

        Parameters
        ----------
        info : Info
            :class:`~mne.Info` containing the measurement information.
        """

    def set_channel_names(self, ch_names: Union[list[str], tuple[str]]) -> None:
        """Set the channel names in the description. Existing labels are overwritten.

        Parameters
        ----------
        ch_names : list of str
            List of channel names, matching the number of total channels.
        """

    def set_channel_types(self, ch_types: Union[str, list[str]]) -> None:
        """Set the channel types in the description. Existing types are overwritten.

        The types are given as human readable strings, e.g. ``'eeg'``.

        Parameters
        ----------
        ch_types : list of str | str
            List of channel types, matching the number of total channels.
            If a single :class:`str` is provided, the type is applied to all channels.
        """

    def set_channel_units(
        self, ch_units: Union[str, list[str], int, list[int], ScalarIntArray]
    ) -> None:
        """Set the channel units in the description. Existing units are overwritten.

        The units are given as human readable strings, e.g. ``'microvolts'``, or as
        multiplication factor, e.g. ``-6`` for ``1e-6`` thus converting e.g. Volts to
        microvolts.

        Parameters
        ----------
        ch_units : list of str | list of int | array of int | str | int
            List of channel units, matching the number of total channels.
            If a single :class:`str` or :class:`int` is provided, the unit is applied to
            all channels.

        Notes
        -----
        Some channel types doch_units not have a unit. The :class:`str` ``none`` or the
        :class:`int` 0 should be used to denote this channel unit, corresponding to
        ``FIFF_UNITM_NONE`` in MNE-Python.
        """

    def _set_channel_info(self, ch_infos: list[str], name: str) -> None:
        """Set the 'channel/name' element in the XML tree."""

    def _set_channel_projectors(self, projs: list[Projection]) -> None:
        """Set the SSP projector."""

    def _set_digitization(self, dig_points: list[DigPoint]) -> None:
        """Set the digitization points."""

    @staticmethod
    def _add_first_node(desc: XMLElement, name: str) -> XMLElement:
        """Add the first node in the description and return it."""

    @staticmethod
    def _prune_description_node(node: XMLElement, parent: XMLElement) -> None:
        """Prune a node and remove outdated entries."""

    @staticmethod
    def _set_description_node(node: XMLElement, mapping: dict[str, Any]) -> None:
        """Set the key: value child(s) of a node."""

    @staticmethod
    def _get_fiff_int_named(
        value: Optional[str], name: str, mapping: dict[int, int]
    ) -> Optional[int]:
        """Try to retrieve the FIFF integer code from the str representation."""

class StreamInfo(_BaseStreamInfo):
    """Base Stream information object, storing the declaration of a stream.

    A StreamInfo contains the following information:

    * Core information (name, number of channels, sampling frequency, channel format,
      ...).
    * Optional metadata about the stream content (channel labels, measurement units,
      ...).
    * Hosting information (uID, hostname, ...) if bound to a
      :class:`~mne_lsl.lsl.StreamInlet` or :class:`~mne_lsl.lsl.StreamOutlet`.

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
    dtype : str | dtype
        Format of each channel. If your channels have different formats, consider
        supplying multiple streams or use the largest type that can hold them all.
        One of ``('string', 'float32', 'float64', 'int8', 'int16', 'int32')``.
        ``'int64'`` is partially supported. Can also be the equivalent numpy type, e.g.
        ``np.int8``.
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
    ) -> None: ...
    @staticmethod
    def _dtype2idxfmt(dtype: Union[str, int, DTypeLike]) -> int:
        """Convert a string format to its LSL integer value."""
