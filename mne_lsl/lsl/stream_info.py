from __future__ import annotations  # c.f. PEP 563, PEP 649

from ctypes import c_char_p, c_double, c_void_p
from typing import TYPE_CHECKING

import numpy as np

from ..utils._checks import check_type, check_value, ensure_int
from ..utils.logs import logger
from .constants import fmt2idx, fmt2numpy, idx2fmt, numpy2fmt, string2fmt
from .load_liblsl import lib
from .utils import XMLElement

if TYPE_CHECKING:
    from typing import Any, List, Optional, Tuple, Union

    from mne import Info, Projection
    from mne.utils import check_version
    from numpy.typing import DTypeLike, NDArray

    if check_version("mne", "1.6"):
        from mne._fiff._digitization import DigPoint
    else:
        from mne.io._digitization import DigPoint


_MAPPING_LSL = {
    "ch_name": "label",
    "ch_type": "type",
    "ch_unit": "unit",
}


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
        """Destroy a `~mne_lsl.lsl.StreamInfo`."""
        try:
            lib.lsl_destroy_streaminfo(self._obj)
        except Exception:
            pass

    def __eq__(self, other: Any) -> bool:
        """Equality == method."""
        if not isinstance(other, _BaseStreamInfo):
            return False
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
    def dtype(self) -> Union[str, DTypeLike]:
        """Channel format of a stream.

        All channels in a stream have the same format.

        :type: :class:`~numpy.dtype` | ``"string"``
        """
        return fmt2numpy.get(self._dtype, "string")

    @property
    def name(self) -> str:
        """Name of the stream.

        The name of the stream is defined by the application creating the LSL outlet.
        Streams with identical names can coexist, at the cost of ambiguity for the
        recording application and/or the experimenter.

        :type: :class:`str`
        """
        return lib.lsl_get_name(self._obj).decode("utf-8")

    @property
    def n_channels(self) -> int:
        """Number of channels.

        A stream must have at least one channel. The number of channels remains constant
        for all samples.

        :type: :class:`int`
        """
        return lib.lsl_get_channel_count(self._obj)

    @property
    def sfreq(self) -> float:
        """Sampling rate of the stream, according to the source (in Hz).

        If a stream is irregularly sampled, the sampling rate is set to ``0``.

        :type: :class:`float`
        """
        return lib.lsl_get_nominal_srate(self._obj)

    @property
    def source_id(self) -> str:
        """Unique identifier of the stream's source.

        The unique source (or device) identifier is an optional piece of information
        that, if available, allows endpoints (such as the recording program) to
        re-acquire a stream automatically once if it came back online.

        :type: :class:`str`
        """
        return lib.lsl_get_source_id(self._obj).decode("utf-8")

    @property
    def stype(self) -> str:
        """Type of the stream.

        The content type is a short string, such as ``"EEG"``, ``"Gaze"``, ... which
        describes the content carried by the channel. If a stream contains mixed
        content, this value should be an empty string and the type should be stored in
        the description of individual channels.

        :type: :class:`str`
        """
        return lib.lsl_get_type(self._obj).decode("utf-8")

    # -- Hosting information, assigned when bound to an outlet/inlet -------------------
    @property
    def created_at(self) -> float:
        """Timestamp at which the stream was created.

        This is the time stamps at which the stream was first created, as determined by
        :func:`mne_lsl.lsl.local_clock` on the providing machine.

        :type: :class:`float`
        """
        return lib.lsl_get_created_at(self._obj)

    @property
    def hostname(self) -> str:
        """Hostname of the providing machine.

        :type: :class:`str`
        """
        return lib.lsl_get_hostname(self._obj).decode("utf-8")

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
        return lib.lsl_get_session_id(self._obj).decode("utf-8")

    @property
    def uid(self) -> str:
        """Unique ID of the `~mne_lsl.lsl.StreamOutlet` instance.

        This ID is guaranteed to be different across multiple instantiations of the same
        :class:`~mne_lsl.lsl.StreamOutlet`, e.g. after a re-start.

        :type: :class:`str`
        """
        return lib.lsl_get_uid(self._obj).decode("utf-8")

    @property
    def protocol_version(self) -> int:
        """Version of the LSL protocol.

        The major version is ``version // 100``.
        The minor version is ``version % 100``.

        :type: :class:`int`
        """
        return lib.lsl_get_version(self._obj)

    # -- Data description --------------------------------------------------------------
    @property
    def as_xml(self) -> str:
        """Retrieve the entire stream_info in XML format.

        This yields an XML document (in string form) whose top-level element is
        ``<info>``. The info element contains one element for each field of the
        `~mne_lsl.lsl.StreamInfo` class, including:

        * the core elements ``name``, ``type`` (eq. ``stype``), ``channel_count``
          (eq. ``n_channels``), ``nominal_srate`` (eq. ``sfreq``), ``channel_format``
          (eq. ``dtype``), ``source_id``
        * the misc elements ``version``, ``created_at``, ``uid``, ``session_id``,
          ``v4address``, ``v4data_port``, ``v4service_port``, ``v6address``,
          ``v6data_port``, ``v6service_port``
        * the extended description element ``desc`` with user-defined sub-elements.

        :type: :class:`str`
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

    # -- Getters and setters for data description --------------------------------------
    def get_channel_info(self, info: Info) -> None:
        """Get the FIFF measurement :class:`~mne.Info` in the description.

        Returns
        -------
        info : Info
            :class:`~mne.Info` containing the measurement information.
        """
        raise NotImplementedError

    def get_channel_names(self) -> Optional[List[str]]:
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
        return self._get_channel_info("ch_name")

    def get_channel_types(self) -> Optional[List[str]]:
        """Get the channel types in the description.

        Returns
        -------
        ch_types : list of str or ``None`` | None
            List of channel names, matching the number of total channels.
            If ``None``, the channel types are not set.

            .. warning::

                If a list of str and ``None`` are returned, some of the channel types
                are missing. This is not expected and could occur if the XML tree in
                the ``desc`` property is tempered with outside of the defined getter and
                setter.
        """
        return self._get_channel_info("ch_type")

    def get_channel_units(self) -> Optional[List[str]]:
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
        return self._get_channel_info("ch_unit")

    def _get_channel_info(self, name: str) -> Optional[List[str]]:
        """Get the 'channel/name' element in the XML tree."""
        if self.desc.child("channels").empty():
            return None
        name = _MAPPING_LSL.get(name, name)

        channels = self.desc.child("channels")
        ch_infos = list()
        ch = channels.child("channel")
        while not ch.empty():
            ch_info = ch.child(name).first_child().value()
            if len(ch_info) != 0:
                ch_infos.append(ch_info)
            else:
                ch_infos.append(None)
            ch = ch.next_sibling()

        if all(ch_info is None for ch_info in ch_infos):
            return None
        if len(ch_infos) != self.n_channels:
            logger.warning(
                "The stream description contains %i elements for %i channels.",
                len(ch_infos),
                self.n_channels,
            )
        return ch_infos

    def set_channel_info(self, info: Info) -> None:
        """Set the channel info from a FIFF measurement :class:`~mne.Info`.

        Parameters
        ----------
        info : Info
            :class:`~mne.Info` containing the measurement information.
        """
        self.set_channel_names(info["ch_names"])
        self.set_channel_types(info.get_channel_types(unique=False))
        self.set_channel_units([ch["unit_mul"] for ch in info["chs"]])
        # integer codes
        for ch_info in ("kind", "coil_type", "coord_frame"):
            self._set_channel_info(
                [str(int(ch[ch_info])) for ch in info["chs"]], ch_info
            )
        # floats, range and cal are multiplied together here because since they are
        # small, it's best to handle the floating point multiplication before
        # transmission.
        self._set_channel_info(
            [str(ch["range"] * ch["cal"]) for ch in info["chs"]], "range_cal"
        )

        # channel location
        # fmt: off
        loc_names = (
            "R0x", "R0y", "R0z",
            "Exx", "Exy", "Exz",
            "Eyx", "Eyy", "Eyz",
            "Ezx", "Ezy", "Ezz",
        )
        # fmt: on
        assert not self.desc.child("channels").empty()  # sanity-check
        channels = self.desc.child("channels")
        ch = channels.child("channel")
        for ch_info in info["chs"]:
            loc = ch.child("loc")
            if loc.empty():
                loc = ch.append_child("loc")
            for key, value in zip(loc_names, ch_info["loc"]):
                if loc.child(key).empty():
                    loc.append_child_value(key, str(value))
                else:
                    loc.child(key).first_child / ().set_value(str(value))
            ch = ch.next_sibling()
        assert ch.empty()  # sanity-check

        # non-channel variables
        if self.desc.child("filters").empty():
            filters = self.desc.append_child("filters")
        else:
            filters = self.desc.child("filters")
        for filt in ("highpass", "lowpass"):
            if filters.child(filt).empty():
                filters.append_child_value(filt, str(info[filt]))
            else:
                filters.child(filt).first_child().set_value(str(info[filt]))

        # projectors and digitization
        self._set_channel_projectors(info["projs"])
        self._set_digitization(info["dig"])

    def set_channel_names(self, ch_names: Union[List[str], Tuple[str]]) -> None:
        """Set the channel names in the description. Existing labels are overwritten.

        Parameters
        ----------
        ch_names : list of str
            List of channel names, matching the number of total channels.
        """
        self._set_channel_info(ch_names, "ch_name")

    def set_channel_types(self, ch_types: Union[str, List[str]]) -> None:
        """Set the channel types in the description. Existing types are overwritten.

        The types are given as human readable strings, e.g. ``'eeg'``.

        Parameters
        ----------
        ch_types : list of str | str
            List of channel types, matching the number of total channels.
            If a single `str` is provided, the type is applied to all channels.
        """
        ch_types = (
            [ch_types] * self.n_channels if isinstance(ch_types, str) else ch_types
        )
        self._set_channel_info(ch_types, "ch_type")

    def set_channel_units(
        self, ch_units: Union[str, List[str], int, List[int], NDArray[int]]
    ) -> None:
        """Set the channel units in the description. Existing units are overwritten.

        The units are given as human readable strings, e.g. ``'microvolts'``, or as
        multiplication factor, e.g. ``-6`` for ``1e-6`` thus converting e.g. Volts to
        microvolts.

        Parameters
        ----------
        ch_units : list of str | list of int | array of int | str | int
            List of channel units, matching the number of total channels.
            If a single `str` or `int` is provided, the unit is applied to all channels.

        Notes
        -----
        Some channel types do not have a unit. The `str` ``none`` or the `int` 0 should
        be used to denote this channel unit, corresponding to ``FIFF_UNITM_NONE`` in
        MNE.
        """
        check_type(ch_units, (list, tuple, np.ndarray, str, "int-like"), "ch_units")
        if isinstance(ch_units, int):
            ch_units = [
                ensure_int(ch_unit, "ch_unit") for ch_unit in ch_units
            ] * self.n_channels
        elif isinstance(ch_units, str):
            ch_units = [ch_units] * self.n_channels
        else:
            if isinstance(ch_units, np.ndarray):
                if ch_units.ndim != 1:
                    raise ValueError(
                        "The channel units can be provided as a 1D array of integers. "
                        f"The provided array has {ch_units.ndim} dimension and is "
                        "invalid."
                    )
                ch_units = [ensure_int(ch_unit, "ch_unit") for ch_unit in ch_units]
            ch_units = [
                str(int(ch_unit)) if isinstance(ch_unit, int) else ch_unit
                for ch_unit in ch_units
            ]
        self._set_channel_info(ch_units, "ch_unit")

    def _set_channel_info(self, ch_infos: List[str], name: str) -> None:
        """Set the 'channel/name' element in the XML tree."""
        check_type(ch_infos, (list, tuple), name)
        for ch_info in ch_infos:
            check_type(ch_info, (str,), name.rstrip("s"))
        if len(ch_infos) != self.n_channels:
            raise ValueError(
                f"The number of provided channel {name.lstrip('ch_')} {len(ch_infos)} "
                f"must match the number of channels {self.n_channels}."
            )
        name = _MAPPING_LSL.get(name, name)

        if self.desc.child("channels").empty():
            channels = self.desc.append_child("channels")
        else:
            channels = self.desc.child("channels")

        # fill the 'channel/name' element of the tree and overwrite existing values
        ch = channels.child("channel")
        for ch_info in ch_infos:
            if ch.empty():
                ch = channels.append_child("channel")

            if ch.child(name).empty():
                ch.append_child_value(name, ch_info)
            else:
                ch.child(name).first_child().set_value(ch_info)
            ch = ch.next_sibling()

        # in case the original sinfo was tempered with and had more 'channel' than the
        # correct number of channels
        while not ch.empty():
            ch_next = ch.next_sibling()
            channels.remove_child(ch)
            ch = ch_next

    def _set_channel_projectors(self, projs: List[Projection]) -> None:
        """Set the SSP projector."""
        if self.desc.child("projectors").empty():
            projectors = self.desc.append_child("projectors")
        else:
            projectors = self.desc.child("projectors")

        # fill the 'channel/name' element of the tree and overwrite existing values
        projector = projectors.child("projector")
        for proj in projs:
            if projector.empty():
                projector = projectors.append_child("projector")

            if projector.child("desc").empty():
                projector.append_child_value("desc", proj["desc"])
            else:
                projector.child("desc").first_child().set_value(proj["desc"])

            value = str(int(proj["kind"]))
            if projector.child("kind").empty():
                projector.append_child_value("kind", value)
            else:
                projector.child("kind").first_child().set_value(value)

            # data = projector.child("data")
            # if data.empty():
            #     data = projector.append_child("data")

            projector = projector.next_sibling()

        # in case the original sinfo was tempered with and had more 'channel' than the
        # correct number of channels
        while not projector.empty():
            projector_next = projector.next_sibling()
            projectors.remove_child(projector)
            projector = projector_next



        # # fill the 'projector' element of the tree and overwrite existing integer codes
        # projector = projectors.child("projector")
        # for proj in projs:
        #     if projector.empty():
        #         projector = projectors.append_child("projector")

        #     for key in ("kind", "desc"):
        #         value = (
        #             str(int(proj[key]))
        #             if isinstance(proj[key], int)
        #             else str(proj[key])
        #         )
        #         if projector.child(key).empty():
        #             projector.append_child_value(key, value)
        #         else:
        #             projector.child(key).first_child().set_value(value)

        #     ch = projector.child("channel")
        #     for ch_name, ch_data in zip(
        #         proj["data"]["col_names"], np.squeeze(proj["data"]["data"])
        #     ):
        #         if ch.empty():
        #             ch = projector.append_child("channel")
        #         if ch.child("label").empty():
        #             ch.append_child_value("label", ch_name)
        #         else:
        #             ch.child("label").first_child().set_value(ch_name)
        #         if ch.child("data").empty():
        #             ch.append_child_value("data", str(ch_data))
        #         else:
        #             ch.child("data").first_child().set_value(str(ch_data))
        #         ch = ch.next_sibling()

        #     # in case the original sinfo was tempered with and had more 'channel'
        #     # than the correct number of channels
        #     while not ch.empty():
        #         ch_next = ch.next_sibling()
        #         projector.remove_child(ch)
        #         ch = ch_next
        #     projector.next_sibling()

        # while not projector.empty():
        #     projector_next = projector.next_sibling()
        #     projectors.remove(projector)
        #     projector = projector_next

    def _set_digitization(self, dig_points: List[DigPoint]) -> None:
        """Set the digitization points."""
        check_type(dig_points, (list,), "dig_points")
        if self.desc.child("dig").empty():
            dig = self.desc.append_child("dig")
        else:
            dig = self.desc.child("dig")

        # fill the 'point' element of the tree and overwrite existing integer codes
        point = dig.child("point")
        for dig_point in dig_points:
            if point.empty():
                point = dig.append_child("point")

            for key in ("kind", "ident"):
                value = str(int(dig_point[key]))
                if point.child(key).empty():
                    point.append_child_value(key, value)
                else:
                    point.child(key).first_child().set_value(value)

            loc = point.child("loc")
            if loc.empty():
                loc = point.append_child("loc")
            for key, value in zip(("X", "Y", "Z"), dig_point["r"]):
                if loc.child(key).empty():
                    loc.append_child_value(key, str(value))
                else:
                    loc.child(key).first_child / ().set_value(str(value))
            point = point.next_sibling()

        # in case the original sinfo was tempered with and had more 'point' than the
        # correct number of points
        while not point.empty():
            point_next = point.next_sibling()
            dig.remove_child(point)
            point = point_next


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
        if sfreq < 0:
            raise ValueError(
                "The sampling frequency 'sfreq' must be a positive number. "
                f"{sfreq} is invalid."
            )
        check_type(source_id, (str,), "source_id")

        obj = lib.lsl_create_streaminfo(
            c_char_p(str.encode(name)),
            c_char_p(str.encode(stype)),
            n_channels,
            c_double(sfreq),
            StreamInfo._dtype2idxfmt(dtype),
            c_char_p(str.encode(source_id)),
        )
        super().__init__(obj)

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _dtype2idxfmt(dtype: Union[str, int, DTypeLike]) -> int:
        """Convert a string format to its LSL integer value."""
        if dtype in fmt2idx:
            return fmt2idx[dtype]
        elif dtype in numpy2fmt:
            return fmt2idx[numpy2fmt[dtype]]
        elif isinstance(dtype, str):
            dtype = dtype.lower()
            check_value(dtype, string2fmt, "dtype")
            dtype = fmt2idx[string2fmt[dtype]]
        elif isinstance(dtype, int):
            dtype = ensure_int(dtype)
            check_value(dtype, idx2fmt, "dtype")
        else:
            raise ValueError(
                "The provided dtype could not be interpreted as a supported type. "
                f"{dtype} is invalid."
            )
        return dtype
