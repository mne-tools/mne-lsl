from __future__ import annotations

from ctypes import c_char_p, c_double, c_void_p
from typing import TYPE_CHECKING

import numpy as np
from mne import Info, Projection
from mne.utils import check_version

from ..utils._checks import check_type, check_value, ensure_int
from ..utils.logs import warn
from ..utils.meas_info import create_info
from ._utils import XMLElement
from .constants import fmt2idx, fmt2numpy, idx2fmt, numpy2fmt, string2fmt
from .load_liblsl import lib

if check_version("mne", "1.6"):
    from mne._fiff._digitization import DigPoint
    from mne._fiff.constants import (
        FIFF,
        _ch_coil_type_named,
        _ch_kind_named,
        _coord_frame_named,
        _dig_cardinal_named,
        _dig_kind_named,
    )
else:
    from mne.io._digitization import DigPoint
    from mne.io.constants import (
        FIFF,
        _ch_coil_type_named,
        _ch_kind_named,
        _coord_frame_named,
        _dig_cardinal_named,
        _dig_kind_named,
    )


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import DTypeLike

    from .._typing import ScalarIntArray


_MAPPING_LSL = {
    "ch_name": "label",
    "ch_type": "type",
    "ch_unit": "unit",
}
# fmt: off
_LOC_NAMES = (
    "R0x", "R0y", "R0z",
    "Exx", "Exy", "Exz",
    "Eyx", "Eyy", "Eyz",
    "Ezx", "Ezy", "Ezz",
)
# fmt: on


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
        if not self._obj:  # pragma: no cover
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
    def dtype(self) -> str | DTypeLike:
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
        """Unique ID of the :class:`~mne_lsl.lsl.StreamOutlet` instance.

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
    def get_channel_info(self) -> Info:
        """Get the FIFF measurement :class:`~mne.Info` in the description.

        Returns
        -------
        info : Info
            :class:`~mne.Info` containing the measurement information.
        """
        info = create_info(self.n_channels, self.sfreq, self.stype, self)
        # complete the info object with additional information present from the FIFF
        # standard format.
        kinds = self._get_channel_info("kind")
        coil_types = self._get_channel_info("coil_type")
        coord_frames = self._get_channel_info("coord_frame")
        range_cals = self._get_channel_info("range_cal")

        locs = list()
        channels = self.desc.child("channels")
        ch = channels.child("channel")
        while not ch.empty():
            loc_array = list()
            loc = ch.child("loc")
            for loc_name in _LOC_NAMES:
                try:
                    value = float(loc.child(loc_name).first_child().value())
                except ValueError:
                    value = np.nan
                loc_array.append(value)
            locs.append(loc_array)
            ch = ch.next_sibling()
        locs = (
            np.array([[np.nan] * 12] * self.n_channels)
            if len(locs) == 0
            else np.array(locs)
        )

        with info._unlock(update_redundant=True):
            for var, name, fiff_named in (
                (kinds, "kind", _ch_kind_named),
                (coil_types, "coil_type", _ch_coil_type_named),
                (coord_frames, "coord_frame", _coord_frame_named),
            ):
                if var is None:
                    continue
                for k, value in enumerate(var):
                    value = _BaseStreamInfo._get_fiff_int_named(value, name, fiff_named)
                    if value is not None:
                        info["chs"][k][name] = value

            if range_cals is not None:
                for k, range_cal in enumerate(range_cals):
                    if range_cal is not None:
                        try:
                            info["chs"][k]["range"] = 1.0
                            info["chs"][k]["cal"] = float(range_cal)
                        except ValueError:
                            warn(
                                f"Could not cast 'range_cal' factor {range_cal} to "
                                "float.",
                            )
            for k, loc in enumerate(locs):
                info["chs"][k]["loc"] = loc

        # filters
        filters = self.desc.child("filters")
        if not filters.empty() and self.sfreq != 0:
            highpass = filters.child("highpass").first_child().value()
            lowpass = filters.child("lowpass").first_child().value()
            with info._unlock():
                for name, value in zip(
                    ("highpass", "lowpass"), (highpass, lowpass), strict=False
                ):
                    if len(value) != 0:
                        try:
                            info[name] = float(value)
                        except ValueError:
                            warn(f"Could not cast '{name}' {value} to float.")
        elif not filters.empty() and self.sfreq == 0:
            warn(
                "Node 'filters' found in the description of an irregularly sampled "
                "stream. This is inconsistent and will be skipped."
            )

        projs = self._get_channel_projectors()
        dig = self._get_digitization()
        with info._unlock(update_redundant=True, check_after=True):
            info["projs"] = projs
            if len(dig) != 0:
                info["dig"] = dig
        return info

    def get_channel_names(self) -> list[str] | None:
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

    def get_channel_types(self) -> list[str] | None:
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
        return self._get_channel_info("ch_type")

    def get_channel_units(self) -> list[str] | None:
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

    def _get_channel_info(self, name: str) -> list[str] | None:
        """Get the 'channel/name' element in the XML tree."""
        if self.desc.child("channels").empty():
            return None
        name = _MAPPING_LSL.get(name, name)

        ch_infos = list()
        channels = self.desc.child("channels")
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
            warn(
                f"The stream description contains {len(ch_infos)} elements for "
                f"{self.n_channels} channels."
            )
        return ch_infos

    def _get_channel_projectors(self) -> list[Projection]:
        """Get the SSP vectors in the XML tree."""
        projs = list()
        projectors = self.desc.child("projectors")
        projector = projectors.child("projector")
        while not projector.empty():
            desc = projector.child("desc").first_child().value()
            if len(desc) == 0:
                warn("An SSP projector without description was found. Skipping.")
                projector = projector.next_sibling()
                continue
            kind = projector.child("kind").first_child().value()
            try:
                kind = int(kind)
            except ValueError:
                warn(f"Could not cast the SSP kind {kind} to integer.")
                projector = projector.next_sibling()
                continue

            ch_names = list()
            ch_datas = list()
            data = projector.child("data")
            ch = data.child("channel")
            while not ch.empty():
                ch_name = ch.child("label").first_child().value()
                if len(ch_name) == 0:
                    warn(
                        "SSP projector has an empty-channel label. The channel will "
                        "be skipped."
                    )
                    ch.next_sibling()
                    continue
                ch_data = ch.child("data").first_child().value()
                try:
                    ch_data = float(ch_data)
                except ValueError:
                    warn(
                        f"Could not cast the SSP value {ch_data} for channel {ch_name} "
                        "to float.",
                    )
                    ch.next_sibling()
                    continue
                ch_names.append(ch_name)
                ch_datas.append(ch_data)
                ch = ch.next_sibling()

            assert len(ch_names) == len(ch_datas)  # sanity-check
            proj_data = {
                "nrow": 1,
                "ncol": len(ch_names),
                "row_names": None,
                "col_names": ch_names,
                "data": np.array(ch_datas).reshape(1, -1),
            }
            projs.append(Projection(data=proj_data, desc=desc, kind=kind))
            projector = projector.next_sibling()
        return projs

    def _get_digitization(self) -> list[DigPoint]:
        """Get the digitization in the XML tree."""
        dig = self.desc.child("dig")
        dig_points = list()
        point = dig.child("point")
        while not point.empty():
            kind = point.child("kind").first_child().value()
            kind = _BaseStreamInfo._get_fiff_int_named(
                kind, "dig_kind", _dig_kind_named
            )
            if kind is None:
                point = point.next_sibling()
                continue
            ident = point.child("ident").first_child().value()
            if kind == FIFF.FIFFV_POINT_CARDINAL:
                ident = _BaseStreamInfo._get_fiff_int_named(
                    ident,
                    "dig_ident",
                    _dig_cardinal_named,
                )
            else:
                try:
                    ident = int(ident)
                except ValueError:
                    warn(f"Could not cast 'ident' {ident} to integer.")
                    point = point.next_sibling()
                    continue
            loc = point.child("loc")
            r = [loc.child(pos).first_child().value() for pos in ("X", "Y", "Z")]
            if ident is None or any(len(elt) == 0 for elt in r):
                point = point.next_sibling()
                continue
            try:
                r = np.array([float(elt) for elt in r], dtype=np.float32)
            except ValueError:
                warn(f"Could not cast dig point location {r} to float.")
                point = point.next_sibling()
                continue
            dig_points.append(
                DigPoint(kind=kind, ident=ident, r=r, coord_frame=FIFF.FIFFV_COORD_HEAD)
            )
            point = point.next_sibling()
        return dig_points

    def set_channel_info(self, info: Info) -> None:
        """Set the channel info from a FIFF measurement :class:`~mne.Info`.

        Parameters
        ----------
        info : Info
            :class:`~mne.Info` containing the measurement information.
        """
        check_type(info, (Info,), "info")
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
        assert not self.desc.child("channels").empty()  # sanity-check
        channels = self.desc.child("channels")
        ch = channels.child("channel")
        for ch_info in info["chs"]:
            loc = ch.child("loc")
            loc = ch.append_child("loc") if loc.empty() else loc
            _BaseStreamInfo._set_description_node(
                loc,
                {
                    key: value
                    for key, value in zip(_LOC_NAMES, ch_info["loc"], strict=False)
                },
            )
            ch = ch.next_sibling()
        assert ch.empty()  # sanity-check

        # non-channel variables
        filters = _BaseStreamInfo._add_first_node(self.desc, "filters")
        _BaseStreamInfo._set_description_node(
            filters, {key: info[key] for key in ("highpass", "lowpass")}
        )

        # projectors and digitization
        if len(info["projs"]) != 0:
            self._set_channel_projectors(info["projs"])
        if info["dig"] is not None:
            self._set_digitization(info["dig"])

    def set_channel_names(self, ch_names: list[str] | tuple[str]) -> None:
        """Set the channel names in the description. Existing labels are overwritten.

        Parameters
        ----------
        ch_names : list of str
            List of channel names, matching the number of total channels.
        """
        self._set_channel_info(ch_names, "ch_name")

    def set_channel_types(self, ch_types: str | list[str]) -> None:
        """Set the channel types in the description. Existing types are overwritten.

        The types are given as human readable strings, e.g. ``'eeg'``.

        Parameters
        ----------
        ch_types : list of str | str
            List of channel types, matching the number of total channels.
            If a single :class:`str` is provided, the type is applied to all channels.
        """
        ch_types = (
            [ch_types] * self.n_channels if isinstance(ch_types, str) else ch_types
        )
        self._set_channel_info(ch_types, "ch_type")

    def set_channel_units(
        self, ch_units: str | list[str] | int | list[int] | ScalarIntArray
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
        check_type(ch_units, (list, tuple, np.ndarray, str, "int-like"), "ch_units")
        if isinstance(ch_units, int):
            ch_units = [ensure_int(ch_units, "ch_units")] * self.n_channels
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

    def _set_channel_info(self, ch_infos: list[str], name: str) -> None:
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

        channels = _BaseStreamInfo._add_first_node(self.desc, "channels")
        # fill the 'channel/name' element of the tree and overwrite existing values
        ch = channels.child("channel")
        for ch_info in ch_infos:
            ch = channels.append_child("channel") if ch.empty() else ch
            _BaseStreamInfo._set_description_node(ch, {name: ch_info})
            ch = ch.next_sibling()
        _BaseStreamInfo._prune_description_node(ch, channels)

    def _set_channel_projectors(self, projs: list[Projection]) -> None:
        """Set the SSP projector."""
        check_type(projs, (list,), "projs")
        for elt in projs:
            check_type(elt, (Projection,), "proj")
        projectors = _BaseStreamInfo._add_first_node(self.desc, "projectors")
        # fill the 'channel/name' element of the tree and overwrite existing values
        projector = projectors.child("projector")
        for proj in projs:
            projector = (
                projectors.append_child("projector") if projector.empty() else projector
            )
            _BaseStreamInfo._set_description_node(
                projector, {key: proj[key] for key in ("desc", "kind")}
            )

            data = projector.child("data")
            data = projector.append_child("data") if data.empty() else data
            ch = data.child("channel")
            for ch_name, ch_data in zip(
                proj["data"]["col_names"],
                np.squeeze(proj["data"]["data"]),
                strict=False,
            ):
                ch = data.append_child("channel") if ch.empty() else ch
                _BaseStreamInfo._set_description_node(
                    ch, {"label": ch_name, "data": ch_data}
                )
                ch = ch.next_sibling()
            _BaseStreamInfo._prune_description_node(ch, data)
            projector = projector.next_sibling()
        _BaseStreamInfo._prune_description_node(projector, projectors)

    def _set_digitization(self, dig_points: list[DigPoint]) -> None:
        """Set the digitization points."""
        check_type(dig_points, (list,), "dig_points")
        for elt in dig_points:
            check_type(elt, (DigPoint,), "dig_point")
        dig = _BaseStreamInfo._add_first_node(self.desc, "dig")
        # fill the 'point' element of the tree and overwrite existing integer codes
        point = dig.child("point")
        for dig_point in dig_points:
            point = dig.append_child("point") if point.empty() else point
            _BaseStreamInfo._set_description_node(
                point, {key: dig_point[key] for key in ("kind", "ident")}
            )
            loc = point.child("loc")
            if loc.empty():
                loc = point.append_child("loc")
            _BaseStreamInfo._set_description_node(
                loc,
                {
                    key: value
                    for key, value in zip(("X", "Y", "Z"), dig_point["r"], strict=False)
                },
            )
            point = point.next_sibling()
        _BaseStreamInfo._prune_description_node(point, dig)

    # -- Helper methods to interact with the XMLElement tree ---------------------------
    @staticmethod
    def _add_first_node(desc: XMLElement, name: str) -> XMLElement:
        """Add the first node in the description and return it."""
        if desc.child(name).empty():
            node = desc.append_child(name)
        else:
            node = desc.child(name)
        return node

    @staticmethod
    def _prune_description_node(node: XMLElement, parent: XMLElement) -> None:
        """Prune a node and remove outdated entries."""
        # this is useful in case the sinfo is tempered with and had more entries of type
        # 'node' than it should.
        while not node.empty():
            node_next = node.next_sibling()
            parent.remove_child(node)
            node = node_next

    @staticmethod
    def _set_description_node(node: XMLElement, mapping: dict[str, Any]) -> None:
        """Set the key: value child(s) of a node."""
        for key, value in mapping.items():
            value = str(int(value)) if isinstance(value, int) else str(value)
            if node.child(key).empty():
                node.append_child_value(key, value)
            else:
                node.child(key).first_child().set_value(value)

    # -- Helper methods to retrieve FIFF elements in the XMLElement tree ---------------
    @staticmethod
    def _get_fiff_int_named(
        value: str | None,
        name: str,
        mapping: dict[int, int],
    ) -> int | None:
        """Try to retrieve the FIFF integer code from the str representation."""
        if value is None:
            return None
        try:
            idx = int(value)
            value = mapping[idx]
            return value
        except ValueError:
            warn(f"Could not cast '{name}' {value} to integer.")
        except KeyError:
            warn(
                f"Could not convert '{name}' {int(value)} to a known FIFF "
                "code: {tuple(mapping.keys())}.",
            )
        return None


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
    def _dtype2idxfmt(dtype: str | int | DTypeLike) -> int:
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
