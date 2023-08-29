from __future__ import annotations  # c.f. PEP 563, PEP 649

from contextlib import contextmanager
from math import ceil
from threading import Timer
from typing import TYPE_CHECKING

import numpy as np
from mne import pick_info, pick_types
from mne.channels import rename_channels
from mne.utils import check_version

if check_version("mne", "1.5"):
    from mne.io.constants import FIFF, _ch_unit_mul_named
    from mne.io.meas_info import ContainsMixin, SetChannelsMixin
    from mne.io.pick import _ELECTRODE_CH_TYPES, _picks_to_idx
elif check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF, _ch_unit_mul_named
    from mne._fiff.meas_info import ContainsMixin, SetChannelsMixin
    from mne._fiff.pick import _ELECTRODE_CH_TYPES, _picks_to_idx
else:
    from mne.io.constants import FIFF, _ch_unit_mul_named
    from mne.io.meas_info import ContainsMixin
    from mne.io.pick import _picks_to_idx, _ELECTRODE_CH_TYPES
    from mne.channels.channels import SetChannelsMixin

from .lsl import StreamInlet, resolve_streams
from .lsl.constants import fmt2numpy
from .utils._checks import check_type, check_value
from .utils._docs import copy_doc, fill_doc
from .utils.logs import logger
from .utils.meas_info import _HUMAN_UNITS, _set_channel_units, create_info

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

    from mne import Info
    from mne.channels import DigMontage
    from numpy.typing import NDArray, DTypeLike

    from bsl.lsl.stream_info import _BaseStreamInfo


class Stream(ContainsMixin, SetChannelsMixin):
    """Stream object representing a single LSL stream.

    Parameters
    ----------
    bufsize : float | int
        Size of the buffer keeping track of the data received from the stream. If
        the stream sampling rate ``sfreq`` is regular, ``bufsize`` is expressed in
        seconds. The buffer will hold the last ``bufsize * sfreq`` samples (ceiled).
        If the strean sampling sampling rate ``sfreq`` is irregular, ``bufsize`` is
        expressed in samples. The buffer will hold the last ``bufsize`` samples.
    name : str
        Name of the LSL stream.
    stype : str
        Type of the LSL stream.
    source_id : str
        ID of the source of the LSL stream.

    Notes
    -----
    The 3 arguments ``name``, ``stype``, and ``source_id`` must uniquely identify an
    LSL stream. If this is not possible, please resolve the available LSL streams
    with `~bsl.lsl.resolve_streams` and create an inlet with `~bsl.lsl.StreamInlet`.
    """

    def __init__(
        self,
        bufsize: float,
        name: Optional[str] = None,
        stype: Optional[str] = None,
        source_id: Optional[str] = None,
    ):
        check_type(bufsize, ("numeric",), "bufsize")
        if bufsize <= 0:
            raise ValueError(
                "The buffer size 'bufsize' must be a strictly positive number. "
                f"{bufsize} is invalid."
            )
        check_type(name, (str, None), "name")
        check_type(stype, (str, None), "stype")
        check_type(source_id, (str, None), "source_id")
        self._name = name
        self._stype = stype
        self._source_id = source_id
        self._bufsize = bufsize

        # -- variables defined after connection ----------------------------------------
        self._sinfo = None
        self._inlet = None
        self._info = None
        # The buffer shape is similar to a pull_sample/pull_chunk from an inlet:
        # (n_samples, n_channels). New samples are added to the right of the buffer
        # while old samples are removed from the left of the buffer.
        self._acquisition_delay = None
        self._acquisition_thread = None
        self._buffer = None
        # picks_inlet represent the selection of channels from the inlet.
        self._picks_inlet = None
        self._timestamps = None

        # -- variables defined for processing ------------------------------------------
        self._ref_channels = []

    @copy_doc(ContainsMixin.__contains__)
    def __contains__(self, ch_type) -> bool:
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required by "
                "the 'in' operator. Please connect to the stream to create the Info."
            )
        return super().__contains__(ch_type)

    def __del__(self):
        """Try to disconnect the stream when deleting the object."""
        try:
            self.disconnect()
        except Exception:
            pass

    @fill_doc
    def add_reference_channels(
        self,
        ref_channels: Union[str, List[str], Tuple[str]],
        ref_units: Optional[
            Union[str, int, List[Union[str, int]], Tuple[Union[str, int]]]
        ] = None,
    ) -> None:
        """Add EEG reference channels to data that consists of all zeros.

        Adds EEG reference channels that are not part of the streamed data. This is
        useful when you need to re-reference your data to different channels. These
        added channels will consist of all zeros.

        Parameters
        ----------
        %(ref_channels)s
        ref_units : str | int | list of str | list of int | None
            The unit or unit multiplication factor of the reference channels. The unit
            can be given as a human-readable string or as a unit multiplication factor,
            e.g. ``-6`` for microvolts corresponding to ``1e-6``.
            If not provided, the added EEG reference channel has a unit multiplication
            factor set to ``0`` which corresponds to Volts. Use
            `~Stream.set_channel_units` to change the unit multiplication factor.
        """
        self._check_connected(name="Stream.add_reference_channels()")
        # error checking and conversion of the arguments to valid values
        if isinstance(ref_channels, str):
            ref_channels = [ref_channels]
        if isinstance(ref_units, (str, int)):
            ref_units = [ref_units]
        elif ref_units is None:
            ref_units = [0] * len(ref_channels)
        check_type(ref_channels, (list, tuple), "ref_channels")
        check_type(ref_units, (list, tuple), "ref_units")
        if len(ref_channels) != len(ref_units):
            raise ValueError(
                "The number of reference channels and of reference units provided must "
                f"match. {len(ref_channels)} channels and {len(ref_units)} units were "
                "provided."
            )
        for ch in ref_channels:
            check_type(ch, (str,), "ref_channel")
            if ch in self.ch_names:
                raise ValueError(f"The channel {ch} is already part of the stream.")
        for k, unit in enumerate(ref_units):
            check_type(unit, (str, "int-like"), unit)
            if isinstance(unit, str):
                if unit not in _HUMAN_UNITS[FIFF.FIFF_UNIT_V]:
                    raise ValueError(
                        f"The human-readable unit {unit} for the channel "
                        f"{ref_channels[k]} is unknown to BSL. Please contact the "
                        "developers on GitHub if you want to add support for that unit."
                    )
                ref_units[k] = _HUMAN_UNITS[FIFF.FIFF_UNIT_V][unit]
            elif isinstance(unit, int):
                check_value(unit, _ch_unit_mul_named, "unit")
                ref_units[k] = _ch_unit_mul_named[unit]

        # try to figure out the reference channels location
        if self.get_montage() is None:
            ref_dig_array = np.full(12, np.nan)
            logger.info(
                "Location for this channel is unknown, consider calling set_montage() "
                "again if needed."
            )
        else:
            ref_dig_loc = [
                dl
                for dl in self._info["dig"]
                if (dl["kind"] == FIFF.FIFFV_POINT_EEG and dl["ident"] == 0)
            ]
            if len(ref_channels) > 1 or len(ref_dig_loc) != len(ref_channels):
                ref_dig_array = np.full(12, np.nan)
                logger.warning(
                    "The locations of multiple reference channels are ignored."
                )
            else:  # n_ref_channels == 1 and a single ref digitization exists
                ref_dig_array = np.concatenate(
                    (ref_dig_loc[0]["r"], ref_dig_loc[0]["r"], np.zeros(6))
                )
                # replace the (possibly new) ref location for each channel
                with self._info._unlock():
                    for idx in pick_types(self._info, meg=False, eeg=True, exclude=[]):
                        self._info["chs"][idx]["loc"][3:6] = ref_dig_loc[0]["r"]

        # add the reference channels to the info
        nchan = len(self.ch_names)
        with self._info._unlock(update_redundant=True):
            for ch in ref_channels:
                chan_info = {
                    "ch_name": ch,
                    "coil_type": FIFF.FIFFV_COIL_EEG,
                    "kind": FIFF.FIFFV_EEG_CH,
                    "logno": nchan + 1,
                    "scanno": nchan + 1,
                    "cal": 1,
                    "range": 1.0,
                    "unit_mul": FIFF.FIFF_UNITM_NONE,
                    "unit": FIFF.FIFF_UNIT_V,
                    "coord_frame": FIFF.FIFFV_COORD_HEAD,
                    "loc": ref_dig_array,
                }
                self._info["chs"].append(chan_info)

        # save reference channels
        self._ref_channels.extend(ref_channels)

        # create the associated numpy array and edit buffer
        refs = np.zeros((self._timestamps.size, len(ref_channels)), dtype=self.dtype)
        with self._interrupt_acquisition():
            self._buffer = np.hstack((self._buffer, refs), dtype=self.dtype)

    @fill_doc
    def anonymize(self, daysback=None, keep_his=False, *, verbose=None):
        """Anonymize the measurement information in-place.

        Parameters
        ----------
        %(daysback_anonymize_info)s
        %(keep_his_anonymize_info)s
        %(verbose)s

        Notes
        -----
        %(anonymize_info_notes)s
        """
        self._check_connected(name="Stream.anonymize()")
        super().anonymize(daysback=daysback, keep_his=keep_his, verbose=verbose)

    def connect(
        self,
        processing_flags: Optional[Union[str, Sequence[str]]] = None,
        timeout: Optional[float] = 2,
        acquisition_delay: float = 0.2,
    ) -> None:
        """Connect to the LSL stream and initiate data collection in the buffer.

        If the streams were not resolve with the method `~bsl.Stream.resolve`,

        Parameters
        ----------
        processing_flags : list of str | ``'all'`` | None
            Set the post-processing options. By default, post-processing is disabled.
            Any combination of the processing flags is valid. The available flags are:

            * ``'clocksync'``: Automatic clock synchronization, equivalent to
              manually adding the estimated `~bsl.lsl.StreamInlet.time_correction`.
            * ``'dejitter'``: Remove jitter on the received timestamps with a
              smoothing algorithm.
            * ``'monotize'``: Force the timestamps to be monotically ascending.
              This option should not be enable if ``'dejitter'`` is not enabled.
        timeout : float | None
            Optional timeout (in seconds) of the operation. ``None`` disables the
            timeout. The timeout value is applied once to every operation supporting it.
        acquisition_delay : float
            Delay in seconds between 2 acquisition during which chunks of data are
            pulled from the `~bsl.lsl.StreamInlet`.

        Notes
        -----
        If all 3 stream identifiers ``name``, ``stype`` and ``source_id`` are left to
        ``None``, resolution of the available streams will require a full ``timeout``,
        blocking the execution until this function returns. If at least one of the 3
        stream identifiers is specified, resolution will stop as soon as one stream
        matching the identifier is found.
        """
        if self.connected:
            logger.warning("The stream is already connected. Skipping.")
            return
        # The threadsafe processing flag should not be needed for this class. If it is
        # provided, then it means the user is retrieving and doing something with the
        # inlet in a different thread. This use-case is not supported, and users which
        # needs this level of control should create the inlet themselves.
        if processing_flags is not None and (
            processing_flags == "threadsafe" or "threadsafe" in processing_flags
        ):
            raise ValueError(
                "The 'threadsafe' processing flag should not be provided for a BSL "
                "Stream. If you require access to the underlying StreamInlet in a "
                "separate thread, please instantiate the StreamInlet directly from "
                "bsl.lsl.StreamInlet."
            )
        if processing_flags == "all":
            processing_flags = ("clocksync", "dejitter", "monotize")
        check_type(acquisition_delay, ("numeric",), "acquisition_delay")
        if acquisition_delay <= 0:
            raise ValueError(
                "The acquisition delay must be a strictly positive number "
                "defining the delay at which new samples are acquired in seconds. For "
                "instance, 0.2 corresponds to a pull every 200 ms. The provided "
                f"{acquisition_delay} is invalid."
            )

        # resolve and connect to available streams
        sinfos = resolve_streams(timeout, self._name, self._stype, self._source_id)
        if len(sinfos) != 1:
            raise RuntimeError(
                "The provided arguments 'name', 'stype', and 'source_id' do not "
                f"uniquely identify an LSL stream. {len(sinfos)} were found: "
                f"{[(sinfo.name, sinfo.stype, sinfo.source_id) for sinfo in sinfos]}."
            )
        if sinfos[0].dtype == "string":
            raise RuntimeError(
                "The Stream class is designed for numerical types. It does not support "
                "string LSL streams. Please use a bsl.lsl.StreamInlet directly to "
                "interact with this stream."
            )
        # create inlet and retrieve stream info
        self._inlet = StreamInlet(
            sinfos[0], max_buffered=self._bufsize, processing_flags=processing_flags
        )
        self._inlet.open_stream(timeout=timeout)
        self._sinfo = self._inlet.get_sinfo()
        self._name = self._sinfo.name
        self._stype = self._sinfo.stype
        self._source_id = self._sinfo.source_id
        # create MNE info from the LSL stream info returned by an open stream inlet
        self._info = create_info(
            self._sinfo.n_channels,
            self._sinfo.sfreq,
            self._sinfo.stype,
            self._sinfo,
        )
        # initiate time-correction
        tc = self._inlet.time_correction(timeout=timeout)
        logger.info("The estimated timestamp offset is %.2f seconds.", tc)

        # create buffer of shape (n_samples, n_channels) and (n_samples,)
        if self._inlet.sfreq == 0:
            self._buffer = np.zeros(
                (self._bufsize, self._inlet.n_channels),
                dtype=fmt2numpy[self._inlet._dtype],
            )
            self._timestamps = np.zeros(self._bufsize, dtype=np.float64)
        else:
            self._buffer = np.zeros(
                (ceil(self._bufsize * self._inlet.sfreq), self._inlet.n_channels),
                dtype=fmt2numpy[self._inlet._dtype],
            )
            self._timestamps = np.zeros(
                ceil(self._bufsize * self._inlet.sfreq), dtype=np.float64
            )
        self._picks_inlet = np.arange(0, self._inlet.n_channels)

        # define the acquisition thread
        self._acquisition_delay = acquisition_delay
        self._acquisition_thread = Timer(0, self._acquire)
        self._acquisition_thread.daemon = True
        self._acquisition_thread.start()

    def disconnect(self) -> None:
        """Disconnect from the LSL stream and interrupt data collection."""
        self._check_connected(name="Stream.disconnect()")
        while self._acquisition_thread.is_alive():
            self._acquisition_thread.cancel()
        self._inlet.close_stream()
        del self._inlet
        self._reset_variables()

    def drop_channels(self, ch_names: Union[str, List[str], Tuple[str]]) -> None:
        """Drop channel(s).

        Parameters
        ----------
        ch_names : str | list of str
            Name or list of names of channels to remove.

        See Also
        --------
        pick
        """
        self._check_connected(name="Stream.drop_channels()")
        if isinstance(ch_names, str):
            ch_names = [ch_names]
        check_type(ch_names, (list, tuple), "ch_names")
        try:
            idx = np.array([self._info.ch_names.index(ch_name) for ch_name in ch_names])
        except ValueError:
            raise ValueError(
                "The argument 'ch_names' must contain existing channel names."
            )

        picks = np.setdiff1d(np.arange(len(self._info.ch_names)), idx)
        self._pick(picks)

    def filter(self) -> None:
        self._check_connected(name="Stream.filter()")
        self._check_regular_sampling(name="Stream.filter()")
        raise NotImplementedError

    @copy_doc(ContainsMixin.get_channel_types)
    def get_channel_types(
        self, picks=None, unique=False, only_data_chs=False
    ) -> List[str]:
        self._check_connected(name="Stream.get_channel_types()")
        return super().get_channel_types(
            picks=picks, unique=unique, only_data_chs=only_data_chs
        )

    @fill_doc
    def get_channel_units(
        self, picks=None, only_data_chs: bool = False
    ) -> List[Tuple[int, int]]:
        """Get a list of channel type for each channel.

        Parameters
        ----------
        %(picks_all)s
        only_data_chs : bool
            Whether to ignore non-data channels. Default is ``False``.

        Returns
        -------
        channel_units : list of tuple of shape (2,)
            A list of 2-element tuples. The first element contains the unit FIFF code
            and its associated name, e.g. ``107 (FIFF_UNIT_V)`` for Volts. The second
            element contains the unit multiplication factor, e.g. ``-6 (FIFF_UNITM_MU)``
            for micro (corresponds to ``1e-6``).
        """
        self._check_connected(name="Stream.get_channel_units()")
        check_type(only_data_chs, (bool,), "only_data_chs")
        none = "data" if only_data_chs else "all"
        picks = _picks_to_idx(self._info, picks, none, (), allow_empty=False)
        channel_units = list()
        for idx in picks:
            channel_units.append(
                (self._info["chs"][idx]["unit"], self._info["chs"][idx]["unit_mul"])
            )
        return channel_units

    def get_data(
        self,
        winsize: Optional[float],
    ) -> Tuple[NDArray[float], NDArray[float]]:
        """Retrieve the latest data from the buffer.

        Parameters
        ----------
        winsize : float | int | None
            Size of the window of data to view. If the stream sampling rate ``sfreq`` is
            regular, ``winsize`` is expressed in seconds. The window will view the last
            ``winsize * sfreq`` samples (ceiled) from the buffer. If the stream sampling
            sampling rate ``sfreq`` is irregular, ``winsize`` is expressed in samples.
            The window will view the last ``winsize`` samples. If ``None``, the entire
            buffer is returned.

        Returns
        -------
        data : array of shape (n_samples, n_channels)
            Data in the given window.
        timestamps : array of shape (n_samples,)
            Timestamps in the given window.
        """
        try:
            if winsize is None:
                n_samples = self._buffer.shape[0]
            else:
                assert (
                    0 <= winsize
                ), "The window size must be a strictly positive number."
                n_samples = (
                    winsize
                    if self._inlet.sfreq == 0
                    else ceil(winsize * self._inlet.sfreq)
                )
            return self._buffer[-n_samples:, :].T, self._timestamps[-n_samples:]
        except Exception:
            if not self.connected:
                raise RuntimeError(
                    "The Stream is not connected. Please connect to the stream before "
                    "retrieving data from the buffer."
                )
            else:
                logger.error(
                    "Something went wrong while retrieving data from a connected "
                    "stream. Please open an issue on GitHub and provide the error "
                    "traceback to the developers."
                )
            raise

    @copy_doc(SetChannelsMixin.get_montage)
    def get_montage(self) -> Optional[DigMontage]:
        self._check_connected(name="Stream.get_montage()")
        return super().get_montage()

    def load_stream_config(self) -> None:
        raise NotImplementedError

    def plot(self):
        self._check_connected(name="Stream.plot()")
        raise NotImplementedError

    @fill_doc
    def pick(self, picks, exclude=()) -> None:
        """Pick a subset of channels.

        Parameters
        ----------
        %(picks_all)s
        exclude : str | list of str
            Set of channels to exclude, only used when picking is based on types, e.g.
            ``exclude='bads'`` when ``picks="meg"``.

        See Also
        --------
        drop_channels
        """
        self._check_connected(name="Stream.pick()")
        picks = _picks_to_idx(self._info, picks, "all", exclude, allow_empty=False)
        self._pick(picks)

    def record(self):
        self._check_connected(name="Stream.record()")
        raise NotImplementedError

    @fill_doc
    def rename_channels(
        self,
        mapping: Union[Dict[str, str], Callable],
        allow_duplicates: bool = False,
        *,
        verbose=None,
    ) -> None:
        """Rename channels.

        Parameters
        ----------
        mapping : dict | callable
            A dictionary mapping the old channel to a new channel name e.g.
            ``{'EEG061' : 'EEG161'}``. Can also be a callable function that takes and
            returns a string.
        allow_duplicates : bool
            If True (default False), allow duplicates, which will automatically be
            renamed with ``-N`` at the end.

            .. versionadded:: MNE 0.22.0
        %(verbose)s
        """
        self._check_connected(name="Stream.rename_channels()")
        rename_channels(
            self._info,
            mapping=mapping,
            allow_duplicates=allow_duplicates,
            verbose=verbose,
        )

    def save_stream_config(self) -> None:
        raise NotImplementedError

    def set_bipolar_reference(self):
        self._check_connected(name="Stream.set_bipolar_reference()")
        self._check_regular_sampling(name="Stream.set_bipolar_reference()")
        raise NotImplementedError

    @fill_doc
    def set_channel_types(
        self, mapping: Dict[str, str], *, on_unit_change: str = "warn", verbose=None
    ) -> None:
        """Define the sensor type of channels.

        If the new channel type changes the unit type, e.g. from ``T/m`` to ``V``, the
        unit multiplication factor is reset to ``0``. Use `~Stream.set_channel_units` to
        change the multiplication factor, e.g. from ``0`` to ``-6`` to change from Volts
        to microvolts.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping a channel to a sensor type (str), e.g.,
            ``{'EEG061': 'eog'}`` or ``{'EEG061': 'eog', 'TRIGGER': 'stim'}``.
        on_unit_change : ``'raise'`` | ``'warn'`` | ``'ignore'``
            What to do if the measurement unit of a channel is changed automatically to
            match the new sensor type.

            .. versionadded:: MNE 1.4
        %(verbose)s
        """
        self._check_connected(name="Stream.set_channel_types()")
        super().set_channel_types(
            mapping=mapping, on_unit_change=on_unit_change, verbose=verbose
        )

    def set_channel_units(self, mapping: Dict[str, Union[str, int]]) -> None:
        """Define the channel unit multiplication factor.

        The unit itself is defined by the sensor type. Use `~Stream.set_channel_types`
        to change the channel type, e.g. from planar gradiometers in ``T/m`` to EEG in
        ``V``.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping a channel to a unit, e.g. ``{'EEG061': 'microvolts'}``.
            The unit can be given as a human-readable string or as a unit multiplication
            factor, e.g. ``-6`` for microvolts corresponding to ``1e-6``.

        Notes
        -----
        If the human-readable unit of your channel is not yet supported by BSL, please
        contact the developers on GitHub to add your units to the known set.
        """
        self._check_connected(name="Stream.set_channel_units()")
        _set_channel_units(self._info, mapping)

    def set_eeg_reference(
        self,
        ref_channels: Union[str, List[str], Tuple[str]],
        ch_type: Union[str, List[str], Tuple[str]] = "eeg",
    ) -> None:
        """Specify which reference to use for EEG data.

        Use this function to explicitly specify the desired reference for EEG. This can
        be either an existing electrode or a new virtual channel. This function will
        re-reference the data according to the desired reference.

        Parameters
        ----------
        ref_channels : str | list of str
            Name(s) of the channel(s) used to construct the reference. Can also be set
            to ``'average'`` to apply a common average reference.
        ch_type : str | list of str
            The name of the channel type to apply the reference to. Valid channel types
            are ``'eeg'``, ``'ecog'``, ``'seeg'``, ``'dbs'``.
        """
        self._check_connected(name="Stream.set_eeg_reference()")
        self._check_regular_sampling(name="Stream.set_eeg_reference()")

        if isinstance(ch_type, str):
            ch_type = [ch_type]
        check_type(ch_type, (tuple, list), "ch_type")
        for type_ in ch_type:
            check_value(type_, _ELECTRODE_CH_TYPES, "ch_type")
            if type_ not in self:
                raise ValueError(
                    f"There are no channels of type {type_} in this stream."
                )

        picks = _picks_to_idx(self._info, ch_type, "all", (), allow_empty=False)
        raise NotImplementedError

    def set_meas_date(
        self, meas_date: Optional[Union[datetime, float, Tuple[float]]]
    ) -> None:
        """Set the measurement start date.

        Parameters
        ----------
        meas_date : datetime | float | tuple | None
            The new measurement date.
            If datetime object, it must be timezone-aware and in UTC.
            A tuple of (seconds, microseconds) or float (alias for
            ``(meas_date, 0)``) can also be passed and a datetime
            object will be automatically created. If None, will remove
            the time reference.

        See Also
        --------
        anonymize
        """
        self._check_connected(name="Stream.set_meas_date()")
        super().set_meas_date(meas_date)

    @fill_doc
    def set_montage(
        self,
        montage,
        match_case=True,
        match_alias=False,
        on_missing="raise",
        *,
        verbose=None,
    ) -> None:
        """Set %(montage_types)s channel positions and digitization points.

        Parameters
        ----------
        %(montage)s
        %(match_case)s
        %(match_alias)s
        %(on_missing_montage)s
        %(verbose)s

        See Also
        --------
        mne.channels.make_standard_montage
        mne.channels.make_dig_montage
        mne.channels.read_custom_montage

        Notes
        -----
        .. warning::

            Only %(montage_types)s channels can have their positions set using a
            montage. Other channel types (e.g., MEG channels) should have their
            positions defined properly using their data reading functions.
        """
        self._check_connected(name="Stream.set_montage()")
        super().set_montage(
            montage=montage,
            match_case=match_case,
            match_alias=match_alias,
            on_missing=on_missing,
            verbose=verbose,
        )

    def _acquire(self) -> None:
        """Update function pulling new samples in the buffer at a regular interval."""
        try:
            # pull data
            data, timestamps = self._inlet.pull_chunk(timeout=0.0)
            if timestamps.size == 0:
                return None  # interrupt early

            # process acquisition window
            data = data[:, self._picks_inlet]
            if len(self._ref_channels) != 0:
                refs = np.zeros(
                    (timestamps.size, len(self._ref_channels)), dtype=self.dtype
                )
                data = np.hstack((data, refs), dtype=self.dtype)

            # roll and update buffers
            self._buffer = np.roll(self._buffer, -data.shape[0], axis=0)
            self._timestamps = np.roll(self._timestamps, -timestamps.size, axis=0)
            self._buffer[-timestamps.size :, :] = data  # noqa: E203
            self._timestamps[-timestamps.size :] = timestamps  # noqa: E203
        except Exception:
            self._reset_variables()
            return None  # equivalent to an interrupt
        else:
            # recreate the timer thread as it is one-call only
            self._acquisition_thread = Timer(self._acquisition_delay, self._acquire)
            self._acquisition_thread.daemon = True
            self._acquisition_thread.start()

    def _check_connected(self, name: str):
        """Check that the stream is connected before calling the function 'name'."""
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                f"use {name}. Please connect to the stream to create the Info."
            )

    def _check_regular_sampling(self, name: str):
        """Check that the stream has a regular sampling rate."""
        if self.info["sfreq"] == 0:
            raise RuntimeError(
                f"The method {name} can not be used on a stream with an irregular "
                "sampling rate."
            )

    @contextmanager
    def _interrupt_acquisition(self):
        """Context manager interrupting the acquisition thread."""
        if not self.connected:
            raise RuntimeError(
                "Interruption of the acquisition thread was requested but the stream "
                "is not connected. Please open an issue on GitHub and provide the "
                "error traceback to the developers."
            )

        while self._acquisition_thread.is_alive():
            self._acquisition_thread.cancel()
        yield
        self._acquisition_thread = Timer(self._acquisition_delay, self._acquire)
        self._acquisition_thread.start()

    def _pick(self, picks: NDArray[int]) -> None:
        """Interrupt acquisition and apply the channel selection."""
        picks_inlet = picks[np.where(picks < self._picks_inlet.size)[0]]
        if picks_inlet.size == 0:
            raise RuntimeError(
                "The requested channel selection would not leave any channel from the "
                "LSL Stream."
            )

        with self._interrupt_acquisition():
            self._info = pick_info(self._info, picks)
            self._picks_inlet = self._picks_inlet[picks_inlet]
            self._buffer = self._buffer[:, picks]

            # prune added channels which are not part of the inlet
            for ch in self._ref_channels[::-1]:
                if ch not in self.ch_names:
                    self._ref_channels.remove(ch)

    def _reset_variables(self) -> None:
        """Reset variables define after connection."""
        self._sinfo = None
        self._inlet = None
        self._info = None
        self._acquisition_delay = None
        self._acquisition_thread = None
        self._buffer = None
        self._picks_inlet = None
        self._ref_channels = []
        self._timestamps = None

    # ----------------------------------------------------------------------------------
    @property
    def compensation_grade(self) -> Optional[int]:
        """The current gradient compensation grade.

        :type: `int` | None
        """
        self._check_connected(name="Stream.compensation_grade")
        return super().compensation_grade

    # ----------------------------------------------------------------------------------
    @property
    def ch_names(self) -> List[str]:
        """Name of the channels.

        :type: `list` of `str`
        """
        self._check_connected(name="Stream.ch_names")
        return self._info.ch_names

    @property
    def connected(self) -> bool:
        """Connection status of the stream.

        :type: `bool`
        """
        attributes = (
            "_sinfo",
            "_inlet",
            "_info",
            "_acquisition_delay",
            "_acquisition_thread",
            "_buffer",
            "_picks_inlet",
            "_timestamps",
        )
        if all(getattr(self, attr) is None for attr in attributes):
            return False
        else:
            # sanity-check
            assert not any(getattr(self, attr) is None for attr in attributes)
            return True

    @property
    def dtype(self) -> Optional[DTypeLike]:
        """Channel format of the stream."""
        return getattr(self._buffer, "dtype", None)

    @property
    def info(self) -> Optional[Info]:
        """Info of the LSL stream.

        :type: `~mne.Info` | None
        """
        return self._info

    @property
    def name(self) -> Optional[str]:
        """Name of the LSL stream.

        :type: `str` | None
        """
        return self._name

    @property
    def sinfo(self) -> Optional[_BaseStreamInfo]:
        """StreamInfo of the connected stream.

        :type: `~bsl.lsl.StreamInfo` | None
        """
        return self._sinfo

    @property
    def stype(self) -> Optional[str]:
        """Type of the LSL stream.

        :type: `str` | None
        """
        return self._stype

    @property
    def source_id(self) -> Optional[str]:
        """ID of the source of the LSL stream.

        :type: `str` | None
        """
        return self._source_id
