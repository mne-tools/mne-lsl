from __future__ import annotations  # c.f. PEP 563, PEP 649

from abc import ABC, abstractmethod
from contextlib import contextmanager
from math import ceil
from threading import Timer
from typing import TYPE_CHECKING

import numpy as np
from mne import pick_info, pick_types
from mne.channels import rename_channels
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.constants import FIFF, _ch_unit_mul_named
    from mne._fiff.meas_info import ContainsMixin, SetChannelsMixin
    from mne._fiff.pick import _picks_to_idx
elif check_version("mne", "1.5"):
    from mne.io.constants import FIFF, _ch_unit_mul_named
    from mne.io.meas_info import ContainsMixin, SetChannelsMixin
    from mne.io.pick import _picks_to_idx
else:
    from mne.io.constants import FIFF, _ch_unit_mul_named
    from mne.io.meas_info import ContainsMixin
    from mne.io.pick import _picks_to_idx
    from mne.channels.channels import SetChannelsMixin

from ..utils._checks import check_type, check_value
from ..utils._docs import copy_doc, fill_doc
from ..utils.logs import logger
from ..utils.meas_info import _HUMAN_UNITS, _set_channel_units

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Callable, Optional, Union

    from mne import Info
    from mne.channels import DigMontage
    from numpy.typing import DTypeLike, NDArray

    from .._typing import ScalarIntType, ScalarType


@fill_doc
class BaseStream(ABC, ContainsMixin, SetChannelsMixin):
    """Stream object representing a single real-time stream.

    Parameters
    ----------
    %(stream_bufsize)s
    """

    @abstractmethod
    def __init__(
        self,
        bufsize: float,
    ) -> None:
        check_type(bufsize, ("numeric",), "bufsize")
        if bufsize <= 0:
            raise ValueError(
                "The buffer size 'bufsize' must be a strictly positive number. "
                f"{bufsize} is invalid."
            )
        self._bufsize = bufsize

    @copy_doc(ContainsMixin.__contains__)
    def __contains__(self, ch_type: str) -> bool:
        self._check_connected("the 'in' operator")
        return super().__contains__(ch_type)

    def __del__(self):
        """Try to disconnect the stream when deleting the object."""
        logger.debug("Deleting %s", self)
        try:
            self.disconnect()
        except Exception:
            pass

    @abstractmethod
    def __repr__(self) -> str:  # pragma: no cover
        """Representation of the instance."""
        # This method needs to define the str representation of the class based on the
        # attributes of the Stream. For instance, an LSL stream is defined by 3
        # attributes: name, stype, source_id. Thus a possible representation is:
        # <Stream: ON | {name} - {stype} (source: {source_id})>
        pass

    @fill_doc
    def add_reference_channels(
        self,
        ref_channels: Union[str, list[str], tuple[str]],
        ref_units: Optional[
            Union[str, int, list[Union[str, int]], tuple[Union[str, int]]]
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
            ``Stream.set_channel_units`` to change the unit multiplication factor.
        """
        self._check_connected_and_regular_sampling("add_reference_channels()")

        # don't allow to add reference channels after a custom reference has been set
        # with Stream.set_eeg_reference, for simplicity.
        if self._info["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_ON:
            raise RuntimeError(
                "The method Stream.add_reference_channels() can only be called before "
                "Stream.set_eeg_reference is called and the reference is changed. "
                "If you want to add other reference to this Stream, please disconnect "
                "and reconnect to reset the Stream."
            )

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
                        f"{ref_channels[k]} is unknown to MNE-LSL. Please contact the "
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

        # create the associated numpy array and edit buffer
        refs = np.zeros((self._timestamps.size, len(ref_channels)), dtype=self.dtype)
        with self._interrupt_acquisition():
            self._added_channels.extend(ref_channels)  # save reference channels
            self._buffer = np.hstack((self._buffer, refs), dtype=self.dtype)

    @fill_doc
    def anonymize(
        self,
        daysback: Optional[int] = None,
        keep_his: bool = False,
        *,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> None:
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
        self._check_connected(name="anonymize()")
        super().anonymize(daysback=daysback, keep_his=keep_his, verbose=verbose)

    @abstractmethod
    def connect(
        self,
        acquisition_delay: float,
    ) -> None:
        """Connect to the stream and initiate data collection in the buffer.

        Parameters
        ----------
        acquisition_delay : float
            Delay in seconds between 2 acquisition during which chunks of data are
            pulled from the connected device.
        """
        if self.connected:
            logger.warning("The stream is already connected. Skipping.")
            return None
        check_type(acquisition_delay, ("numeric",), "acquisition_delay")
        if acquisition_delay < 0:
            raise ValueError(
                "The acquisition delay must be a positive number "
                "defining the delay at which new samples are acquired in seconds. For "
                "instance, 0.2 corresponds to a pull every 200 ms. The provided "
                f"{acquisition_delay} is invalid."
            )
        self._acquisition_delay = acquisition_delay
        self._n_new_samples = 0
        # This method needs to connect to a stream, retrieve the stream information and
        # create the ringbuffer. By the end of this method, the following variables
        # must exist:
        # - self._info: mne.Info
        # - self._buffer: array of shape (n_samples, n_channels)
        # - self._timestamps: array of shape (n_samples,) with n_samples which differs
        #   between regularly and irregularly sampled streams.
        # - self._picks_inlet: array of shape (n_channels,)
        # plus any additional variables needed by the source and the stream-specific
        # methods.

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the LSL stream and interrupt data collection."""
        self._check_connected(name="disconnect()")
        self._interrupt = True
        while self._acquisition_thread.is_alive():
            self._acquisition_thread.cancel()
        # This method needs to close any inlet/network object and need to end with
        # self._reset_variables().

    def drop_channels(self, ch_names: Union[str, list[str], tuple[str]]) -> None:
        """Drop channel(s).

        Parameters
        ----------
        ch_names : str | list of str
            Name or list of names of channels to remove.

        See Also
        --------
        pick
        """
        self._check_connected(name="drop_channels()")
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
        """Filter the stream. Not implemented."""
        self._check_connected_and_regular_sampling("filter()")
        raise NotImplementedError

    @copy_doc(ContainsMixin.get_channel_types)
    def get_channel_types(
        self, picks=None, unique=False, only_data_chs=False
    ) -> list[str]:
        self._check_connected(name="get_channel_types()")
        return super().get_channel_types(
            picks=picks, unique=unique, only_data_chs=only_data_chs
        )

    @fill_doc
    def get_channel_units(
        self, picks=None, only_data_chs: bool = False
    ) -> list[tuple[int, int]]:
        """Get a list of channel unit for each channel.

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
        self._check_connected(name="get_channel_units()")
        check_type(only_data_chs, (bool,), "only_data_chs")
        none = "data" if only_data_chs else "all"
        picks = _picks_to_idx(self._info, picks, none, (), allow_empty=False)
        channel_units = list()
        for idx in picks:
            channel_units.append(
                (self._info["chs"][idx]["unit"], self._info["chs"][idx]["unit_mul"])
            )
        return channel_units

    @fill_doc
    def get_data(
        self,
        winsize: Optional[float] = None,
        picks: Optional[
            Union[str, list[str], list[int], NDArray[+ScalarIntType]]
        ] = None,
    ) -> tuple[NDArray[+ScalarType], NDArray[np.float64]]:
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
        %(picks_all)s

        Returns
        -------
        data : array of shape (n_channels, n_samples)
            Data in the given window.
        timestamps : array of shape (n_samples,)
            Timestamps in the given window.

        Notes
        -----
        The number of newly available samples stored in the property ``n_new_samples``
        is reset at every function call, even if all channels were not selected with
        the argument ``picks``.
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
                    if self._info["sfreq"] == 0
                    else ceil(winsize * self._info["sfreq"])
                )
            # Support channel selection since the performance impact is small.
            # >>> %timeit _picks_to_idx(raw.info, "eeg")
            # 256 µs ± 5.03 µs per loop
            # >>> %timeit _picks_to_idx(raw.info, ["F7", "vEOG"])
            # 8.68 µs ± 113 ns per loop
            # >>> %timeit _picks_to_idx(raw.info, None)
            # 253 µs ± 1.22 µs per loop
            picks = _picks_to_idx(self._info, picks, none="all")
            self._n_new_samples = 0  # reset the number of new samples
            return self._buffer[-n_samples:, picks].T, self._timestamps[-n_samples:]
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
        self._check_connected(name="get_montage()")
        return super().get_montage()

    def plot(self):
        """Open a real-time stream viewer. Not implemented."""
        self._check_connected(name="plot()")
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

        Notes
        -----
        Contrary to MNE-Python, re-ordering channels is not supported in ``MNE-LSL``.
        Thus, if explicit channel names are provided in ``picks``, they are sorted to
        match the order of existing channel names.
        """
        self._check_connected(name="pick()")
        picks = _picks_to_idx(self._info, picks, "all", exclude, allow_empty=False)
        picks = np.sort(picks)
        self._pick(picks)

    def record(self):
        """Record the stream data to disk. Not implemented."""
        self._check_connected(name="record()")
        raise NotImplementedError

    @fill_doc
    def rename_channels(
        self,
        mapping: Union[dict[str, str], Callable],
        allow_duplicates: bool = False,
        *,
        verbose: Optional[Union[bool, str, int]] = None,
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
        %(verbose)s
        """
        self._check_connected(name="rename_channels()")
        rename_channels(
            self._info,
            mapping=mapping,
            allow_duplicates=allow_duplicates,
            verbose=verbose,
        )

    def set_bipolar_reference(self):
        """Set a bipolar reference. Not implemented."""
        self._check_connected_and_regular_sampling("set_bipolar_reference()")
        raise NotImplementedError

    @fill_doc
    def set_channel_types(
        self,
        mapping: dict[str, str],
        *,
        on_unit_change: str = "warn",
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> None:
        """Define the sensor type of channels.

        If the new channel type changes the unit type, e.g. from ``T/m`` to ``V``, the
        unit multiplication factor is reset to ``0``. Use
        ``Stream.set_channel_units`` to change the multiplication factor, e.g. from
        ``0`` to ``-6`` to change from Volts to microvolts.

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
        self._check_connected(name="set_channel_types()")
        super().set_channel_types(
            mapping=mapping, on_unit_change=on_unit_change, verbose=verbose
        )

    def set_channel_units(self, mapping: dict[str, Union[str, int]]) -> None:
        """Define the channel unit multiplication factor.

        The unit itself is defined by the sensor type. Use
        ``Stream.set_channel_types`` to change the channel type, e.g. from planar
        gradiometers in ``T/m`` to EEG in ``V``.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping a channel to a unit, e.g. ``{'EEG061': 'microvolts'}``.
            The unit can be given as a human-readable string or as a unit multiplication
            factor, e.g. ``-6`` for microvolts corresponding to ``1e-6``.

        Notes
        -----
        If the human-readable unit of your channel is not yet supported by MNE-LSL,
        please contact the developers on GitHub to add your units to the known set.
        """
        self._check_connected(name="set_channel_units()")
        _set_channel_units(self._info, mapping)

    @fill_doc
    def set_eeg_reference(
        self,
        ref_channels: Union[str, list[str], tuple[str]],
        ch_type: Union[str, list[str], tuple[str]] = "eeg",
    ) -> None:
        """Specify which reference to use for EEG-like data.

        Use this function to explicitly specify the desired reference for EEG-like
        channels. This can be either an existing electrode or a new virtual channel
        added with ``Stream.add_reference_channels``. This function will re-reference
        the data in the ringbuffer according to the desired reference.

        Parameters
        ----------
        ref_channels : str | list of str
            Name(s) of the channel(s) used to construct the reference. Can also be set
            to ``'average'`` to apply a common average reference.
        ch_type : str | list of str
            The name of the channel type to apply the reference to. Valid channel types
            are ``'eeg'``, ``'ecog'``, ``'seeg'``, ``'dbs'``.
        """
        self._check_connected_and_regular_sampling("set_eeg_reference()")

        # allow only one-call to this function for simplicity, and if one day someone
        # want to apply 2 or more different reference to 2 or more types of channels,
        # then we can remove this limitation.
        if self._info["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_ON:
            raise RuntimeError(
                "The method Stream.set_eeg_reference() can only be called once. "
                "If you want to change the reference of this Stream, please disconnect "
                "and reconnect to reset the Stream."
            )

        if isinstance(ch_type, str):
            ch_type = [ch_type]
        check_type(ch_type, (tuple, list), "ch_type")
        for type_ in ch_type:
            if type_ not in self:
                raise ValueError(
                    f"There are no channels of type {type_} in this stream."
                )
        picks = _picks_to_idx(self._info, ch_type, "all", (), allow_empty=False)
        if ref_channels == "average":
            picks_ref = picks
        else:
            if isinstance(ref_channels, str):
                ref_channels = [ref_channels]
            check_type(ref_channels, (tuple, list), "ref_channels")
            picks_ref = _picks_to_idx(
                self._info, ref_channels, "all", (), allow_empty=False
            )
        if np.intersect1d(picks, picks_ref, assume_unique=True).size == 0:
            raise ValueError(
                f"The new reference channel(s) must be of the type(s) {ch_type} "
                "provided in the argument 'ch_type'."
            )

        with self._interrupt_acquisition():
            self._ref_channels = picks_ref
            self._ref_from = picks
            data_ref = self._buffer[:, self._ref_channels].mean(axis=1, keepdims=True)
            self._buffer[:, self._ref_from] -= data_ref
            with self._info._unlock():
                self._info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

    def set_meas_date(
        self, meas_date: Optional[Union[datetime, float, tuple[float]]]
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
        self._check_connected(name="set_meas_date()")
        super().set_meas_date(meas_date)

    @fill_doc
    def set_montage(
        self,
        montage: Optional[Union[str, DigMontage]],
        match_case: bool = True,
        match_alias: Union[bool, dict[str, str]] = False,
        on_missing: str = "raise",
        *,
        verbose: Optional[Union[bool, str, int]] = None,
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
        self._check_connected(name="set_montage()")
        super().set_montage(
            montage=montage,
            match_case=match_case,
            match_alias=match_alias,
            on_missing=on_missing,
            verbose=verbose,
        )

    @staticmethod
    def _acquire(self) -> None:  # pragma: no cover
        """Update function pulling new samples in the buffer at a regular interval."""
        pass

    def _check_connected(self, name: str):
        """Check that the stream is connected before calling the function 'name'."""
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                f"use {type(self).__name__}.{name}. Please connect to the stream to "
                "create the Info."
            )

    def _check_connected_and_regular_sampling(self, name: str):
        """Check that the stream has a regular sampling rate."""
        self._check_connected(name)
        if self._info["sfreq"] == 0:
            raise RuntimeError(
                f"The method {type(self).__name__}.{name} can not be used on a stream "
                "with an irregular sampling rate."
            )

    def _create_acquisition_thread(self, delay: float) -> None:
        """Create and start the daemonic acquisition thread.

        Parameters
        ----------
        delay : float
            Delay after which the thread will call the acquire function.
        """
        self._acquisition_thread = Timer(delay, self._acquire)
        self._acquisition_thread.daemon = True
        self._acquisition_thread.start()

    @contextmanager
    def _interrupt_acquisition(self):
        """Context manager interrupting the acquisition thread."""
        if not self.connected:
            raise RuntimeError(
                "Interruption of the acquisition thread was requested but the stream "
                "is not connected. Please open an issue on GitHub and provide the "
                "error traceback to the developers."
            )
        self._interrupt = True
        while self._acquisition_thread.is_alive():
            self._acquisition_thread.cancel()
        try:  # ensure "finally" is reached even when failures occur
            yield
        finally:
            self._interrupt = False
            self._create_acquisition_thread(0)

    def _pick(self, picks: NDArray[+ScalarIntType]) -> None:
        """Interrupt acquisition and apply the channel selection."""
        # for simplicity, don't allow to select channels after a reference schema has
        # been set, even if we drop channels which are not part of the reference schema,
        # else, we need to figure out where the dropped channel(s) are located relative
        # to _ref_channels and _ref_from and edit those 2 variables accordingly.
        if self._info["custom_ref_applied"] == FIFF.FIFFV_MNE_CUSTOM_REF_ON:
            raise RuntimeError(
                "The channel selection must be done before adding a re-refenrecing "
                "schema with Stream.set_eeg_reference()."
            )

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
            for ch in self._added_channels[::-1]:
                if ch not in self.ch_names:
                    self._added_channels.remove(ch)

    @abstractmethod
    def _reset_variables(self) -> None:
        """Reset variables define after connection."""
        self._acquisition_thread = None
        self._acquisition_delay = None
        self._info = None
        self._interrupt = False
        self._buffer = None
        self._n_new_samples = None
        self._picks_inlet = None
        self._added_channels = []
        self._ref_channels = None
        self._ref_from = None
        self._timestamps = None
        # This method needs to reset any stream-system-specific variables, e.g. an inlet
        # or a StreamInfo for LSL streams.

    # ----------------------------------------------------------------------------------
    @property
    def compensation_grade(self) -> Optional[int]:
        """The current gradient compensation grade.

        :type: :class:`int` | None
        """
        self._check_connected(name="compensation_grade")
        return super().compensation_grade

    # ----------------------------------------------------------------------------------
    @property
    def ch_names(self) -> list[str]:
        """Name of the channels.

        :type: :class:`list` of :class:`str`
        """
        self._check_connected(name="ch_names")
        return self._info.ch_names

    @property
    def connected(self) -> bool:
        """Connection status of the stream.

        :type: :class:`bool`
        """
        attributes = (
            "_info",
            "_acquisition_delay",
            "_acquisition_thread",
            "_buffer",
            "_picks_inlet",
            "_timestamps",
        )
        if all(getattr(self, attr, None) is None for attr in attributes):
            return False
        else:
            # sanity-check
            assert not any(getattr(self, attr, None) is None for attr in attributes)
            return True

    @property
    def dtype(self) -> Optional[DTypeLike]:
        """Channel format of the stream."""
        return getattr(self._buffer, "dtype", None)

    @property
    def info(self) -> Info:
        """Info of the LSL stream.

        :type: :class:`~mne.Info`
        """
        self._check_connected(name="info")
        return self._info

    @property
    def n_buffer(self) -> int:
        """Number of samples that can be stored in the buffer.

        :type: :class:`int`
        """
        self._check_connected(name="n_buffer")
        return self._timestamps.size

    @property
    def n_new_samples(self) -> int:
        """Number of new samples available in the buffer.

        The number of new samples is reset at every ``Stream.get_data`` call.

        :type: :class:`int`
        """
        self._check_connected(name="n_new_samples")
        return self._n_new_samples
