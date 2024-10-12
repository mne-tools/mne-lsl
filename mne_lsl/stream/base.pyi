from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from datetime import datetime as datetime
from typing import Any

import numpy as np
from _typeshed import Incomplete
from mne import Info
from mne.channels import DigMontage as DigMontage
from mne.channels.channels import SetChannelsMixin
from mne.io.meas_info import ContainsMixin
from numpy.typing import DTypeLike as DTypeLike
from numpy.typing import NDArray as NDArray

from .._typing import ScalarArray as ScalarArray
from .._typing import ScalarIntArray as ScalarIntArray
from ..utils._checks import check_type as check_type
from ..utils._checks import check_value as check_value
from ..utils._checks import ensure_int as ensure_int
from ..utils._docs import copy_doc as copy_doc
from ..utils._docs import fill_doc as fill_doc
from ..utils._time import high_precision_sleep as high_precision_sleep
from ..utils.logs import logger as logger
from ..utils.logs import verbose as verbose
from ..utils.logs import warn as warn
from ..utils.meas_info import _HUMAN_UNITS as _HUMAN_UNITS
from ..utils.meas_info import _set_channel_units as _set_channel_units
from ._filters import StreamFilter as StreamFilter
from ._filters import create_filter as create_filter
from ._filters import ensure_sos_iir_params as ensure_sos_iir_params

class BaseStream(ABC, ContainsMixin, SetChannelsMixin):
    """Stream object representing a single real-time stream.

    Parameters
    ----------
    bufsize : float | int
        Size of the buffer keeping track of the data received from the stream. If
        the stream sampling rate ``sfreq`` is regular, ``bufsize`` is expressed in
        seconds. The buffer will hold the last ``bufsize * sfreq`` samples (ceiled).
        If the stream sampling rate ``sfreq`` is irregular, ``bufsize`` is
        expressed in samples. The buffer will hold the last ``bufsize`` samples.
    """

    _bufsize: Incomplete

    @abstractmethod
    def __init__(self, bufsize: float): ...
    def __contains__(self, ch_type: str) -> bool:
        """Check channel type membership.

        Parameters
        ----------
        ch_type : str
            Channel type to check for. Can be e.g. ``'meg'``, ``'eeg'``,
            ``'stim'``, etc.

        Returns
        -------
        in : bool
            Whether or not the instance contains the given channel type.

        Examples
        --------
        Channel type membership can be tested as::

            >>> 'meg' in inst  # doctest: +SKIP
            True
            >>> 'seeg' in inst  # doctest: +SKIP
            False

        """

    def __del__(self) -> None:
        """Try to disconnect the stream when deleting the object."""

    @abstractmethod
    def __repr__(self) -> str:
        """Representation of the instance."""

    @abstractmethod
    def acquire(self) -> None:
        """Pull new samples in the buffer.

        Notes
        -----
        This method is not needed if the stream was connected with an acquisition delay
        different from ``0``. In this case, the acquisition is done automatically in a
        background thread.
        """
    _buffer: Incomplete

    def add_reference_channels(
        self,
        ref_channels: str | list[str] | tuple[str],
        ref_units: str | int | list[str | int] | tuple[str | int, ...] | None = None,
    ) -> BaseStream:
        """Add EEG reference channels to data that consists of all zeros.

        Adds EEG reference channels that are not part of the streamed data. This is
        useful when you need to re-reference your data to different channels. These
        added channels will consist of all zeros.

        Parameters
        ----------
        ref_channels : str | list of str
            Name of the electrode(s) which served as the reference in the
            recording. If a name is provided, a corresponding channel is added
            and its data is set to 0. This is useful for later re-referencing.
        ref_units : str | int | list of str | list of int | None
            The unit or unit multiplication factor of the reference channels. The unit
            can be given as a human-readable string or as a unit multiplication factor,
            e.g. ``-6`` for microvolts corresponding to ``1e-6``.
            If not provided, the added EEG reference channel has a unit multiplication
            factor set to ``0`` which corresponds to Volts. Use
            ``Stream.set_channel_units`` to change the unit multiplication factor.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        Notes
        -----
        Read about the :ref:`processing applied to the underlying
        buffer <resources/implementations:StreamLSL>`.
        """

    def anonymize(
        self,
        daysback: int | None = None,
        keep_his: bool = False,
        *,
        verbose: bool | str | int | None = None,
    ) -> BaseStream:
        """Anonymize the measurement information in-place.

        Parameters
        ----------
        daysback : int | None
            Number of days to subtract from all dates.
            If ``None`` (default), the acquisition date, ``info['meas_date']``,
            will be set to ``January 1ˢᵗ, 2000``. This parameter is ignored if
            ``info['meas_date']`` is ``None`` (i.e., no acquisition date has been set).
        keep_his : bool
            If ``True``, ``his_id`` of ``subject_info`` will **not** be overwritten.
            Defaults to ``False``.

            .. warning:: This could mean that ``info`` is not fully
                         anonymized. Use with caution.
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between
            ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
            If None is provided, the verbosity is set to the currently set logger's level.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        Notes
        -----
        Removes potentially identifying information if it exists in ``info``.
        Specifically for each of the following we use:

        - meas_date, file_id, meas_id
                A default value, or as specified by ``daysback``.
        - subject_info
                Default values, except for 'birthday' which is adjusted
                to maintain the subject age.
        - experimenter, proj_name, description
                Default strings.
        - utc_offset
                ``None``.
        - proj_id
                Zeros.
        - proc_history
                Dates use the ``meas_date`` logic, and experimenter a default string.
        - helium_info, device_info
                Dates use the ``meas_date`` logic, meta info uses defaults.

        If ``info['meas_date']`` is ``None``, it will remain ``None`` during processing
        the above fields.

        Operates in place.
        """
    _acquisition_delay: Incomplete
    _n_new_samples: int
    _executor: Incomplete

    @abstractmethod
    def connect(self, acquisition_delay: float) -> BaseStream:
        """Connect to the stream and initiate data collection in the buffer.

        Parameters
        ----------
        acquisition_delay : float
            Delay in seconds between 2 acquisition during which chunks of data are
            pulled from the connected device. If ``0``, the automatic acquisition in a
            background thread is disabled and the user must manually call the
            acquisition method to pull new samples.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.
        """

    @abstractmethod
    def disconnect(self) -> BaseStream:
        """Disconnect from the LSL stream and interrupt data collection.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.
        """

    def del_filter(self, idx: int | list[int] | tuple[int] | str = "all") -> None:
        """Remove a filter from the list of applied filters.

        Parameters
        ----------
        idx : ``'all'`` | int | list of int | tuple of int
            If the string ``'all'`` (default), remove all filters. If an integer or a
            list of integers, remove the filter(s) at the given index(es) from
            ``Stream.filters``.

        Notes
        -----
        When removing a filter, the initial conditions of all the filters applied on
        overlapping channels are reset. The initial conditions will be re-estimated as
        a step response steady-state.
        """

    def drop_channels(self, ch_names: str | list[str] | tuple[str]) -> BaseStream:
        """Drop channel(s).

        Parameters
        ----------
        ch_names : str | list of str
            Name or list of names of channels to remove.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        See Also
        --------
        pick
        """

    def filter(
        self,
        l_freq: float | None,
        h_freq: float | None,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        iir_params: dict[str, Any] | None = None,
        *,
        verbose: bool | str | int | None = None,
    ) -> BaseStream:
        """Filter the stream with an IIR causal filter.

        Once a filter is applied, the buffer is updated in real-time with the filtered
        data. It is possible to apply more than one filter.

        .. code-block:: python

            stream = Stream(2.0).connect()
            stream.filter(1.0, 40.0, picks="eeg")
            stream.filter(1.0, 15.0, picks="ecg").filter(0.1, 5, picks="EDA")

        Parameters
        ----------
        l_freq : float | None
            The lower cutoff frequency. If None, the buffer is only low-passed.
        h_freq : float | None
            The higher cutoff frequency. If None, the buffer is only high-passed.
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Note that
            channels in ``info['bads']`` *will be included* if their names or indices
            are explicitly provided.
        iir_params : dict | None
            Dictionary of parameters to use for IIR filtering. If None, a 4th order
            Butterworth will be used. For more information, see
            :func:`mne.filter.construct_iir_filter`.

            .. note::

                The output ``sos`` must be used. The ``ba`` output is not supported.
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between
            ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
            If None is provided, the verbosity is set to the currently set logger's level.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        Notes
        -----
        Read about the :ref:`processing applied to the underlying
        buffer <resources/implementations:StreamLSL>`.
        """

    def get_channel_types(
        self,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        unique: bool = False,
        only_data_chs: bool = False,
    ) -> list[str]:
        """Get a list of channel type for each channel.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Note that
            channels in ``info['bads']`` *will be included* if their names or indices
            are explicitly provided.
        unique : bool
            Whether to return only unique channel types. Default is ``False``.
        only_data_chs : bool
            Whether to ignore non-data channels. Default is ``False``.

        Returns
        -------
        channel_types : list
            The channel types.
        """

    def get_channel_units(
        self,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        only_data_chs: bool = False,
    ) -> list[tuple[int, int]]:
        """Get a list of channel unit for each channel.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Note that
            channels in ``info['bads']`` *will be included* if their names or indices
            are explicitly provided.
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

    def get_data(
        self,
        winsize: float | None = None,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        exclude: str | list[str] | tuple[str] = "bads",
    ) -> tuple[ScalarArray, NDArray[np.float64]]:
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
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Note that
            channels in ``info['bads']`` *will be included* if their names or indices
            are explicitly provided.
        exclude : str | list of str | tuple of str
            Set of channels to exclude, only used when picking based on types (e.g.,
            ``exclude="bads"`` when ``picks="meg"``) or when picking is set to ``None``.

        Returns
        -------
        data : array of shape (n_channels, n_samples)
            Data in the given window.
        timestamps : array of shape (n_samples,)
            Timestamps in the given window.

        Notes
        -----
        The number of newly available samples stored in the property ``n_new_samples``
        is reset at every function call, even if all channels were not selected with the
        argument ``picks``.
        """

    def get_montage(self) -> DigMontage | None:
        """Get a DigMontage from instance.

        Returns
        -------

        montage : None | str | DigMontage
            A montage containing channel positions. If a string or
            :class:`~mne.channels.DigMontage` is
            specified, the existing channel information will be updated with the
            channel positions from the montage. Valid strings are the names of the
            built-in montages that ship with MNE-Python; you can list those via
            :func:`mne.channels.get_builtin_montages`.
            If ``None`` (default), the channel positions will be removed from the
            :class:`~mne.Info`.
        """

    def notch_filter(
        self,
        freqs: float,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        notch_widths: float | None = None,
        trans_bandwidth: int = 1,
        iir_params: dict[str, Any] | None = None,
        *,
        verbose: bool | str | int | None = None,
    ) -> BaseStream:
        """Filter the stream with an IIR causal notch filter.

        Once a filter is applied, the buffer is updated in real-time with the filtered
        data. It is possible to apply more than one filter.

        .. code-block:: python

            stream = Stream(2.0).connect()
            stream.filter(1.0, 40.0, picks="eeg")
            stream.notch_filter(50, picks="ecg")

        Parameters
        ----------
        freqs : float
            Specific frequencies to filter out from data, e.g. ``60`` Hz in the US or
            ``50`` Hz in Europe for line noise.
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Note that
            channels in ``info['bads']`` *will be included* if their names or indices
            are explicitly provided.
        notch_widths : float | None
            Width of the stop band in Hz. If ``None``, ``freqs / 200`` is used.
        trans_bandwidth : float
            Width of the transition band in Hz.
        iir_params : dict | None
            Dictionary of parameters to use for IIR filtering. If None, a 4th order
            Butterworth will be used. For more information, see
            :func:`mne.filter.construct_iir_filter`.

            .. note::

                The output ``sos`` must be used. The ``ba`` output is not supported.
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between
            ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
            If None is provided, the verbosity is set to the currently set logger's level.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        Notes
        -----
        Read about the :ref:`processing applied to the underlying
        buffer <resources/implementations:StreamLSL>`.
        """

    def plot(self) -> None:
        """Open a real-time stream viewer. Not implemented."""

    def pick(
        self,
        picks: str | list[str] | int | list[int] | ScalarIntArray | None = None,
        exclude: str | list[str] | int | list[int] | ScalarIntArray = (),
    ) -> BaseStream:
        """Pick a subset of channels.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels.
        exclude : str | list of str
            Set of channels to exclude, only used when picking is based on types, e.g.
            ``exclude='bads'`` when ``picks="meg"``.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        See Also
        --------
        drop_channels

        Notes
        -----
        Contrary to MNE-Python, re-ordering channels is not supported in ``MNE-LSL``.
        Thus, if explicit channel names are provided in ``picks``, they are sorted to
        match the order of existing channel names.
        """

    def record(self) -> None:
        """Record the stream data to disk. Not implemented."""

    def rename_channels(
        self,
        mapping: dict[str, str] | Callable,
        allow_duplicates: bool = False,
        *,
        verbose: bool | str | int | None = None,
    ) -> BaseStream:
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
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between
            ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
            If None is provided, the verbosity is set to the currently set logger's level.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.
        """

    def set_bipolar_reference(self) -> BaseStream:
        """Set a bipolar reference. Not implemented.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.
        """

    def set_channel_types(
        self,
        mapping: dict[str, str],
        *,
        on_unit_change: str = "warn",
        verbose: bool | str | int | None = None,
    ) -> BaseStream:
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
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between
            ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
            If None is provided, the verbosity is set to the currently set logger's level.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.
        """

    def set_channel_units(self, mapping: dict[str, str | int]) -> BaseStream:
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

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        Notes
        -----
        If the human-readable unit of your channel is not yet supported by MNE-LSL,
        please contact the developers on GitHub to add your units to the known set.
        """
    _ref_channels: Incomplete
    _ref_from: Incomplete

    def set_eeg_reference(
        self,
        ref_channels: str | list[str] | tuple[str],
        ch_type: str | list[str] | tuple[str] = "eeg",
    ) -> BaseStream:
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

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        Notes
        -----
        Read about the :ref:`processing applied to the underlying
        buffer <resources/implementations:StreamLSL>`.
        """

    def set_meas_date(
        self, meas_date: datetime | float | tuple[float] | None
    ) -> BaseStream:
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

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        See Also
        --------
        anonymize
        """

    def set_montage(
        self,
        montage: str | DigMontage | None,
        match_case: bool = True,
        match_alias: bool | dict[str, str] = False,
        on_missing: str = "raise",
        *,
        verbose: bool | str | int | None = None,
    ) -> BaseStream:
        """Set EEG/sEEG/ECoG/DBS/fNIRS channel positions and digitization points.

        Parameters
        ----------
        montage : None | str | DigMontage
            A montage containing channel positions. If a string or
            :class:`~mne.channels.DigMontage` is
            specified, the existing channel information will be updated with the
            channel positions from the montage. Valid strings are the names of the
            built-in montages that ship with MNE-Python; you can list those via
            :func:`mne.channels.get_builtin_montages`.
            If ``None`` (default), the channel positions will be removed from the
            :class:`~mne.Info`.
        match_case : bool
            If True (default), channel name matching will be case sensitive.

            .. versionadded:: MNE  0.20
        match_alias : bool | dict
            Whether to use a lookup table to match unrecognized channel location names
            to their known aliases. If True, uses the mapping in
            ``mne.io.constants.CHANNEL_LOC_ALIASES``. If a :class:`dict` is passed, it
            will be used instead, and should map from non-standard channel names to
            names in the specified ``montage``. Default is ``False``.

            .. versionadded:: MNE  0.23
        on_missing : 'raise' | 'warn' | 'ignore'
            Can be ``'raise'`` (default) to raise an error, ``'warn'`` to emit a
            warning, or ``'ignore'`` to ignore when channels have missing coordinates.

            .. versionadded:: MNE  0.20.1
        verbose : int | str | bool | None
            Sets the verbosity level. The verbosity increases gradually between
            ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
            If None is provided, the verbosity is set to the currently set logger's level.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        stream : instance of ``Stream``
            The stream instance modified in-place.

        See Also
        --------
        mne.channels.make_standard_montage
        mne.channels.make_dig_montage
        mne.channels.read_custom_montage

        Notes
        -----
        .. warning::

            Only EEG/sEEG/ECoG/DBS/fNIRS channels can have their positions set using a
            montage. Other channel types (e.g., MEG channels) should have their
            positions defined properly using their data reading functions.
        """

    @abstractmethod
    def _acquire(self) -> None:
        """Update function pulling new samples in the buffer at a regular interval."""

    def _check_connected(self, name: str) -> None:
        """Check that the stream is connected before calling the function 'name'."""

    def _check_connected_and_regular_sampling(self, name: str) -> None:
        """Check that the stream has a regular sampling rate."""

    def _check_not_epoched(self, name: str) -> None:
        """Check that the stream is not being epoched."""

    def _interrupt_acquisition(self) -> Generator[None, None, None]:
        """Context manager interrupting the acquisition thread."""
    _info: Incomplete
    _picks_inlet: Incomplete

    def _pick(self, picks: ScalarIntArray) -> None:
        """Interrupt acquisition and apply the channel selection."""
    _added_channels: Incomplete
    _epochs: Incomplete
    _filters: Incomplete
    _timestamps: Incomplete

    @abstractmethod
    def _reset_variables(self) -> None:
        """Reset variables define after connection."""

    def _submit_acquisition_job(self) -> None:
        """Submit a new acquisition job, if applicable."""

    @property
    def compensation_grade(self) -> int | None:
        """The current gradient compensation grade.

        :type: :class:`int` | None
        """

    @property
    def ch_names(self) -> list[str]:
        """Name of the channels.

        :type: :class:`list` of :class:`str`
        """

    @property
    def connected(self) -> bool:
        """Connection status of the stream.

        :type: :class:`bool`
        """

    @property
    def dtype(self) -> DTypeLike | None:
        """Channel format of the stream."""

    @property
    def filters(self) -> list[StreamFilter]:
        """List of filters applied to the real-time Stream.

        :type: :class:`list` of ```StreamFilter``
        """

    @property
    def info(self) -> Info:
        """Info of the LSL stream.

        :type: :class:`~mne.Info`
        """

    @property
    def n_buffer(self) -> int:
        """Number of samples that can be stored in the buffer.

        :type: :class:`int`
        """

    @property
    def n_new_samples(self) -> int:
        """Number of new samples available in the buffer.

        The number of new samples is reset at every ``Stream.get_data`` call.

        :type: :class:`int`
        """
