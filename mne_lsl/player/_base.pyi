from abc import ABC, abstractmethod
from datetime import datetime as datetime
from pathlib import Path
from typing import Any, Callable

from _typeshed import Incomplete
from mne import Info
from mne.channels.channels import SetChannelsMixin
from mne.io import BaseRaw
from mne.io.meas_info import ContainsMixin

from ..utils._checks import check_type as check_type
from ..utils._checks import ensure_int as ensure_int
from ..utils._checks import ensure_path as ensure_path
from ..utils._docs import fill_doc as fill_doc
from ..utils.logs import logger as logger
from ..utils.logs import verbose as verbose
from ..utils.logs import warn as warn
from ..utils.meas_info import _set_channel_units as _set_channel_units

class BasePlayer(ABC, ContainsMixin, SetChannelsMixin):
    """Class for creating a mock real-time stream.

    Parameters
    ----------
    fname : path-like | Raw
        Path to the file to re-play as a mock real-time stream. MNE-Python must be able
        to load the file with :func:`mne.io.read_raw`. An :class:`~mne.io.Raw` object
        can be provided directly.
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the mock real-time stream.
    n_repeat : int | ``np.inf``
        Number of times to repeat the file. If ``np.inf``, the file is re-played
        indefinitely.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended. Once
    the end-of-file is reached, the player loops back to the beginning which can lead to
    a small discontinuity in the data stream.
    """

    _chunk_size: Incomplete
    _n_repeat: Incomplete
    _fname: Incomplete
    _raw: Incomplete

    @abstractmethod
    def __init__(
        self,
        fname: str | Path | BaseRaw,
        chunk_size: int = 10,
        n_repeat: int | float = ...,
    ): ...
    def anonymize(
        self,
        daysback: int | None = None,
        keep_his: bool = False,
        *,
        verbose: bool | str | int | None = None,
    ) -> BasePlayer:
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
        player : instance of ``Player``
            The player instance modified in-place.

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

    def get_channel_units(
        self, picks: Incomplete | None = None, only_data_chs: bool = False
    ) -> list[tuple[int, int]]:
        """Get a list of channel unit for each channel.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values "all" to pick all channels, or "data" to pick :term:`data
            channels`. None (default) will pick all channels. Note that channels in
            ``info['bads']`` *will be included* if their names or indices are
            explicitly provided.
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

    @abstractmethod
    def rename_channels(
        self,
        mapping: dict[str, str] | Callable,
        allow_duplicates: bool = False,
        *,
        verbose: bool | str | int | None = None,
    ) -> BasePlayer:
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
        player : instance of ``Player``
            The player instance modified in-place.
        """
    _executor: Incomplete

    @abstractmethod
    def start(self) -> BasePlayer:
        """Start streaming data."""

    @abstractmethod
    def set_channel_types(
        self,
        mapping: dict[str, str],
        *,
        on_unit_change: str = "warn",
        verbose: bool | str | int | None = None,
    ) -> BasePlayer:
        """Define the sensor type of channels.

        If the new channel type changes the unit type, e.g. from ``T/m`` to ``V``, the
        unit multiplication factor is reset to ``0``. Use
        ``Player.set_channel_units`` to change the multiplication factor, e.g. from
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
        player : instance of ``Player``
            The player instance modified in-place.
        """

    @abstractmethod
    def set_channel_units(self, mapping: dict[str, str | int]) -> BasePlayer:
        """Define the channel unit multiplication factor.

        By convention, MNE stores data in SI units. But systems often stream in non-SI
        units. For instance, EEG amplifiers often stream in microvolts. Thus, to mock a
        stream from an MNE-compatible file, the data might need to be scale to match
        the unit of the system to mock. This function will both change the unit
        multiplication factor and rescale the associated data.

        The unit itself is defined by the sensor type. Change the channel type in the
        ``raw`` recording with :meth:`mne.io.Raw.set_channel_types` before providing the
        recording to the player.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping a channel to a unit, e.g. ``{'EEG061': 'microvolts'}``.
            The unit can be given as a human-readable string or as a unit multiplication
            factor, e.g. ``-6`` for microvolts corresponding to ``1e-6``.

        Returns
        -------
        player : instance of ``Player``
            The player instance modified in-place.

        Notes
        -----
        If the human-readable unit of your channel is not yet supported by MNE-LSL,
        please contact the developers on GitHub to add your units to the known set.
        """

    def set_meas_date(
        self, meas_date: datetime | float | tuple[float, float] | None
    ) -> BasePlayer:
        """Set the measurement start date.

        Parameters
        ----------
        meas_date : datetime | float | tuple | None
            The new measurement date.
            If datetime object, it must be timezone-aware and in UTC. A tuple of
            (seconds, microseconds) or float (alias for ``(meas_date, 0)``) can also be
            passed and a datetime object will be automatically created. If None, will
            remove the time reference.

        Returns
        -------
        player : instance of ``Player``
            The player instance modified in-place.

        See Also
        --------
        anonymize
        """

    @abstractmethod
    def stop(self) -> BasePlayer:
        """Stop streaming data on the mock real-time stream."""

    def _check_not_started(self, name: str):
        """Check that the player is not started before calling the function 'name'."""

    @abstractmethod
    def _stream(self) -> None:
        """Push a chunk of data from the raw object to the real-time stream.

        Don't use raw.get_data but indexing which is faster.

        >>> [In] %timeit raw[:, 0:16][0]
        >>> 19 µs ± 50.3 ns per loo
        >>> [In] %timeit raw.get_data(start=0, stop=16)
        >>> 1.3 ms ± 1.01 µs per loop
        >>> [In] %timeit np.ascontiguousarray(raw[:, 0:16][0].T)
        >>> 23.7 µs ± 183 ns per loop
        """
    _end_streaming: bool
    _n_repeated: int
    _start_idx: int
    _streaming_delay: Incomplete

    def _reset_variables(self) -> None:
        """Reset variables for streaming."""

    def __enter__(self):
        """Context manager entry point."""

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any):
        """Context manager exit point."""

    @staticmethod
    def __repr__(self) -> str:
        """Representation of the instance."""

    @property
    def ch_names(self) -> list[str]:
        """Name of the channels.

        :type: :class:`list` of :class:`str`
        """

    @property
    def chunk_size(self) -> int:
        """Number of samples in a chunk.

        :type: :class:`int`
        """

    @property
    def fname(self) -> Path | None:
        """Path to file played.

        :type: :class:`~pathlib.Path` | None
        """

    @property
    def info(self) -> Info:
        """Info of the real-time stream.

        :type: :class:`~mne.Info`
        """

    @property
    def n_repeat(self) -> int | None:
        """Number of times the file is repeated.

        :type: :class:`int` | ``np.inf``
        """

    @property
    def running(self) -> bool:
        """Status of the player, True if it is running and pushing data."""
