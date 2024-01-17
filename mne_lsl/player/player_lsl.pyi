from pathlib import Path as Path
from typing import Callable, Optional, Union

from _typeshed import Incomplete

from ..lsl import StreamInfo as StreamInfo
from ..lsl import StreamOutlet as StreamOutlet
from ..lsl import local_clock as local_clock
from ..utils._checks import check_type as check_type
from ..utils._docs import copy_doc as copy_doc
from ..utils.logs import logger as logger
from ._base import BasePlayer as BasePlayer

class PlayerLSL(BasePlayer):
    """Class for creating a mock LSL stream.

    Parameters
    ----------
    fname : path-like
        Path to the file to re-play as a mock LSL stream. MNE-Python must be able to
        load the file with :func:`mne.io.read_raw`.
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the :class:`~mne_lsl.lsl.StreamOutlet`.
        If these chunks are too small then the thread-based timing might not work
        properly.
    name : str | None
        Name of the mock LSL stream. If ``None``, the name ``MNE-LSL-Player`` is used.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended. Once
    the end-of-file is reached, the player loops back to the beginning which can lead to
    a small discontinuity in the data stream.
    """

    _name: Incomplete
    _sinfo: Incomplete

    def __init__(
        self, fname: Union[str, Path], chunk_size: int = 64, name: Optional[str] = None
    ) -> None: ...
    def rename_channels(
        self,
        mapping: Union[dict[str, str], Callable],
        allow_duplicates: bool = False,
        *,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> PlayerLSL:
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
            If None is provided, the verbosity is set to ``"WARNING"``.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        player : instance of ``Player``
            The player instance modified in-place.
        """
    _outlet: Incomplete
    _streaming_delay: Incomplete
    _streaming_thread: Incomplete
    _target_timestamp: Incomplete

    def start(self) -> PlayerLSL:
        """Start streaming data on the LSL :class:`~mne_lsl.lsl.StreamOutlet`.

        Returns
        -------
        player : instance of :class:`~mne_lsl.player.PlayerLSL`
            The player instance modified in-place.
        """

    def set_channel_types(
        self,
        mapping: dict[str, str],
        *,
        on_unit_change: str = "warn",
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> PlayerLSL:
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
            If None is provided, the verbosity is set to ``"WARNING"``.
            If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
            to ``"INFO"`` for True.

        Returns
        -------
        player : instance of ``Player``
            The player instance modified in-place.
        """

    def set_channel_units(self, mapping: dict[str, Union[str, int]]) -> PlayerLSL:
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

    def stop(self) -> PlayerLSL:
        """Stop streaming data on the LSL :class:`~mne_lsl.lsl.StreamOutlet`.

        Returns
        -------
        player : instance of :class:`~mne_lsl.player.PlayerLSL`
            The player instance modified in-place.
        """
    _start_idx: Incomplete

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

    def _reset_variables(self) -> None:
        """Reset variables for streaming."""

    def __del__(self) -> None:
        """Delete the player and destroy the :class:`~mne_lsl.lsl.StreamOutlet`."""

    def __repr__(self) -> str:
        """Representation of the instance."""

    @property
    def name(self) -> str:
        """Name of the LSL stream.

        :type: :class:`str`
        """
