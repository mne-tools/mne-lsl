from __future__ import annotations  # c.f. PEP 563, PEP 649

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from mne import rename_channels
from mne.io import read_raw
from mne.utils import check_version

if check_version("mne", "1.5"):
    from mne.io.meas_info import ContainsMixin, SetChannelsMixin
    from mne.io.pick import _picks_to_idx
elif check_version("mne", "1.6"):
    from mne._fiff.meas_info import ContainsMixin, SetChannelsMixin
    from mne._fiff.pick import _picks_to_idx
else:
    from mne.io.meas_info import ContainsMixin
    from mne.io.pick import _picks_to_idx
    from mne.channels.channels import SetChannelsMixin

from ..utils._checks import check_type, ensure_int, ensure_path
from ..utils._docs import fill_doc
from ..utils.logs import logger
from ..utils.meas_info import _set_channel_units

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path
    from typing import Callable, Dict, List, Optional, Tuple, Union

    from mne import Info


class BasePlayer(ABC, ContainsMixin, SetChannelsMixin):
    """Class for creating a mock real-time stream.

    Parameters
    ----------
    fname : path-like
        Path to the file to re-play as a mock real-time stream. MNE-Python must be able
        to load the file with :func:`mne.io.read_raw`.
    chunk_size : int ``≥ 1``
        Number of samples pushed at once on the mock real-time stream.

    Notes
    -----
    The file re-played is loaded in memory. Thus, large files are not recommended. Once
    the end-of-file is reached, the player loops back to the beginning which can lead to
    a small discontinuity in the data stream.
    """

    @abstractmethod
    def __init__(self, fname: Union[str, Path], chunk_size: int = 16) -> None:
        self._fname = ensure_path(fname, must_exist=True)
        self._chunk_size = ensure_int(chunk_size, "chunk_size")
        if self._chunk_size <= 0:
            raise ValueError(
                "The argument 'chunk_size' must be a strictly positive integer. "
                f"{chunk_size} is invalid."
            )
        # load raw recording
        self._raw = read_raw(self._fname, preload=True)
        # This method should end on a self._reset_variables()

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
        self._check_not_started("anonymize()")
        logger.warning(
            "Player.anonymize() is partially implemented and does not impact the "
            "stream information yet."
        )
        super().anonymize(daysback=daysback, keep_his=keep_his, verbose=verbose)

    @fill_doc
    def get_channel_units(
        self, picks=None, only_data_chs: bool = False
    ) -> List[Tuple[int, int]]:
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
        check_type(only_data_chs, (bool,), "only_data_chs")
        none = "data" if only_data_chs else "all"
        picks = _picks_to_idx(self.info, picks, none, (), allow_empty=False)
        channel_units = list()
        for idx in picks:
            channel_units.append(
                (self.info["chs"][idx]["unit"], self.info["chs"][idx]["unit_mul"])
            )
        return channel_units

    @abstractmethod
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
        %(verbose)s
        """
        self._check_not_started("rename_channels()")
        rename_channels(self.info, mapping, allow_duplicates)

    @abstractmethod
    def start(self) -> None:
        """Start streaming data."""
        pass

    @abstractmethod
    @fill_doc
    def set_channel_types(
        self, mapping: Dict[str, str], *, on_unit_change: str = "warn", verbose=None
    ) -> None:
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
        %(verbose)s
        """
        self._check_not_started("set_channel_types()")
        super().set_channel_types(
            mapping=mapping, on_unit_change=on_unit_change, verbose=verbose
        )
        self._sinfo.set_channel_types(self.get_channel_types(unique=False))

    @abstractmethod
    def set_channel_units(self, mapping: Dict[str, Union[str, int]]) -> None:
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

        Notes
        -----
        If the human-readable unit of your channel is not yet supported by MNE-LSL,
        please contact the developers on GitHub to add your units to the known set.
        """
        self._check_not_started("set_channel_units()")
        ch_units_before = np.array(
            [ch["unit_mul"] for ch in self.info["chs"]], dtype=np.int8
        )
        _set_channel_units(self.info, mapping)
        ch_units_after = np.array(
            [ch["unit_mul"] for ch in self.info["chs"]], dtype=np.int8
        )
        # re-scale channels
        factors = ch_units_before - ch_units_after
        self._raw.apply_function(
            lambda x: (x.T * np.power(np.ones(factors.shape) * 10, factors)).T,
            channel_wise=False,
            picks="all",
        )

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
        self._check_not_started(name=f"{type(self).__name__}.set_meas_date()")
        logger.warning(
            "Player.set_meas_date() is partially implemented and does not impact the "
            "stream information yet."
        )
        super().set_meas_date(meas_date)

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming data on the mock real-time stream."""
        if self._streaming_thread is None:
            raise RuntimeError(
                "The player is not started. Use Player.start() to begin streaming."
            )
        self._interrupt = True
        while self._streaming_thread.is_alive():
            self._streaming_thread.cancel()
        # This method must end with self._reset_variables()

    def _check_not_started(self, name: str):
        """Check that the player is not started before calling the function 'name'."""
        if self._streaming_thread is not None:
            raise RuntimeError(
                "The player is already started. Please stop the streaming before using "
                f"{{type(self).__name__}}.{name}."
            )

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
        pass

    def _reset_variables(self) -> None:
        """Reset variables for streaming."""
        self._start_idx = 0
        self._streaming_delay = None
        self._streaming_thread = None
        self._interrupt = False

    # ----------------------------------------------------------------------------------
    def __del__(self):
        """Delete the player."""
        if hasattr(self, "_streaming_thread") and self._streaming_thread is not None:
            self.stop()

    def __enter__(self):
        """Context manager entry point."""
        self.start()

    def __exit__(self, exc_type, exc_value, exc_tracebac):
        """Context manager exit point."""
        self.stop()

    @staticmethod
    def __repr__(self):
        """Representation of the instance."""
        # This method must define the string representation of the player, e.g.
        # <Player: {self._fname}>
        pass

    # ----------------------------------------------------------------------------------
    @property
    def ch_names(self) -> List[str]:
        """Name of the channels.

        :type: :class:`list` of :class:`str`
        """
        return self.info.ch_names

    @property
    def chunk_size(self) -> int:
        """Number of samples in a chunk.

        :type: :class:`int`
        """
        return self._chunk_size

    @property
    def fname(self) -> Path:
        """Path to file played.

        :type: :class:`~pathlib.Path`
        """
        return self._fname

    @property
    def info(self) -> Info:
        """Info of the real-time stream.

        :type: :class:`~mne.Info`
        """
        return self._raw.info
