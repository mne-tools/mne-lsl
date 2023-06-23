# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from contextlib import contextmanager
from math import ceil
from threading import Timer
from typing import TYPE_CHECKING

import numpy as np
from mne import pick_info
from mne.channels import rename_channels
from mne.channels.channels import SetChannelsMixin
from mne.io.meas_info import ContainsMixin
from mne.io.pick import _picks_to_idx

from .lsl import StreamInlet, resolve_streams
from .lsl.constants import fmt2numpy
from .utils._checks import check_type
from .utils._docs import copy_doc, fill_doc
from .utils._exceptions import _GH_ISSUES
from .utils.logs import logger
from .utils.meas_info import create_info, _set_channel_units

if TYPE_CHECKING:
    from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

    from mne import Info
    from mne.channels import DigMontage
    from numpy.typing import NDArray


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
        self._buffer = None
        self._timestamps = None
        self._picks = None  # picks defines the channel selected from the StreamInlet
        self._acquisition_delay = None
        self._acquisition_thread = None

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

    def connect(
        self,
        processing_flags: Optional[Union[str, Sequence[str]]] = None,
        timeout: Optional[float] = 10,
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
        """
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
        self._sinfo = sinfos[0]
        # create inlet
        self._inlet = StreamInlet(self._sinfo, processing_flags=processing_flags)
        self._inlet.open_stream(timeout=timeout)
        # create MNE info from the LSL stream info returned by an open stream inlet
        self._info = create_info(
            self._sinfo.n_channels,
            self._sinfo.sfreq,
            self._sinfo.stype,
            self._inlet.get_sinfo(),
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
        self._picks = np.arange(0, self._inlet.n_channels)

        # define the acquisition thread
        self._acquisition_delay = acquisition_delay
        self._acquisition_thread = Timer(1 / self._acquisition_delay, self._acquire)
        self._acquisition_thread.start()

    def disconnect(self) -> None:
        """Disconnect from the LSL stream and interrupt data collection."""
        while self._acquisition_thread.is_alive():
            self._acquisition_thread.cancel()
        self._inlet.close_stream()
        del self._inlet

        # reset variables defined after resolution and connection
        self._sinfo = None
        self._inlet = None
        self._info = None
        self._buffer = None
        self._timestamps = None
        self._picks = None
        self._acquisition_delay = None
        self._acquisition_thread = None

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
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "drop channels. Please connect to the stream to create the Info."
            )

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
        self._info = pick_info(self._info, picks)
        with self._interrupt_acquisition():
            self._picks = self._picks[picks]
            self._buffer = self._buffer[:, picks]

    @copy_doc(ContainsMixin.get_channel_types)
    def get_channel_types(
        self, picks=None, unique=False, only_data_chs=False
    ) -> List[str]:
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "retrieve the channel types. Please connect to the stream to create "
                "the Info."
            )
        return super().get_channel_types(
            picks=picks, unique=unique, only_data_chs=only_data_chs
        )

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
                    "stream. " + _GH_ISSUES
                )
            raise

    @copy_doc(SetChannelsMixin.get_montage)
    def get_montage(self) -> Optional[DigMontage]:
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "retrieve the channel montage. Please connect to the stream to create "
                "the Info."
            )
        return super().get_montage()

    def load_stream_config(self) -> None:
        pass

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
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "pick channels. Please connect to the stream to create the Info."
            )

        picks = _picks_to_idx(self._info, picks, "all", exclude, allow_empty=False)
        self._info = pick_info(self._info, picks)
        with self._interrupt_acquisition():
            self._picks = self._picks[picks]
            self._buffer = self._buffer[:, picks]

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
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "rename channels. Please connect to the stream to create the Info."
            )
        rename_channels(
            self._info,
            mapping=mapping,
            allow_duplicates=allow_duplicates,
            verbose=verbose,
        )

    def reorder_channels(self, ch_names) -> None:
        """Reorder channels.

        Parameters
        ----------
        ch_names : list of str
            The desired channel order.
        """
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "reorder the channels. Please connect to the stream to create the Info."
            )

        check_type(ch_names, (list, tuple), "ch_names")
        try:
            idx = np.array([self._info.ch_names.index(ch_name) for ch_name in ch_names])
        except ValueError:
            raise ValueError(
                "The argument 'ch_names' must contain existing channel names."
            )
        if np.unique(idx).size != len(ch_names):
            raise ValueError(
                "The argument 'ch_names' must contain the desired channel order "
                "without duplicated channel name."
            )
        if len(ch_names) != len(self._info.ch_names):
            raise ValueError(
                "The argument 'ch_names' must contain all the existing channels in the "
                "desired order."
            )

        with self._interrupt_acquisition():
            self._picks = idx
            self._buffer = self._buffer[:, self._picks]

    def save_stream_config(self) -> None:
        pass

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
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "set the channel types. Please connect to the stream to create the "
                "Info."
            )
        super().set_channel_types(
            mapping=mapping, on_unit_change=on_unit_change, verbose=verbose
        )

    def set_channel_units(self, mapping: Dict[str, Union[str, int]]) -> None:
        """Define the channel unit.

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
        _set_channel_units(self._info, mapping)

    def set_eeg_reference(self) -> None:
        pass

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
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "set the channel montage. Please connect to the stream to create "
                "the Info."
            )
        super().set_montage(
            montage=montage,
            match_case=match_case,
            match_alias=match_alias,
            on_missing=on_missing,
            verbose=verbose,
        )

    def _acquire(self) -> None:
        """Update function pulling new samples in the buffer at a regular interval."""
        data, timestamps = self._inlet.pull_chunk(timeout=0.0)
        if timestamps.size != 0:
            self._buffer = np.roll(self._buffer, -data.shape[0], axis=0)
            self._timestamps = np.roll(self._timestamps, -timestamps.size, axis=0)
            self._buffer[-data.shape[0] :, :] = data[:, self._picks]
            self._timestamps[-timestamps.size :] = timestamps

        # recreate the timer thread as it is one-call only
        self._acquisition_thread = Timer(self._acquisition_delay, self._acquire)
        self._acquisition_thread.start()

    @contextmanager
    def _interrupt_acquisition(self):
        """Context manager interrupting the acquisition thread."""
        if not self.connected:
            raise RuntimeError(
                "Interruption of the acquisition thread was requested but the stream "
                "is not connected. " + _GH_ISSUES
            )

        while self._acquisition_thread.is_alive():
            self._acquisition_thread.cancel()
        yield
        self._acquisition_thread = Timer(self._acquisition_delay, self._acquire)
        self._acquisition_thread.start()

    # ----------------------------------------------------------------------------------
    @property
    def compensation_grade(self) -> Optional[int]:
        """The current gradient compensation grade.

        :type: `int` | None
        """
        if not self.connected:
            raise RuntimeError(
                "The Stream attribute 'info' is None. An Info instance is required to "
                "retrieve the current gradient compensation grade. Please connect to "
                "the stream to create the Info."
            )
        return super().compensation_grade

    # ----------------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        """Connection status of the stream.

        :type: `bool`
        """
        attributes = (
            "_sinfo",
            "_info",
            "_inlet",
            "_buffer",
            "_timestamps",
            "_picks",
            "_acquisition_delay",
            "_acquisition_thread",
        )
        if all(getattr(self, attr) is None for attr in attributes):
            return False
        else:
            assert not any(getattr(self, attr) is None for attr in attributes)
            return True

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
