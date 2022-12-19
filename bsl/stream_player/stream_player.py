import multiprocessing as mp
import time
from pathlib import Path

import mne
import numpy as np

from ..lsl import StreamInfo, StreamOutlet, local_clock
from ..triggers import TriggerDef
from ..utils import find_event_channel
from ..utils._checks import _check_type
from ..utils._logs import logger


class StreamPlayer:
    """Class for playing a recorded file on LSL network in another process.

    Parameters
    ----------
    stream_name : str
        Stream's server name, displayed on LSL network.
    fif_file : file-like
        Path to the compatible raw ``.fif`` file to play.
    repeat : int | ``float('inf')``
        Number of times the stream player will loop on the FIF file before
        interrupting. Default ``float('inf')`` can be passed to never interrupt
        streaming.
    trigger_def : None | file-like | TriggerDef
        If not ``None``, a TriggerDef instance is used to log events with a
        descriptive string instead of their ID. If not ``None``, either a
        TriggerDef instance or the path to a valid ``.ini`` file passed to
        TriggerDef.
    chunk_size : int
        Number of samples to send at once (usually ``16-32`` is good enough).
    high_resolution : bool
        If ``True``, it uses `~time.perf_counter` instead of `~time.sleep`
        for higher time resolution. However, it uses more CPU.
    """

    def __init__(
        self,
        stream_name,
        fif_file,
        repeat=float("inf"),
        trigger_def=None,
        chunk_size=16,
        high_resolution=False,
    ):
        _check_type(stream_name, (str,), item_name="stream_name")
        self._stream_name = stream_name
        self._fif_file = StreamPlayer._check_fif_file(fif_file)
        self._repeat = StreamPlayer._check_repeat(repeat)
        self._trigger_def = StreamPlayer._check_trigger_def(trigger_def)
        self._chunk_size = StreamPlayer._check_chunk_size(chunk_size)
        _check_type(high_resolution, (bool,), item_name="high_resolution")
        self._high_resolution = high_resolution

        self._process = None
        self._state = mp.Value("i", 0)

    def start(self, blocking=True):
        """Start streaming data on LSL network in a new process.

        Parameters
        ----------
        blocking : bool
            If ``True``, waits for the child process to start streaming data.
        """
        _check_type(blocking, (bool,), item_name="blocking")
        raw = mne.io.read_raw_fif(self._fif_file, preload=True, verbose=False)

        logger.info("Streaming started.")
        self._process = mp.Process(
            target=self._stream,
            args=(
                self._stream_name,
                raw,
                self._repeat,
                self._trigger_def,
                self._chunk_size,
                self._high_resolution,
                self._state,
            ),
        )
        self._process.start()

        if blocking:
            while self._state.value == 0:
                pass

        logger.info(self.__repr__())

    def stop(self):
        """Stop the streaming by terminating the process."""
        if self._process is None:
            logger.warning("StreamPlayer was not started. Skipping.")
            return

        with self._state.get_lock():
            self._state.value = 0

        logger.info(
            "Waiting for StreamPlayer %s process to finish.", self._stream_name
        )
        self._process.join(10)
        if self._process.is_alive():
            logger.error("StreamPlayer process not finishing..")
            self._process.kill()
            raise RuntimeError
        logger.info("Streaming finished.")

        self._process = None

    def _stream(
        self,
        stream_name,
        raw,
        repeat,
        trigger_def,
        chunk_size,
        high_resolution,
        state,
    ):  # noqa
        """Function called in the new process.

        Instantiate a _Streamer and start streaming.
        """
        streamer = _Streamer(
            stream_name,
            raw,
            repeat,
            trigger_def,
            chunk_size,
            high_resolution,
            state,
        )
        streamer.stream()

    # --------------------------------------------------------------------
    def __enter__(self):
        """Context manager entry point."""
        self.start(blocking=True)

    def __exit__(self, exc_type, exc_value, exc_tracebac):
        """Context manager exit point."""
        self.stop()

    def __repr__(self):
        """Representation of the instance."""
        status = "ON" if self._state.value == 1 else "OFF"
        return f"<Player: {self._stream_name} | {status} | {self._fif_file}>"

    # --------------------------------------------------------------------
    @staticmethod
    def _check_fif_file(fif_file):
        """Check if the provided fif_file is valid."""
        _check_type(fif_file, ("path-like",), item_name="fif_file")
        mne.io.read_raw_fif(fif_file, preload=False, verbose=None)
        return Path(fif_file)

    @staticmethod
    def _check_repeat(repeat):
        """Check that repeat is valid."""
        if repeat == float("inf"):
            return repeat
        _check_type(repeat, ("int",), item_name="repeat")
        if repeat <= 0:
            raise ValueError(
                "Argument repeat must be a strictly positive "
                "integer. Provided: %i" % repeat
            )
        return repeat

    @staticmethod
    def _check_trigger_def(trigger_def):
        """Check that trigger_def is valid."""
        _check_type(
            trigger_def,
            (None, TriggerDef, "path-like"),
            item_name="trigger_def",
        )
        if isinstance(trigger_def, (type(None), TriggerDef)):
            return trigger_def
        else:
            trigger_def = Path(trigger_def)
            if not trigger_def.exists():
                raise ValueError(
                    "Argument trigger_def is a path that does "
                    "not exist. Provided: %s" % trigger_def
                )
            trigger_def = TriggerDef(trigger_def)
            return trigger_def

    @staticmethod
    def _check_chunk_size(chunk_size):
        """Check that chunk_size is a strictly positive integer."""
        _check_type(chunk_size, ("int",), item_name="chunk_size")
        if chunk_size <= 0:
            raise ValueError(
                "Argument chunk_size must be a strictly positive "
                "integer. Provided: %i" % chunk_size
            )
        if chunk_size not in (16, 32):
            logger.warning(
                "The chunk size %i is different from the usual "
                "values 16 or 32.",
                chunk_size,
            )
        return chunk_size

    # --------------------------------------------------------------------
    @property
    def stream_name(self):
        """Stream's server name, displayed on LSL network.

        :type: str
        """
        return self._stream_name

    @property
    def fif_file(self):
        """Path to the compatible raw ``.fif`` file to play.

        :type: str | Path
        """
        return self._fif_file

    @property
    def repeat(self):
        """
        Number of times the stream player will loop on the FIF file before
        interrupting. Default ``float('inf')`` can be passed to never interrupt
        streaming.

        :type: int | ``float('ìnf')``
        """
        return self._repeat

    @property
    def trigger_def(self):
        """
        Either ``None`` or TriggerDef instance converting event numbers into
        event strings.

        :type: TriggerDef
        """
        return self._trigger_def

    @property
    def chunk_size(self):
        """
        Number of samples to send at once (usually ``16-32`` is good enough)
        ``[samples]``.

        :type: int
        """
        return self._chunk_size

    @property
    def high_resolution(self):
        """
        If ``True``, it uses `~time.perf_counter` instead of `~time.sleep`
        for higher time resolution. However, it uses more CPU.

        :type: bool
        """
        return self._high_resolution

    @property
    def process(self):
        """Launched streaming process.

        :type: Process
        """
        return self._process

    @property
    def state(self):
        """
        Streaming state of the player:
            - ``0``: Not streaming.
            - ``1``: Streaming.

        :type: `multiprocessing.Value`
        """
        return self._state


class _Streamer:
    """Class for playing a recorded file on LSL network."""

    def __init__(
        self,
        stream_name,
        raw,
        repeat,
        trigger_def,
        chunk_size,
        high_resolution,
        state,
    ):
        self._stream_name = stream_name
        self._raw = raw
        self._repeat = repeat
        self._trigger_def = trigger_def
        self._chunk_size = chunk_size
        self._high_resolution = high_resolution
        self._state = state

        self._sinfo = _Streamer._create_lsl_info(
            stream_name=self._stream_name,
            channel_count=len(self._raw.ch_names),
            nominal_srate=self._raw.info["sfreq"],
            ch_names=self._raw.ch_names,
        )
        self._tch = find_event_channel(inst=self._raw)
        # TODO: patch to be improved for multi-trig channel recording
        if isinstance(self._tch, list):
            self._tch = self._tch[0]
        self._scale_raw_data()
        self._outlet = StreamOutlet(self._sinfo, chunk_size=self._chunk_size)

    def _scale_raw_data(self):
        """Assumes raw data is in Volt and convert to microvolts.

        # TODO: Base the scaling on the units in the raw info
        """
        idx = np.arange(self._raw._data.shape[0]) != self._tch
        self._raw._data[idx, :] = self._raw.get_data()[idx, :] * 1e6

    def stream(self):
        """Stream data on LSL network."""
        idx_chunk = 0
        t_chunk = self._chunk_size / self._raw.info["sfreq"]
        finished = False

        if self._high_resolution:
            t_start = time.perf_counter()
        else:
            t_start = time.time()

        played = 0

        with self._state.get_lock():
            self._state.value = 1

        # Streaming loop
        while self._state.value == 1:

            idx_current = idx_chunk * self._chunk_size
            idx_next = idx_current + self._chunk_size
            chunk = self._raw._data[:, idx_current:idx_next]
            data = chunk.transpose()

            if idx_current >= self._raw._data.shape[1] - self._chunk_size:
                finished = True

            _Streamer._sleep(
                self._high_resolution, idx_chunk, t_start, t_chunk
            )

            self._outlet.push_chunk(data)
            logger.debug(
                "[%8.3fs] sent %d samples (LSL %8.3f)",
                time.perf_counter(),
                len(data),
                local_clock(),
            )

            self._log_event(chunk)
            idx_chunk += 1

            if finished:
                idx_chunk = 0
                finished = False
                if self._high_resolution:
                    t_start = time.perf_counter()
                else:
                    t_start = time.time()
                played += 1

                if played < self._repeat:
                    logger.info("Reached the end of data. Restarting.")
                else:
                    logger.info("Reached the end of data. Stopping.")
                    with self._state.get_lock():
                        self._state.value = 0

    def _log_event(self, chunk):
        """Look for an event on the data chunk and log it."""
        if self._tch is not None:
            event_values = set(chunk[self._tch]) - set([0])

            if len(event_values) > 0:
                if self._trigger_def is None:
                    logger.info("Events: %s", event_values)
                else:
                    for event_value in event_values:
                        if event_value in self._trigger_def.by_value:
                            logger.info(
                                "Events: %s (%s)",
                                event_value,
                                self._trigger_def.by_value[event_value],
                            )
                        else:
                            logger.info(
                                "Events: %s (Undefined event)", event_value
                            )

    # --------------------------------------------------------------------
    @staticmethod
    def _create_lsl_info(
        stream_name, channel_count, nominal_srate, ch_names
    ):  # noqa
        """
        Extract information from raw and set the LSL server's information
        needed to create the LSL stream.
        """
        sinfo = StreamInfo(
            stream_name,
            n_channels=channel_count,
            dtype="float32",
            sfreq=nominal_srate,
            stype="EEG",
            source_id=stream_name,
        )

        desc = sinfo.desc
        channel_desc = desc.append_child("channels")
        for channel in ch_names:
            channel_desc.append_child("channel").append_child_value(
                "label", str(channel)
            ).append_child_value("type", "EEG").append_child_value(
                "unit", "microvolts"
            )

        desc.append_child("amplifier").append_child(
            "settings"
        ).append_child_value("is_slave", "false")

        desc.append_child("acquisition").append_child_value(
            "manufacturer", "BSL"
        ).append_child_value("serial_number", "N/A")

        return sinfo

    @staticmethod
    def _sleep(high_resolution, idx_chunk, t_start, t_chunk):
        """Determine the time to sleep."""
        # if a resolution over 2 KHz is needed.
        if high_resolution:
            t_sleep_until = t_start + idx_chunk * t_chunk
            while time.perf_counter() < t_sleep_until:
                pass
        # time.sleep() can have 500 us resolution.
        else:
            t_wait = t_start + idx_chunk * t_chunk - time.time()
            if t_wait > 0.001:
                time.sleep(t_wait)
