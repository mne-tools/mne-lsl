import time
from pathlib import Path
import multiprocessing as mp

import mne
import pylsl
import numpy as np

from .. import logger
from ..triggers import TriggerDef
from ..utils import find_event_channel
from ..utils._docs import fill_doc


@fill_doc
class StreamPlayer:
    """
    Class for playing a recorded file on LSL network in another process.

    Parameters
    ----------
    %(player_stream_name)s
    %(player_fif_file)s
    %(player_repeat)s
    %(trigger_file)s  # TODO change doc.
    %(player_chunk_size)s
    %(player_high_resolution)s
    """

    def __init__(self, stream_name, fif_file, repeat=float('inf'),
                 trigger_def=None, chunk_size=16, high_resolution=False):
        self._stream_name = str(stream_name)
        self._fif_file = StreamPlayer._check_fif_file(fif_file)
        self._repeat = StreamPlayer._check_repeat(repeat)
        self._trigger_def = StreamPlayer._check_trigger_def(trigger_def)
        self._chunk_size = StreamPlayer._check_chunk_size(chunk_size)
        self._high_resolution = bool(high_resolution)

        self._process = None
        self._state = mp.Value('i', 0)

    @fill_doc
    def start(self):
        """
        Start streaming data on LSL network in a new process.
        """
        logger.info('Streaming started.')
        self._process = mp.Process(target=self._stream,
                                   args=(self._repeat, self._high_resolution))
        self._process.start()

    def stop(self):
        """
        Stop the streaming, by terminating the process.
        """
        if self._process is not None and self._process.is_alive():
            logger.info(
                f"Stop streaming data from: '{self._stream_name}'.")
            self._process.kill()
            self._process = None

    def _stream(self, repeat, high_resolution):
        """
        The function called in the new process.
        Instance a _Streamer and start streaming.
        """
        streamer = _Streamer(
            self._stream_name, self._fif_file,
            self._chunk_size, self._trigger_def)
        streamer.stream(repeat, high_resolution)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_fif_file(fif_file):
        """
        Check if the provided fif_file is valid.
        """
        try:
            mne.io.read_raw_fif(fif_file, preload=False)
            return fif_file
        except Exception:
            raise ValueError(
                'Argument fif_file must be a path to a valid MNE raw file. '
                f'Provided: {fif_file}.')

    @staticmethod
    def _check_repeat(repeat):
        """
        Checks that repeat is either infinity or a strictly positive integer.
        """
        if repeat == float('inf'):
            return repeat
        elif isinstance(repeat, (int, float)):
            repeat = int(repeat)
            if 0 < repeat:
                return repeat
            else:
                logger.error(
                    'Argument repeat must be a strictly positive integer. '
                    f'Provided: {repeat} -> Changing to +inf.')
                return float('inf')

    @staticmethod
    def _check_trigger_def(trigger_def):
        """
        Checks that the trigger file is either a path to a valid trigger
        definition file, in which case it is loader and pass as a TriggerDef,
        or a TriggerDef instance. Else sets it as None.
        """
        if isinstance(trigger_def, TriggerDef):
            return trigger_def
        elif isinstance(trigger_def, (str, Path)):
            trigger_def = Path(trigger_def)
            if not trigger_def.exists():
                logger.error(
                    'Argument trigger_def is a path that does not exist. '
                    f'Provided: {trigger_def} -> Ignoring.')
                return None
            trigger_def = TriggerDef(trigger_def)
            return trigger_def
        else:
            logger.error(
                'Argument trigger_def was not a TriggerDef instance or a path '
                'to a trigger definition ini file. '
                f'Provided: {type(trigger_def)} -> Ignoring.')
            return None

    @staticmethod
    def _check_chunk_size(chunk_size):
        """
        Checks that chunk_size is a strictly positive integer.
        """
        invalid = False
        try:
            chunk_size = int(chunk_size)
            if chunk_size <= 0:
                invalid = True
        except:
            invalid = True

        if not invalid and chunk_size not in (16, 32):
            logger.warning(
                f'The chunk size {chunk_size} is different from the usual '
                'values 16 or 32.')
        if invalid:
            logger.error(
                'Argument chunk_size must be a strictly positive integer. '
                f'Provided: {chunk_size} -> Changing to 16.')
            return 16
        else:
            return chunk_size

    # --------------------------------------------------------------------
    @property
    def stream_name(self):
        """
        Stream's server name, displayed on LSL network.

        :type: `str`
        """
        return self._stream_name

    @property
    def fif_file(self):
        """
        Path to the ``.fif`` file to play.

        :type: `str` | `~pathlib.Path`
        """
        return self._fif_file

    @property
    def repeat(self):
        """
        Number of times the stream player will loop on the FIF file before
        interrupting. Default ``float('inf')`` can be passed to never interrupt
        streaming.

        :type: `int` | ``float('ìnf')``
        """
        return self._repeat

    @property
    def trigger_file(self):
        """
        Path to the file containing the table converting event numbers into
        event strings.

        :type: `str` | `~pathlib.Path`
        """
        return self._trigger_file

    @property
    def chunk_size(self):
        """
        Size of a chunk of data ``[samples]``.

        :type: `int`
        """
        return self._chunk_size

    @property
    def high_resolution(self):
        """
        If True, use an high resolution counter instead of a sleep.

        :type: `bool`
        """
        return self._high_resolution

    @property
    def process(self):
        """
        Launched process.

        :type: `multiprocessing.Process`
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


@fill_doc
class _Streamer:
    """
    Class for playing a recorded file on LSL network.

    Parameters
    ----------
    %(player_stream_name)s
    %(player_fif_file)s
    %(player_chunk_size)s
    %(trigger_file)s
    """

    def __init__(self, stream_name, fif_file, chunk_size, trigger_file=None):
        self._stream_name = str(stream_name)
        self._chunk_size = StreamPlayer._check_chunk_size(chunk_size)

        if trigger_file is None:
            self._tdef = None
        else:
            try:
                self._tdef = TriggerDef(trigger_file)
            except Exception:
                self._tdef = None

        self._load_data(fif_file)

    def _load_data(self, fif_file):
        """
        Load the data to play from a .fif file.
        Multiplies all channel except trigger by 1e6 to convert to uV.

        Parameters
        ----------
        %(player_fif_file)s
        """
        self._raw = mne.io.read_raw_fif(fif_file, preload=True)
        self._tch = find_event_channel(inst=self._raw)
        self._sample_rate = self._raw.info['sfreq']
        self._ch_count = len(self._raw.ch_names)
        idx = np.arange(self._raw._data.shape[0]) != self._tch
        self._raw._data[idx, :] = self._raw.get_data()[idx, :] * 1E6
        # TODO: Base the scaling on the units in the raw info

        self._set_lsl_info(self._stream_name)
        self._outlet = pylsl.StreamOutlet(
            self._sinfo, chunk_size=self._chunk_size)
        self._show_info()

    def _set_lsl_info(self, stream_name):
        """
        Set the LSL server's infos needed to create the LSL stream.

        Parameters
        ----------
        %(player_stream_name)s
        """
        sinfo = pylsl.StreamInfo(
            stream_name, channel_count=self._ch_count,
            channel_format='float32', nominal_srate=self._sample_rate,
            type='EEG', source_id=stream_name)

        desc = sinfo.desc()
        channel_desc = desc.append_child("channels")
        for channel in self._raw.ch_names:
            channel_desc.append_child('channel')\
                        .append_child_value('label', str(channel))\
                        .append_child_value('type', 'EEG')\
                        .append_child_value('unit', 'microvolts')

        desc.append_child('amplifier')\
            .append_child('settings')\
            .append_child_value('is_slave', 'false')

        desc.append_child('acquisition')\
            .append_child_value('manufacturer', 'BSL')\
            .append_child_value('serial_number', 'N/A')

        self._sinfo = sinfo

    def _show_info(self):
        """
        Display the informations about the created LSL stream.
        """
        logger.info(f'Stream name: {self._stream_name}')
        logger.info(f'Sampling frequency {self._sample_rate:.3f} Hz')
        logger.info(f'Number of channels : {self._ch_count}')
        logger.info(f'Chunk size : {self._chunk_size}')
        for i, channel in enumerate(self._raw.ch_names):
            logger.info(f'{i} {channel}')
        logger.info(f'Trigger channel : {self._tch}')

    def stream(self, repeat=float('inf'), high_resolution=False):
        """
        Stream data on LSL network.

        Parameters
        ----------
        %(player_repeat)s
        %(player_high_resolution)s
        """
        idx_chunk = 0
        t_chunk = self._chunk_size / self._sample_rate
        finished = False

        if high_resolution:
            t_start = time.perf_counter()
        else:
            t_start = time.time()

        # start streaming
        played = 0
        while True:

            idx_current = idx_chunk * self._chunk_size
            idx_next = idx_current + self._chunk_size
            chunk = self._raw._data[:, idx_current:idx_next]
            data = chunk.transpose().tolist()

            if idx_current >= self._raw._data.shape[1] - self._chunk_size:
                finished = True

            self._sleep(high_resolution, idx_chunk, t_start, t_chunk)

            self._outlet.push_chunk(data)
            logger.debug(
                '[%8.3fs] sent %d samples (LSL %8.3f)'
                % (time.perf_counter(), len(data), pylsl.local_clock()))

            self._log_event(chunk)
            idx_chunk += 1

            if finished:
                idx_chunk = 0
                finished = False
                if high_resolution:
                    t_start = time.perf_counter()
                else:
                    t_start = time.time()
                played += 1

                if played < repeat:
                    logger.info('Reached the end of data. Restarting.')
                else:
                    logger.info('Reached the end of data. Stopping.')
                    break

    def _sleep(self, high_resolution, idx_chunk, t_start, t_chunk):
        """
        Determine the time to sleep.
        """
        if high_resolution:
            # if a resolution over 2 KHz is needed
            t_sleep_until = t_start + idx_chunk * t_chunk
            while time.perf_counter() < t_sleep_until:
                pass
        else:
            # time.sleep() can have 500 us resolution.
            t_wait = t_start + idx_chunk * t_chunk - time.time()
            if t_wait > 0.001:
                time.sleep(t_wait)

    def _log_event(self, chunk):
        """
        Look for an event on the data chunk and log it.
        """
        if self._tch is not None:
            event_values = set(chunk[self._tch]) - set([0])

            if len(event_values) > 0:
                if self._tdef is None:
                    logger.info(f'Events: {event_values}')
                else:
                    for event in event_values:
                        if event in self._tdef.by_value:
                            logger.info(
                                f'Events: {event} '
                                f'({self._tdef.by_value[event]})')
                        else:
                            logger.info(
                                f'Events: {event} (Undefined event {event})')
