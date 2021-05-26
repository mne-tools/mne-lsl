#!/usr/bin/env python3

"""
Strean a recorded fif file on LSL network.

Command-line arguments:
    #1: Server name
    #2: Raw FIF file to stream
    #3: Chunk size
    #4: Trigger file
"""

import time
import pylsl
import numpy as np
import multiprocessing as mp

from .. import logger
from ..triggers import TriggerDef
from ..utils.io import read_raw_fif
from ..utils.events import find_event_channel


class StreamPlayer:
    """
    Class for playing a recorded file on LSL network in another process.

    Parameters
    ----------
    server_name : str
        The stream's name, displayed on LSL network.
    fif_file : str
        The absolute path to the .fif file to play.
    chunk_size : int
        The number of samples to send at once (usually 16-32 is good enough).
    trigger_file : str
        The absolute path to the file containing the table converting event
        numbers into event strings.

    Notes
    -----
    It instances a Streamer in a new process and call Streamer.stream().
    """

    def __init__(self, server_name, fif_file, chunk_size,
                 trigger_file=None, logger=logger):

        self._server_name = server_name
        self._fif_file = fif_file
        self._chunk_size = chunk_size
        self._trigger_file = trigger_file

        self._logger = logger
        self._process = None
        self._state = mp.Value('i', 0)

    def start(self, repeat=np.float('inf'), high_resolution=False):
        """
        Start streaming data on LSL network in a new process by calling
        stream().

        Parameters
        ----------
        repeat : float
            The number of times to replay the data.
        high_resolution : bool
            If True, it uses perf_counter() instead of sleep() for higher time
            resolution. However, it uses more CPU.
        """
        self._process = mp.Process(target=self._stream,
                                   args=(repeat,
                                         high_resolution,
                                         self._logger,
                                         self._state))
        self._process.start()

        while self._state.value == 0:
            # time.sleep(0.1)
            pass

    def wait(self, timeout=None):
        """
        Wait that the data streaming finishes.

        Attributes
        ----------
        timeout : float
            Block until timeout is reached.
            If None, block until streaming is finished.
        """
        self._process.join(timeout)

    def stop(self):
        """
        Stop the streaming, by terminating the process.
        """
        if self._process:
            self._logger.info(f"Stop streaming data from: '{server_name}'.")
            self._process.terminate()

    def _stream(self, repeat, high_resolution, logger, state):
        """
        The function called in the new process.

        Instance a Streamer and start streaming.
        """
        s = Streamer(self.server_name, self.fif_file, self.chunk_size,
                     self.trigger_file, self._logger, state)
        s.stream(repeat, high_resolution)

    @property
    def server_name(self):
        """
        The stream's name, displayed on LSL network.

        Returns
        -------
        str
        """
        return self._server_name

    @property
    def fif_file(self):
        """
        The absolute path to the .fif file to play.

        Returns
        -------
        str
        """
        return self._fif_file

    @property
    def chunk_size(self):
        """
        The size of a chunk of data.

        Returns
        -------
        int
        """
        return self._chunk_size

    @property
    def trigger_file(self):
        """
        The absolute path to the file containing the table converting event
        numbers into event strings.

        Returns
        -------
        int
        """
        return self._trigger_file

    @property
    def process(self):
        """
        The launched process

        Returns
        -------
        multiprocessing.Process
        """
        return self._process


class Streamer:
    """
    Class for playing a recorded file on LSL network.

    Parameters
    ----------
    server_name : str
        The stream's name, displayed on LSL network.
    fif_file : str
        The absolute path to the .fif file to play.
    chunk_size : int
        The number of samples to send at once (usually 16-32 is good enough).
    trigger_file : str
        The absolute path to the file containing the table converting event
        numbers into event strings.
    state : mp.Value
        The mp sharing variable (used to wait that the streaming is launched).

    Notes
    -----
    Run neurodecode.set_log_level('DEBUG') to print out the relative
    timestamps since started.
    """

    def __init__(self, server_name, fif_file, chunk_size, trigger_file=None,
                 logger=logger, state=mp.Value('i', 0)):

        self._raw = None
        self._events = None
        self._server_name = server_name
        self._chunk_size = chunk_size

        self._thread = None
        self._tdef = None
        self._logger = logger
        self._state = state

        if trigger_file is not None:
            self._tdef = TriggerDef(trigger_file)

        self.load_data(fif_file)
        sinfo = self.set_lsl_info(server_name)
        self._outlet = pylsl.StreamOutlet(sinfo, chunk_size=chunk_size)
        self.get_info()

    def stream(self, repeat=np.float('inf'), high_resolution=False):
        """
        Stream data on LSL network.

        Parameters
        ----------
        repeat : int
            The number of times to replay the data (Default=inf).
        high_resolution : bool
            If True, it uses perf_counter() instead of sleep() for higher time
            resolution. However, it uses much more CPU.
        """
        self._logger.info('Streaming started.')

        # Change sharing variable to 1 to let other process know that it is now streaming
        with self._state.get_lock():
            self._state.value = 1

        idx_chunk = 0
        t_chunk = self.chunk_size / self.get_sample_rate()
        finished = False

        if high_resolution:
            t_start = time.perf_counter()
        else:
            t_start = time.time()

        # start streaming
        played = 0
        while played <= repeat:

            idx_current = idx_chunk * self.chunk_size
            chunk = self.raw._data[:,
                                   idx_current:idx_current + self.chunk_size]
            data = chunk.transpose().tolist()

            if idx_current >= self.raw._data.shape[1] - self.chunk_size:
                finished = True

            self._sleep(high_resolution, idx_chunk, t_start, t_chunk)

            self._outlet.push_chunk(data)
            self._logger.debug(
                '[%8.3fs] sent %d samples (LSL %8.3f)'
                % (time.perf_counter(), len(data), pylsl.local_clock()))

            self._log_event(chunk)
            idx_chunk += 1

            if finished:
                self._logger.info('Reached the end of data. Restarting.')
                idx_chunk = 0
                finished = False
                if high_resolution:
                    t_start = time.perf_counter()
                else:
                    t_start = time.time()
                played += 1

    def set_lsl_info(self, server_name):
        """
        Set the lsl server's infos needed to create the LSL stream.

        Parameters
        ----------
        server_name : str
            The stream's name, displayed on LSL network.

        Returns
        -------
        pylsl.StreamInfo
            The info to create the stream on LSL network.
        """
        sinfo = pylsl.StreamInfo(server_name, channel_count=self.get_nb_ch(),
                                 channel_format='float32',
                                 nominal_srate=self.get_sample_rate(),
                                 type='EEG', source_id=server_name)

        desc = sinfo.desc()
        channel_desc = desc.append_child("channels")
        for ch in self.raw.ch_names:
            channel_desc.append_child('channel')\
                        .append_child_value('label', str(ch))\
                        .append_child_value('type', 'EEG')\
                        .append_child_value('unit', 'microvolts')

        desc.append_child('amplifier')\
            .append_child('settings')\
            .append_child_value('is_slave', 'false')

        desc.append_child('acquisition')\
            .append_child_value('manufacturer', 'NeuroDecode')\
            .append_child_value('serial_number', 'N/A')

        return sinfo

    def load_data(self, fif_file):
        """
        Load the data to play from a fif file.
        Multiplies all channel except trigger by 1e6 to convert to uV.

        Parameters
        ----------
        fif_file : str
            The absolute path to the .fif file to play.
        """
        self._raw, self._events = read_raw_fif(fif_file)

        tch = self.get_trg_index()
        idx = [k for k in range(self.raw._data.shape[0]) if k != tch]
        self.raw._data[idx, :] = self._raw.get_data()[idx, :] * 1E6

        if self.raw is not None:
            self._logger.info_green(f'Successfully loaded {fif_file}')
        else:
            self._logger.error(f"Error while loading '{fif_file}'.")
            raise IOError

    def get_sample_rate(self):
        """
        Get the sample rate

        Returns
        -------
        float
            The sampling rate [Hz]
        """
        return self.raw.info['sfreq']

    def get_nb_ch(self):
        """
        Get the number of channels.

        Returns
        -------
        int
            The number of channels.
        """
        return len(self.raw.ch_names)

    def get_trg_index(self):
        """
        Return the index of the trigger channel.

        Returns
        -------
        int
            The trigger channel's index.
        """
        return find_event_channel(inst=self.raw)

    def get_info(self):
        """
        Log the info about the created LSL stream.
        """
        self._logger.info(
            f'Server name: {self.server_name}')
        self._logger.info(
            f'Sampling frequency {self.get_sample_rate() = :.3f} Hz')
        self._logger.info(
            f'Number of channels : {self.get_nb_ch()}')
        self._logger.info(
            f'Chunk size : {self.chunk_size}')
        for i, ch in enumerate(self.raw.ch_names):
            self._logger.info(f'{i} {ch}')
        self._logger.info(
            f'Trigger channel : {self.get_trg_index()}')

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
            # time.sleep() can have 500 us resolution using the tweak tool provided.
            t_wait = t_start + idx_chunk * t_chunk - time.time()
            if t_wait > 0.001:
                time.sleep(t_wait)

    def _log_event(self, chunk):
        """
        Look for an event on the data chunk and log it.
        """
        event_ch = self.get_trg_index()

        if event_ch is not None:
            event_values = set(chunk[event_ch]) - set([0])

            if len(event_values) > 0:
                if self._tdef is None:
                    self._logger.info(f'Events: {event_values}')
                else:
                    for event in event_values:
                        if event in self._tdef.by_value:
                            self._logger.info(
                                f'Events: {event} ({self._tdef.by_value[event]})')
                        else:
                            self._logger.info(
                                f'Events: {event} (Undefined event {event})')

    @property
    def raw(self):
        """
        The raw data to stream on LSL network.

        Returns
        -------
        mne.io.RawArray
        """
        return self._raw

    @property
    def events(self):
        """
        The mne-compatible events (N x [frame, 0, type]).

        Returns
        -------
        np.array
        """
        return self._events

    @property
    def server_name(self):
        """
        The stream's name, displayed on LSL network.

        Returns
        -------
        str
        """
        return self._server_name

    @property
    def chunk_size(self):
        """
        The size of a chunk of data.

        Returns
        -------
        int
        """
        return self._chunk_size


if __name__ == '__main__':

    import sys
    from pathlib import Path

    chunk_size = 16
    trigger_file = None
    fif_file = None
    server_name = None

    if len(sys.argv) > 5:
        raise RuntimeError("Too many arguments provided, maximum is 4.")

    if len(sys.argv) > 4:
        trigger_file = sys.argv[4]

    if len(sys.argv) > 3:
        chunk_size = int(sys.argv[3])

    if len(sys.argv) > 2:
        fif_file = sys.argv[2]

    if len(sys.argv) > 1:
        server_name = sys.argv[1]
        if not fif_file:
            fif_file = str(
                Path(input(">> Provide the path to the .fif file to play: \n>> ")))

    if len(sys.argv) == 1:
        server_name = input(
            ">> Provide the server name displayed on LSL network: \n>> ")
        fif_file = str(
            Path(input(">> Provide the path to the .fif file to play: \n>> ")))

    sp = StreamPlayer(server_name, fif_file, chunk_size, trigger_file)
    sp.start()
    input(">> Press ENTER to stop replaying data \n")
    sp.stop()
