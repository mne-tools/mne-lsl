import os
import time
import pickle
import datetime
from pathlib import Path
import multiprocessing as mp

from ..utils.timer import Timer
from ..utils.lsl import start_server
from ..utils.io import pcl2fif, make_dirs
from ..stream_receiver import StreamReceiver, StreamEEG


class _Recorder:
    """
    Base class for recording signals coming from an LSL stream.
    """

    def __init__(self, record_dir, logger, state=mp.Value('i', 0)):
        self._MAX_BUFSIZE = 86400  # seconds - 24h

        if record_dir == '.':
            self.record_dir = Path.cwd()
        else:
            self.record_dir = Path(record_dir)

        self.sr = None
        self.state = state
        self.logger = logger

    def connect(self, stream_name=None, eeg_only=False):
        """
        Instance a StreamReceiver connecting to the appropriate lsl stream.

        Parameters
        ----------
        stream_name : str
            Connect to a server named 'stream_name'. None: no constraint.
        eeg_only : bool
            If true, ignore non-EEG servers.
        """
        self.sr = StreamReceiver(buffer_size=self._MAX_BUFSIZE,
                                 amp_name=stream_name,  # TODO change name to stream_name
                                 eeg_only=eeg_only)

    def create_files(self):
        """
        Create the files to save the data.

        Returns
        -------
        pcl_files : dict {str: pathlib.Path}
            The data file's name (pickle format) for each stream.
        eve_file : pathlib.Path
            The software events file (txt format).
        """
        pcl_files, eve_file = self.create_filenames()
        self.test_writability(pcl_files)

        self.logger.info('Record to files:')
        print(*pcl_files.values(), sep='\n')

        return pcl_files, eve_file

    def create_filenames(self):
        """
        Create the filenames to save the data and events.

        Returns
        -------
        pcl_files : dict {str: pathlib.Path}
            The data file's name (pickle format) for each stream.
        eve_file : pathlib.Path
            The software events' file (txt format).
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        eve_file = self.record_dir / f'{timestamp}-eve.txt'

        pcl_files = dict()

        for s in self.sr.streams:
            pcl_files[s] = self.record_dir / f'{timestamp}-{s}-raw.pcl'

        return pcl_files, eve_file

    def test_writability(self, pcl_files):
        """
        Test the file's writability.

        Parameters
        ----------
        pcl_files : dict {str: pathlib.Path}
            The data file's name (pickle format) for each stream.
        """
        make_dirs(self.record_dir)

        for pcl_file in pcl_files.keys():
            try:
                with open(pcl_files[pcl_file], 'w') as f:
                    f.write('Data will be written when the recording is finished.')
            except:
                self.logger.error(
                    f"Problem writing to '{pcl_file}'. Check permission.")
                raise RuntimeError

    def create_events_server(self, name='StreamRecorderInfo', eve_file=None):
        """
        Start a LSL server for sending out events when software trigger is used.

        Parameters
        ----------
        name : str
            The server's name displayed on the network.
        eve_file : str
            The software events' file (txt format).
        """
        return start_server(name, channel_format='string',
                            source_id=eve_file, stype='Markers')

    def get_buffer(self, stream_name=None):
        """
        Get all the data collected in the buffer for a stream.

        Parameters
        ----------
        stream_name : str
            The name of the stream to get the data.

        Returns
        -------
        data : np.ndarray
            The data [[samples_ch1],[samples_ch2]...]
        timestamps : np.ndarray
            The associated timestamps [samples]
        """
        data, timestamps = self.sr.get_buffer(stream_name)

        return data, timestamps

    def save_to_file(self, pcl_files, eve_file):
        """
        Save the acquired data to pcl and fif format.

        Parameters
        ----------
        pcl_files : dict {str: pathlib.Path}
            The data file's name (pickle format) for each stream.
        eve_file : pathlib.Path
            The software events' file (txt format).
        """
        self.logger.info('Saving raw data ...')

        for s in self.sr.streams:

            signals, timestamps = self.get_buffer(s)

            if isinstance(self.sr.streams[s], StreamEEG):
                signals[:, 1:] *= 1E-6

            data = self.create_dict_to_save(signals, timestamps, s)

            with open(pcl_files[s], 'wb') as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.info(f"Saved to '{pcl_files[s]}'\n")
            self.logger.info('Converting raw files into fif.')

            # Convert only EEG stream to fif format
            if not isinstance(self.sr.streams[s], StreamEEG):
                continue

            if eve_file.exists():
                self.logger.info('Found matching event file, adding events.')
                pcl2fif(pcl_files[s], external_event=eve_file)
            else:
                pcl2fif(pcl_files[s], external_event=None)

    def create_dict_to_save(self, signals, timestamps, stream_name):
        """
        Create a dictionnary containing the signals, its timestamps, the
        sampling rate, the channels' names and the lsl offset for each
        acquired streams.

        Parameters
        ----------
        signal : np.ndarray
            The data [[samples_ch1],[samples_ch2]...]
        timestamps : np.ndarray
            The associated timestamps [samples]
        stream_name : str
            The name of the stream streaming the signal.

        Returns
        -------
        data : dict
            keys:   'signals', 'timestamps', 'events', 'sample_rate',
                    'channels', 'ch_names', 'lsl_time_offset'
        """
        data = {'signals': signals,
                'timestamps': timestamps,
                'events': None,
                'sample_rate': self.sr.streams[stream_name].sample_rate,
                'channels': len(self.sr.streams[stream_name].ch_list),
                'ch_names': self.sr.streams[stream_name].ch_list,
                'lsl_time_offset': self.sr.streams[stream_name].lsl_time_offset}

        return data

    def record(self, verbose=False):
        """
        Start the recording and save to files the data in pickle and fif format.

        Parameters
        ----------
        verbose : bool
            If True, the recording will log 'RECORDING {time}' every second.
        """
        data_files, eve_file = self.create_files()

        self.outlet = self.create_events_server(eve_file=str(eve_file))
        self.logger.info(f'>> Recording started (PID {os.getpid()}).')
        self.acquire(verbose)

        self.logger.info('>> Stop requested. Copying buffer')
        self.save_to_file(data_files, eve_file)

    def acquire(self, verbose=False):
        """
        Acquire the data from the connected LSL stream.

        Parameters
        ----------
        verbose : bool
            If True, the recording will log 'RECORDING ...' every second.
        """
        with self.state.get_lock():
            self.state.value = 1

        tm = Timer(autoreset=True)

        #  Extract name of the first recorded stream
        first_stream = list(self.sr.streams.keys())[0]

        while self.state.value == 1:
            self.sr.acquire()

            if verbose:
                bufsec = len(self.sr.streams[first_stream].buffer.data) / \
                    self.sr.streams[first_stream].sample_rate

                duration = str(datetime.timedelta(seconds=int(bufsec)))
                self.logger.info(f'RECORDING {duration}')

            tm.sleep_atleast(1)
