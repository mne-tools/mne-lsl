import time
import pickle
from pathlib import Path
import multiprocessing as mp

from .. import logger
from ..utils.io import pcl2fif, make_dirs
from ..stream_receiver import StreamReceiver, StreamEEG
from ..stream_receiver._stream import MAX_BUF_SIZE


class StreamRecorder:
    def __init__(self, record_dir=None, fname=None, stream_name=None):
        self._record_dir = StreamRecorder._check_record_dir(record_dir)
        self._fname = StreamRecorder._check_fname(fname)
        self._stream_name = stream_name

        self._eve_file = None # for SOFTWARE triggers
        self._process = None
        self._state = mp.Value('i', 0)

    def start(self, verbose=False):
        self._fname, self._eve_file = StreamRecorder._create_fname(
            self._record_dir, self._fname)

        self._process = mp.Process(
            target=self._record,
            args=(self._record_dir, self._fname, self._eve_file,
                  self._stream_name, self._state, verbose))
        self._process.start()

    def stop(self):
        with self._state.get_lock():
            self._state.value = 0

        logger.info('Waiting for recorder process to finish.')
        self._process.join(10)
        if self._process.is_alive():
            logger.error(
                'Recorder process not finishing..')
            raise RuntimeError
        logger.info('Recording finished.')

        self._eve_file = None
        self._process = None

    def _record(self, record_dir, fname, eve_file,
                stream_name, state, verbose):
        recorder = _Recorder(
            record_dir, fname, eve_file, stream_name, state, verbose)
        recorder.record()

    # --------------------------------------------------------------------
    @staticmethod
    def _check_record_dir(record_dir):
        if record_dir is None:
            record_dir = Path.cwd()
        else:
            record_dir = Path(record_dir)
        return record_dir

    @staticmethod
    def _check_fname(fname):
        if fname is not None:
            fname = str(fname)
        return fname

    @staticmethod
    def _create_fname(record_dir, fname):
        fname = fname if fname is not None \
            else time.strftime('%Y%m%d-%H%M%S', time.localtime())

        eve_file = record_dir / f'{fname}-eve.txt'

        return fname, eve_file

    # --------------------------------------------------------------------
    @property
    def record_dir(self):
        return self._record_dir

    @record_dir.setter
    def record_dir(self, record_dir):
        if self._state == 1:
            logger.warning(
                'The recording directory cannot be changed during an '
                'ongoing recording.')
        else:
            self._record_dir = StreamRecorder._check_record_dir(record_dir)

    @property
    def fname(self):
        return self._fname

    @fname.setter
    def fname(self, fname):
        if self._state == 1:
            logger.warning(
                'The file name cannot be changed during an '
                'ongoing recording.')
        else:
            self._fname = StreamRecorder._check_fname(fname)

    @property
    def stream_name(self):
        return self._stream_name

    @stream_name.setter
    def stream_name(self, stream_name):
        if self._state == 1:
            logger.warning(
                'The stream name(s) to connect to cannot be changed during an '
                'ongoing recording.')
        else:
            self._stream_name = stream_name

    @property
    def eve_file(self):
        return self._eve_file

    @eve_file.setter
    def eve_file(self, eve_file):
        logger.warning("The event file cannot be changed directly.")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        logger.warning("The state cannot be changed directly.")

    @property
    def process(self):
        """
        The launched process.
        """
        return self._process

    @process.setter
    def process(self, process):
        logger.warning("The recorder process cannot be changed directly.")


class _Recorder:
    def __init__(self, record_dir, fname, eve_file,
                 stream_name, state, verbose):
        self._record_dir = record_dir
        self._fname = fname
        self._eve_file = eve_file
        self._stream_name = stream_name
        self._state = state
        self._verbose = verbose

    def record(self):
        sr = StreamReceiver(
            bufsize=MAX_BUF_SIZE, stream_name=self._stream_name)
        pcl_files = self._create_files(sr)
        self._check_writability(sr, pcl_files)

        with self._state.get_lock():
            self._state.value = 1

        # Acquisition loop
        while self._state.value == 1:
            sr.acquire()

            if self._verbose:
                pass # TODO: Add timing display

        self._save(sr, pcl_files)

    def _create_files(self, sr):
        pcl_files = dict()
        for stream in sr.streams:
            pcl_files[stream] = \
                self._record_dir / f'{self._fname}-{stream}-raw.pcl'

        return pcl_files

    def _check_writability(self, sr, pcl_files):
        make_dirs(self._record_dir)

        for stream in sr.streams:
            try:
                with open(pcl_files[stream], 'w') as file:
                    file.write(
                        'Data will be written when the recording is finished.')
            except Exception as error:
                logger.error(
                    f"Problem writing to '{pcl_files[stream]}'. "
                    "Check permissions.")
                raise error

        logger.info(
            'Record to files:\n' # TODO: This line is not logged?
            '\n'.join(str(file) for file in pcl_files.values()))

    def _save(self, sr, pcl_files):
        logger.info('Saving raw data ...')
        for stream in sr.streams:
            signals, timestamps = sr.get_buffer(stream)

            if isinstance(sr.streams[stream], StreamEEG):
                signals[:, 1:] *= 1E-6

            data = {
                'signals': signals,
                'timestamps': timestamps,
                'events': None,
                'sample_rate': sr.streams[stream].sample_rate,
                'channels': len(sr.streams[stream].ch_list),
                'ch_names': sr.streams[stream].ch_list,
                'lsl_time_offset': sr.streams[stream].lsl_time_offset}

            with open(pcl_files[stream], 'wb') as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Saved to '{pcl_files[stream]}'")

            if not isinstance(sr.streams[stream], StreamEEG):
                continue
            logger.info('Converting raw files into fif.')

            if self._eve_file.exists():
                logger.info('Found matching event file, adding events.')
                pcl2fif(pcl_files[stream], external_event=self._eve_file)
            else:
                pcl2fif(pcl_files[stream], external_event=None)
