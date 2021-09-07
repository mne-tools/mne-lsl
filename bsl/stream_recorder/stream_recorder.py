import os
import time
import pickle
import datetime
from pathlib import Path
import multiprocessing as mp

from .. import logger
from ..utils import Timer
from ..utils._docs import fill_doc
from ..utils.io import pcl2fif
from ..stream_receiver import StreamReceiver, StreamEEG
from ..stream_receiver._stream import MAX_BUF_SIZE


@fill_doc
class StreamRecorder:
    """
    Class for recording the signals coming from LSL streams.

    Parameters
    ----------
    %(recorder_record_dir)s
    %(recorder_fname)s
    %(stream_name)s
    """

    def __init__(self, record_dir=None, fname=None, stream_name=None):
        self._record_dir = StreamRecorder._check_record_dir(record_dir)
        self._fname = StreamRecorder._check_fname(fname)
        self._stream_name = stream_name

        self._eve_file = None  # for SOFTWARE triggers
        self._process = None
        self._state = mp.Value('i', 0)

    @fill_doc
    def start(self, fif_subdir=True, blocking=True, verbose=False):
        """
        Start the recording in a new process.

        Parameters
        ----------
        %(recorder_fif_subdir)s
        blocking : `bool`
            If ``True``, waits for the child process to start recording data.
        %(recorder_verbose)s
        """
        fname, self._eve_file = StreamRecorder._create_fname(
            self._record_dir, self._fname)
        logger.debug("File name stem is '%s'." % fname)
        logger.debug("Event file name is '%s'." % self._eve_file)

        self._process = mp.Process(
            target=self._record,
            args=(self._record_dir, fname, self._eve_file, bool(fif_subdir),
                  self._stream_name, self._state, bool(verbose)))
        self._process.start()

        if blocking:
            while self._state.value == 0:
                pass

    def stop(self):
        """
        Stops the recording.
        """
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

    def _record(self, record_dir, fname, eve_file, fif_subdir,
                stream_name, state, verbose):
        """
        The function called in the new process.
        Instance a _Recorder and start recording.
        """
        recorder = _Recorder(
            record_dir, fname, eve_file, fif_subdir,
            stream_name, state, verbose)
        recorder.record()

    # --------------------------------------------------------------------
    @staticmethod
    def _check_record_dir(record_dir):
        """
        Converts record_dir to a Path, or select the current working directory
        if record_dir is None.
        """
        if record_dir is None:
            record_dir = Path.cwd()
        else:
            record_dir = Path(record_dir)
        return record_dir

    @staticmethod
    def _check_fname(fname):
        """
        Checks that the file name stem is a string or None.
        """
        if fname is not None:
            fname = str(fname)
        return fname

    @staticmethod
    def _create_fname(record_dir, fname):
        """
        Creates the file name path using the current datetime if fname is None.
        """
        fname = fname if fname is not None \
            else time.strftime('%Y%m%d-%H%M%S', time.localtime())

        eve_file = record_dir / f'{fname}-eve.txt'

        return fname, eve_file

    # --------------------------------------------------------------------
    @property
    def record_dir(self):
        """
        Path to the directory where the data will be saved.

        :setter: Change the path if a recording is not on-going.
        :type: `str` | `~pathlib.Path`
        """
        return self._record_dir

    @record_dir.setter
    def record_dir(self, record_dir):
        if self._state.value == 1:
            logger.warning(
                'The recording directory cannot be changed during an '
                'ongoing recording.')
        else:
            self._record_dir = StreamRecorder._check_record_dir(record_dir)

    @property
    def fname(self):
        """
        Set file name stem.

        :setter: Change the file name stem if a recording is not on-going.
        :type: `str`
        """
        return self._fname

    @fname.setter
    def fname(self, fname):
        if self._state.value == 1:
            logger.warning(
                'The file name cannot be changed during an '
                'ongoing recording.')
        else:
            self._fname = StreamRecorder._check_fname(fname)

    @property
    def stream_name(self):
        """
        Servers' name or list of servers' name to connect to.

        :setter: Change the stream to record if a recording is not on-going.
        :type: `str` | `list`
        """
        return self._stream_name

    @stream_name.setter
    def stream_name(self, stream_name):
        if self._state.value == 1:
            logger.warning(
                'The stream name(s) to connect to cannot be changed during an '
                'ongoing recording.')
        else:
            self._stream_name = stream_name

    @property
    def eve_file(self):
        """
        Path to the event file for `~bsl.triggers.software.TriggerSoftware`.

        :type: `~pathlib.Path`
        """
        return self._eve_file

    @property
    def state(self):
        """
        Recording state of the recorder:
            - ``0``: Not recording.
            - ``1``: Recording.

        :type: `multiprocessing.Value`
        """
        return self._state

    @property
    def process(self):
        """
        Launched process.

        :type: `multiprocessing.Process`
        """
        return self._process


@fill_doc
class _Recorder:
    """
    Class creating the .pcl files, recording data through a StreamReceiver and
    saving the data in the .pcl and .fif files.

    Parameters
    ----------
    %(recorder_record_dir)s
    %(recorder_fname)s
    eve_file : str | Path
        Path to the event file for TriggerSoftware.
    %(recorder_fif_subdir)s
    %(stream_name)s
    state : mp.Value
        Recording state of the recorder:
            - 0: Not recording.
            - 1: Recording.
        This variable is used to stop the recording from another process.
    %(recorder_verbose)s
    """

    def __init__(self, record_dir, fname, eve_file, fif_subdir,
                 stream_name, state, verbose):
        self._record_dir = record_dir
        self._fname = fname
        self._eve_file = eve_file
        self._fif_subdir = fif_subdir
        self._stream_name = stream_name
        self._state = state
        self._verbose = verbose

    def record(self):
        """
        Instantiate a StreamReceiver, create the files, record and save.
        """
        sr = StreamReceiver(
            bufsize=MAX_BUF_SIZE, stream_name=self._stream_name)
        pcl_files = _Recorder._create_files(self._record_dir, self._fname, sr)

        with self._state.get_lock():
            self._state.value = 1

        if self._verbose:
            previous_time = -1  # init to -1 to start log at 00:00:00
            verbose_timer = Timer()

        # Acquisition loop
        while self._state.value == 1:
            sr.acquire()

            if self._verbose:
                if verbose_timer.sec() - previous_time >= 1:
                    previous_time = verbose_timer.sec()
                    duration = str(datetime.timedelta(
                        seconds=int(verbose_timer.sec())))
                    logger.info(f'RECORDING {duration}')

        self._save(sr, pcl_files)

    def _save(self, sr, pcl_files):
        """
        Save the data in the StreamReceiver buffer to the .pcl and .fif files.
        """
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

            if self._fif_subdir:
                out_dir = None
            else:
                out_dir = self._record_dir

            if self._eve_file.exists():
                logger.info('Found matching event file, adding events.')
                pcl2fif(
                    pcl_files[stream], out_dir, external_event=self._eve_file)
            else:
                pcl2fif(
                    pcl_files[stream], out_dir, external_event=None)

    # --------------------------------------------------------------------
    @staticmethod
    def _create_files(record_dir, fname, sr):
        """
        Create the .pcl files and check writability.
        """
        os.makedirs(record_dir, exist_ok=True)

        pcl_files = dict()
        for stream in sr.streams:
            pcl_files[stream] = \
                record_dir / f'{fname}-{stream}-raw.pcl'

            try:
                with open(pcl_files[stream], 'w') as file:
                    file.write(
                        'Data will be written when the recording is finished.')
            except Exception as error:
                logger.error(
                    f"Problem writing to '{pcl_files[stream]}'. "
                    "Check permissions.", exc_info=True)
                raise error

        logger.info(
            'Record to files: \n' +
            '\n'.join(str(file) for file in pcl_files.values()))

        return pcl_files
