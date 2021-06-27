import time
import pickle
from pathlib import Path
import multiprocessing as mp

from .. import logger
from ..utils.io import pcl2fif, make_dirs
from ..stream_receiver import StreamReceiver, StreamEEG
from ..stream_receiver._stream import MAX_BUF_SIZE


class StreamRecorder:
    def __init__(self, record_dir, fname=None, stream_name=None):
        self._record_dir = StreamRecorder._check_record_dir(record_dir)
        self._fname = StreamRecorder._check_fname(fname)
        self._stream_name = StreamRecorder._check_stream_name(stream_name)

        self.eve_file = None # for SOFTWARE triggers
        self._proc = None
        self._state = mp.Value('i', 0)

    def start(self, verbose=False):
        pcl_files, self.eve_file = self.create_files()

        self._proc = mp.Process(
            target=self._record,
            args=(self._stream_name, pcl_files, self.eve_file,
                  self._state, verbose))
        self._proc.start()

    def stop(self):
        with self._state.get_lock():
            self._state.value = 0

        logger.info('Waiting for recorder process to finish.')
        self._proc.join(10)
        if self._proc.is_alive():
            logger.error(
                'Recorder process not finishing..')
            raise RuntimeError
        logger.info('Recording finished.')

        self.eve_file = None
        self._proc = None

    def create_files(self):
        # Filenames
        fname = self._fname if self._fname is not None \
            else time.strftime('%Y%m%d-%H%M%S', time.localtime())

        eve_file = self._record_dir / f'{fname}-eve.txt'

        pcl_files = dict()
        for stream in self._stream_name:
            pcl_files[stream] = self._record_dir / f'{fname}-{stream}-raw.pcl'

        # Check writability
        make_dirs(self._record_dir)
        for stream in pcl_files:
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
            'Record to files:\n'
            '\n'.join(str(file) for file in pcl_files.values()))

        return pcl_files, eve_file

    def _record(self, stream_name, pcl_files, eve_file, state, verbose):
        sr = StreamReceiver(bufsize=MAX_BUF_SIZE, stream_name=stream_name)

        with state.get_lock():
            state.value = 1

        # Acquisition loop
        while state.value == 1:
            sr.acquire()

            if verbose:
                pass # TODO: Add timing display

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

            if eve_file.exists():
                logger.info('Found matching event file, adding events.')
                pcl2fif(pcl_files[stream], external_event=eve_file)
            else:
                pcl2fif(pcl_files[stream], external_event=None)

    # --------------------------------------------------------------------
    @staticmethod
    def _check_record_dir(record_dir):
        return Path(record_dir)

    @staticmethod
    def _check_fname(fname):
        if fname is not None:
            fname = str(fname)
        return fname

    @staticmethod
    def _check_stream_name(stream_name):
        return stream_name

    # --------------------------------------------------------------------
