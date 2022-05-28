import datetime
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

from ..stream_receiver import StreamEEG, StreamReceiver
from ..stream_receiver._stream import MAX_BUF_SIZE
from ..utils import Timer
from ..utils._checks import _check_type
from ..utils._docs import fill_doc
from ..utils._logs import logger
from ..utils.io import pcl2fif


@fill_doc
class StreamRecorder:
    """Class for recording the signals coming from LSL streams.

    Parameters
    ----------
    %(recorder_record_dir)s
    %(recorder_fname)s
    %(stream_name)s
    %(recorder_fif_subdir)s
    %(recorder_verbose)s
    """

    def __init__(
        self,
        record_dir=None,
        fname=None,
        stream_name=None,
        fif_subdir=True,
        *,
        verbose=False,
    ):
        self._record_dir = StreamRecorder._check_record_dir(record_dir)
        self._fname = StreamRecorder._check_fname(fname)
        _check_type(
            stream_name, (None, str, list, tuple), item_name="stream_name"
        )
        self._stream_name = stream_name
        _check_type(fif_subdir, (bool,), item_name="fif_subdir")
        self._fif_subdir = fif_subdir
        _check_type(verbose, (bool,), item_name="verbose")
        self._verbose = verbose

        self._eve_file = None  # for SOFTWARE triggers
        self._process = None
        self._state = mp.Value("i", 0)

    def start(self, blocking=True) -> None:
        """Start the recording in a new process.

        Parameters
        ----------
        blocking : bool
            If ``True``, waits for the child process to start recording data.
        """
        _check_type(blocking, (bool,), item_name="blocking")
        fname, self._eve_file = StreamRecorder._create_fname(
            self._record_dir, self._fname
        )
        logger.debug("File name stem is '%s'.", fname)
        logger.debug("Event file name is '%s'.", self._eve_file)

        self._process = mp.Process(
            target=self._record,
            args=(
                self._record_dir,
                fname,
                self._stream_name,
                self._fif_subdir,
                self._verbose,
                self._eve_file,
                self._state,
            ),
        )
        self._process.start()

        if blocking:
            while self._state.value == 0:
                pass

        logger.info(self.__repr__())

    def stop(self) -> None:
        """Stop the recording."""
        if self._process is None:
            logger.warning("StreamRecorder was not started. Skipping.")
            return

        with self._state.get_lock():
            self._state.value = 0

        logger.info("Waiting for StreamRecorder process to finish.")
        self._process.join(10)
        if self._process.is_alive():
            logger.error("Recorder process not finishing..")
            self._process.kill()
            raise RuntimeError
        logger.info("Recording finished.")

        self._eve_file = None
        self._process = None

    def _record(
        self,
        record_dir,
        fname,
        stream_name,
        fif_subdir,
        verbose,
        eve_file,
        state,
    ):  # noqa: D401
        """Function called in the new process.

        Instantiate a _Recorder and start recording.
        """
        recorder = _Recorder(
            record_dir,
            fname,
            stream_name,
            fif_subdir,
            verbose,
            eve_file,
            state,
        )
        recorder.record()

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
        streams = (
            self._stream_name
            if self._stream_name is not None
            else "All streams"
        )
        return f"<Recorder: {streams} | {status} | {self._record_dir}>"

    # --------------------------------------------------------------------
    @staticmethod
    def _check_record_dir(record_dir):  # noqa
        """
        Convert record_dir to a Path, or select the current working directory
        if record_dir is None.
        """
        _check_type(record_dir, (None, "path-like"), item_name="record_dir")
        if record_dir is None:
            record_dir = Path.cwd()
        else:
            record_dir = Path(record_dir)
        return record_dir

    @staticmethod
    def _check_fname(fname):
        """Check that the file name stem is a string or None."""
        _check_type(fname, (None, str), item_name="fname")
        return fname

    @staticmethod
    def _create_fname(record_dir, fname):
        """Create the file name path from the datetime if fname is None."""
        fname = (
            fname
            if fname is not None
            else time.strftime("%Y%m%d-%H%M%S", time.localtime())
        )
        eve_file = record_dir / f"{fname}-eve.txt"

        return fname, eve_file

    # --------------------------------------------------------------------
    @property
    def record_dir(self):
        """Path to the directory where data will be saved.

        :type: Path
        """
        return self._record_dir

    @property
    def fname(self):
        """File name stem used to create the files.

        The StreamRecorder creates 2 files plus an optional third if a software
        trigger was used, respecting the following naming:

        .. code-block:: python

            PCL: '{fname}-[stream_name]-raw.pcl'
            FIF: '{fname}-[stream_name]-raw.fif'
            (optional) SOFTWARE trigger events: '{fname}-eve.txt'

        :type: str
        """
        return self._fname

    @property
    def stream_name(self):
        """
        Servers' name or list of servers' name to connect to.

        :type: str | list
        """
        return self._stream_name

    @property
    def fif_subdir(self):
        """
        If ``True``, the ``.pcl`` files are converted to ``.fif`` in a
        subdirectory ``'fif': record_dir/fif/...`` instead of ``record_dir``.

        :type: bool
        """
        return self._fif_subdir

    @property
    def verbose(self):
        """
        If ``True``, a timer showing since when the recorder started is
        displayed every seconds.

        :type: bool
        """
        return self._verbose

    @property
    def eve_file(self):
        """
        Path to the event file for SoftwareTrigger.

        :type: Path
        """
        return self._eve_file

    @property
    def process(self):
        """
        Launched process.

        :type: Process
        """
        return self._process

    @property
    def state(self):
        """
        Recording state of the recorder

        - ``0``: Not recording.
        - ``1``: Recording.

        :type: `multiprocessing.Value`
        """
        return self._state


@fill_doc
class _Recorder:  # noqa
    """
    Class creating the .pcl files, recording data through a StreamReceiver and
    saving the data in the .pcl and .fif files.

    Parameters
    ----------
    %(recorder_record_dir)s
    %(recorder_fname)s
    %(stream_name)s
    %(recorder_fif_subdir)s
    %(recorder_verbose)s
    state : mp.Value
        Recording state of the recorder:
            - 0: Not recording.
            - 1: Recording.
        This variable is used to stop the recording from another process.
    eve_file : str | Path
        Path to the event file for SoftwareTrigger.
    """

    def __init__(
        self,
        record_dir,
        fname,
        stream_name,
        fif_subdir,
        verbose,
        eve_file,
        state,
    ):
        self._record_dir = record_dir
        self._fname = fname
        self._stream_name = stream_name
        self._fif_subdir = fif_subdir
        self._verbose = verbose
        self._eve_file = eve_file
        self._state = state

    def record(self):
        """Instantiate a StreamReceiver, create the files, record and save."""
        sr = StreamReceiver(
            bufsize=MAX_BUF_SIZE, stream_name=self._stream_name
        )
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
                    duration = str(
                        datetime.timedelta(seconds=int(verbose_timer.sec()))
                    )
                    logger.info("RECORDING %s", duration)

        self._save(sr, pcl_files)

    def _save(self, sr, pcl_files):
        """Save the data in the receiver's buffer to the .pcl and .fif files."""
        logger.info("Saving raw data..")
        for stream in sr.streams:
            signals, timestamps = sr.get_buffer(stream)

            if isinstance(sr.streams[stream], StreamEEG):
                signals[:, 1:] *= 1e-6

            data = {
                "signals": signals,
                "timestamps": timestamps,
                "events": None,
                "sample_rate": sr.streams[stream].sample_rate,
                "channels": len(sr.streams[stream].ch_list),
                "ch_names": sr.streams[stream].ch_list,
                "lsl_time_offset": sr.streams[stream].lsl_time_offset,
            }

            with open(pcl_files[stream], "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("Saved to '%s'", pcl_files[stream])

            if not isinstance(sr.streams[stream], StreamEEG):
                continue
            logger.info("Converting raw files into fif.")

            if self._fif_subdir:
                out_dir = None
            else:
                out_dir = self._record_dir

            if self._eve_file.exists():
                logger.info("Found matching event file, adding events.")
                pcl2fif(
                    pcl_files[stream], out_dir, external_event=self._eve_file
                )
            else:
                pcl2fif(pcl_files[stream], out_dir, external_event=None)

    # --------------------------------------------------------------------
    @staticmethod
    def _create_files(record_dir, fname, sr):
        """Create the .pcl files and check writability."""
        os.makedirs(record_dir, exist_ok=True)

        pcl_files = dict()
        for stream in sr.streams:
            pcl_files[stream] = record_dir / f"{fname}-{stream}-raw.pcl"

            try:
                with open(pcl_files[stream], "w") as file:
                    file.write(
                        "Data will be written when the recording is finished."
                    )
            except Exception as error:
                raise error(
                    "Could not write to '%s'. Check permissions"
                    % {pcl_files[stream]}
                )

        logger.info(
            "Record to files: \n"
            + "\n".join(str(file) for file in pcl_files.values())
        )

        return pcl_files
