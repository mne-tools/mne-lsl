
import time
import multiprocessing as mp

from ._recorder import _Recorder
from .. import logger
from ..gui.streams import redirect_stdout_to_queue


class StreamRecorder:
    """
    Class for recording the signals coming from LSL streams.

    Parameters
    ----------
    record_dir : str
        The directory where the data will be saved.
    logger : Logger
        The logger where to output info. Default is the NeuroDecode logger.
    state : mp.Value
        Multiprocessing sharing variable to stop the recording from another
        process.
    queue : mp.Queue
        Can redirect sys.stdout to a queue (e.g. used for GUI).
    """

    def __init__(self, record_dir, logger=logger,
                 state=mp.Value('i', 0), queue=None):

        self._logger = logger
        self._queue = queue

        if record_dir is None:
            self._logger.error("No recording directory provided.")
            raise RuntimeError

        self._proc = None
        self._amp_name = None
        self._record_dir = record_dir

        self._state = state

    def start(self, amp_name=None, eeg_only=False, verbose=False):
        """
        Start recording data from LSL network, in a new process.

        Parameters
        ----------
        amp_name : str
            Connect to a server named 'amp_name'. None: no constraint.
        eeg_only : bool
            If true, ignore non-EEG servers.
        verbose : bool
            IF true, it will print every second the time since the recording
            start.
        """
        self._amp_name = amp_name

        self._proc = mp.Process(target=self._record,
                                args=(amp_name,
                                      self._record_dir,
                                      eeg_only,
                                      verbose,
                                      self._logger,
                                      self._queue,
                                      self._state))
        self._proc.start()

        while not self._state.value:
            pass

    def wait(self, timeout):
        """
        Wait that the data streaming finishes.

        Parameters
        ----------
        timeout : float
            Block until timeout is reached.
            If None, block until streaming is finished.
        """
        self._proc.join(timeout)

    def stop(self):
        """
        Stop the recording.
        """
        with self._state.get_lock():
            self._state.value = 0

        self._logger.info('(main) Waiting for recorder process to finish.')
        self._proc.join(10)
        if self._proc.is_alive():
            self._logger.error(
                'Recorder process not finishing. Are you running from Spyder?')
        self._logger.info('Recording finished.')

    def _record(self, amp_name, record_dir, eeg_only,
                verbose, logger, queue, state):
        """
        The function launched in a new process.
        """
        redirect_stdout_to_queue(logger, queue, 'INFO')

        recorder = _Recorder(record_dir, logger, state)
        recorder.connect(amp_name, eeg_only)
        recorder.record(verbose)

    def _start_gui(self, protocolState, amp_name, record_dir, eeg_only,
                   logger, queue, state):
        """
        Start the recording when launched from the GUI.
        """
        self._proc = mp.Process(target=self._record,
                                args=(amp_name,
                                      record_dir,
                                      eeg_only,
                                      True,
                                      logger,
                                      queue,
                                      state))
        self._proc.start()

        while not state.value:
            pass

        # Launching the protocol (shared variable)
        with protocolState.get_lock():
            protocolState.value = 1

        # Continue recording until the shared variable changes to 0.
        while protocolState.value:
            time.sleep(1)
        self.stop()

    @property
    def process(self):
        """
        The process where the recording takes place.

        Gives access to all the function associated with mp.Process.

        Returns
        -------
        mp.Process
        """
        return self._proc

    @process.setter
    def process(self):
        self._logger.warning("This attribute cannot be changed.")

    @property
    def amp_name(self):
        """
        The provided amp_name to connect to.

        If None, it will connect to all available streams.

        Returns
        -------
        str
        """
        return self._amp_name

    @amp_name.setter
    def amp_name(self):
        self._logger.warning("This attribute cannot be changed.")

    @property
    def record_dir(self):
        """
        The absolute directory where the data are saved.

        Returns
        -------
        str
        """
        return self._record_dir

    @record_dir.setter
    def record_dir(self):
        self._logger.warning("This attribute cannot be changed.")

    @property
    def state(self):
        """
        Multiprocessing sharing variable to stop the recording from another
        process.
        """
        return self._state

    @state.setter
    def state(self):
        """
        Multiprocessing sharing variable to stop the recording from another
        process.
        """
        self._logger.warning("This attribute cannot be changed.")
