"""
Trigger using software and a .txt file.
"""
import pylsl

from .._trigger import _Trigger
from ... import logger


class TriggerSoftware(_Trigger):
    """
    Trigger saving signal value in a .txt file.
    The SOFTWARE trigger must be created after a stream recorder is started:
        >>> recorder = StreamRecorder('path to dir')
        >>> recorder.start()
        >>> trigger = TriggerSoftware(recorder)

    Parameters
    ----------
    recorder : StreamRecorder
        The neurodecode recorder used.
    verbose : bool
        If True, display a logger.info message when a trigger is sent.
    """

    def __init__(self, recorder, verbose=True):
        super().__init__(verbose)
        self._recorder = recorder
        self._evefile = TriggerSoftware._find_evefile(recorder)
        try:
            self._evefile = open(self._evefile, 'a')
        # Close it before if already opened.
        except IOError:
            self._evefile.close()
            self._evefile = open(self._evefile, 'a')

    def signal(self, value):
        """
        Send a trigger value.
        """
        self._set_data(value)
        super().signal(value)
        return True

    def _signal_off(self):
        """
        Reset trigger signal to 0. Not needed for SOFTWARE triggers.
        """
        pass

    def _set_data(self, value):
        """
        Set the trigger signal to value.
        """
        self._evefile.write('%.6f\t0\t%d\n' % (pylsl.local_clock(), value))

    def close(self):
        """
        Close the event file.
        """
        try:
            self._eve_file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # --------------------------------------------------------------------
    @staticmethod
    def _find_evefile(recorder):
        """
        Find the event file name from the on going StreamRecorder.
        """
        if recorder.evefile is None:
            logger.error(
                'The StreamRecorder must be started before instantiating '
                'a SOFTWARE trigger.')
            raise RuntimeError

        return recorder.evefile

    # --------------------------------------------------------------------
    @property
    def evefile(self):
        """
        The event .txt file.
        """
        return self._evefile

    @evefile.setter
    def evefile(self, evefile):
        logger.warning('This attribute cannot be changed directly.')
