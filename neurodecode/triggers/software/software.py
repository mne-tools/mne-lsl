"""
Trigger using software and a .txt file.

TODO: The Software trigger requires an associated StreamRecorder. A recorder
should be passed as argument in the initialization.
"""
import pylsl

from .._trigger import _Trigger
from ... import logger
from ...utils.lsl import start_client


class TriggerSoftware(_Trigger):
    def __init__(self, verbose=True):
        super().__init__(verbose)
        self._evefile = TriggerSoftware._find_evefile()
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
    def _find_evefile(timeout=10):
        """
        Find the event file name from the on going StreamRecorder.
        """
        inlet = start_client(server_name='StreamRecorderInfo', timeout=timeout)
        evefile = inlet.info().source_id()
        logger.info(f'Event file is: {evefile}')
        return evefile

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
