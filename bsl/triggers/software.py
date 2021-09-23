"""
Trigger using software and a .txt file.
"""
import pylsl

from ._trigger import _Trigger
from .. import logger
from ..utils._docs import fill_doc, copy_doc
from ..stream_recorder import StreamRecorder


@fill_doc
class TriggerSoftware(_Trigger):
    """
    Trigger saving signal value in a ``.txt`` file.
    Software trigger instance must be created after a
    `~bsl.StreamRecorder` is started and close/deleted before a
    `~bsl.StreamRecorder` is stopped.

    .. code-block:: python

        recorder = StreamRecorder('path to dir')
        recorder.start()
        trigger = TriggerSoftware(recorder)
        # do stuff
        trigger.close() # OR >>> del trigger
        recorder.stop()

    Parameters
    ----------
    recorder : `~bsl.StreamRecorder`
        BSL's `~bsl.StreamRecorder` used.
    %(trigger_verbose)s
    """

    def __init__(self, recorder, verbose: bool = True):
        super().__init__(verbose)
        self._recorder = TriggerSoftware._check_recorder(recorder)
        self._eve_file = TriggerSoftware._find_eve_file(recorder)
        try:
            self._eve_file = open(self._eve_file, 'a')
        except Exception as err:
            raise err(f'Could not open {self._eve_file} in append mode.')

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
        self._set_data(value)
        super().signal(value)
        return True

    @copy_doc(_Trigger._set_data)
    def _set_data(self, value: int):
        """
        Set the trigger signal to value.
        """
        super()._set_data(value)
        self._eve_file.write('%.6f\t0\t%d\n' % (pylsl.local_clock(), value))

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
    def _check_recorder(recorder):
        """
        Check that the provided recorder is indeed a StreamRecorder.
        """
        if not isinstance(recorder, StreamRecorder):
            logger.error(
                'You must pass a StreamRecorder instance to the '
                'SOFTWARE triggers.')
            raise TypeError

        return recorder

    @staticmethod
    def _find_eve_file(recorder):
        """
        Find the event file name from the on going StreamRecorder.
        """
        if recorder.eve_file is None:
            logger.error(
                'The StreamRecorder must be started before instantiating '
                'a SOFTWARE trigger.')
            raise RuntimeError

        return recorder.eve_file

    # --------------------------------------------------------------------
    @property
    def recorder(self):
        """
        BSL's `~bsl.StreamRecorder` used.

        :type: `~bsl.StreamRecorder`
        """
        return self._recorder

    @property
    def eve_file(self):
        """
        Event ``.ini`` file.

        :type: `~pathlib.Path`
        """
        return self._eve_file
