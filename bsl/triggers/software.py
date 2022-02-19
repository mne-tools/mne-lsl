"""
Trigger using software and a .txt file.
"""
from ._trigger import _Trigger
from ..externals import pylsl
from ..stream_recorder import StreamRecorder
from ..utils._checks import _check_type
from ..utils._docs import fill_doc, copy_doc


@fill_doc
class SoftwareTrigger(_Trigger):
    """
    Trigger saving signal value in a ``.txt`` file.
    Software trigger instance must be created after a StreamRecorder is started
    and close/deleted before a StreamRecorder is stopped.

    Parameters
    ----------
    recorder : StreamRecorder
        BSL's recorder used.
    %(trigger_verbose)s

    Examples
    --------
    >>> recorder = StreamRecorder('path to dir')
    >>> recorder.start()
    >>> trigger = SoftwareTrigger(recorder)
    >>> trigger.signal(1)
    >>> trigger.close() # OR >>> del trigger
    >>> recorder.stop()
    """

    def __init__(self, recorder, *, verbose: bool = True):
        super().__init__(verbose)
        self._recorder = SoftwareTrigger._check_recorder(recorder)
        self._eve_file = SoftwareTrigger._find_eve_file(recorder)
        try:
            self._eve_file = open(self._eve_file, 'a')
        except Exception:
            raise

    @copy_doc(_Trigger.signal)
    def signal(self, value: int) -> bool:
        _check_type(value, ('int', ), item_name='value')
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
        _check_type(recorder, (StreamRecorder, ), item_name='recorder')
        return recorder

    @staticmethod
    def _find_eve_file(recorder):
        """
        Find the event file name from the on going StreamRecorder.
        """
        if recorder.eve_file is None:
            raise RuntimeError(
                'The StreamRecorder must be started before instantiating '
                'a SOFTWARE trigger.')

        return recorder.eve_file

    # --------------------------------------------------------------------
    @property
    def recorder(self):
        """
        BSL's recorder used.

        :type: StreamRecorder
        """
        return self._recorder

    @property
    def eve_file(self):
        """
        Event ``.ini`` file.

        :type: `~io.TextIOWrapper`
        """
        return self._eve_file
