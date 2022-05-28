import time

from ._checks import _check_type


class Timer:
    """Timer class.

    Parameters
    ----------
    autoreset : bool
        If ``autoreset=True``, timer is reset after any member function call.
    """

    def __init__(self, autoreset=False):
        self.autoreset = autoreset
        self.reset()

    def sec(self):
        """Provide the time since reset in seconds."""
        read = time.time() - self.ref
        if self._autoreset:
            self.reset()
        return read

    def msec(self):
        """Provide the time since reset in milliseconds."""
        return self.sec() * 1000.0

    def reset(self) -> None:
        """Reset the timer to zero."""
        self.ref = time.time()

    def sleep_atleast(self, sec):
        """Sleep up to sec seconds.

        Parameters
        ----------
        sec : float
            Time to sleep in seconds.
        """
        timer_sec = self.sec()

        if timer_sec < sec:
            time.sleep(sec - timer_sec)
            if self._autoreset:
                self.reset()

    @property
    def autoreset(self):
        """Autoreset status.

        :type: bool
        """
        return self._autoreset

    @autoreset.setter
    def autoreset(self, autoreset):
        _check_type(autoreset, (bool,), item_name="autoreset")
        self._autoreset = autoreset
