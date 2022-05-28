from ..utils._docs import fill_doc
from ..utils._logs import logger


class Buffer:
    """Class representing the stream's buffer.

    This is not a Python Buffer Interface compatible class.

    Parameters
    ----------
    bufsize : int
        Buffer's size [samples].
    winsize : int
        Window's size [samples]. Must be smaller than the buffer's size.
    """

    def __init__(self, bufsize, winsize):
        self._bufsize = bufsize
        self._winsize = winsize

        self._data = []
        self._timestamps = []

    @fill_doc
    def fill(self, data, tslist):
        """Fill the data and timestamps to the buffer.

        Parameters
        ----------
        data : list
            Received data [samples x channels].
        %(receiver_tslist)s
        """
        self._data.extend(data)
        self._timestamps.extend(tslist)
        logger.debug("Buffer filled with %d points.", len(tslist))

        if len(self._timestamps) > self._bufsize:
            self._data = self._data[-self._bufsize :]
            self._timestamps = self._timestamps[-self._bufsize :]

    def reset_buffer(self):
        """Clear the buffer's data and timestamps."""
        self._data = []
        self._timestamps = []
        logger.debug("Buffer reset.")

    @property
    def bufsize(self):
        """Buffer's size [samples]."""
        return self._bufsize

    @property
    def winsize(self):
        """Window's size [samples]."""
        return self._winsize

    @property
    def data(self):
        """Buffer's data [samples x channels]."""
        return self._data

    @property
    def timestamps(self):
        """Data's timestamps [samples]."""
        return self._timestamps
