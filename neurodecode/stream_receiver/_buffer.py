class Buffer():
    """
    Class representing the stream's buffer.
    This is not a Python Buffer Interface compatible class.

    Parameters
    ----------
    bufsize : int
        Buffer's size [samples].
    winsize : int
        To extract the latest winsize samples from the buffer [samples].
    """

    def __init__(self, bufsize, winsize):

        self._winsize = winsize
        self._bufsize = bufsize

        self._data = []
        self._timestamps = []

    def fill(self, data, tslist):
        """
        Fill the data and timestamps to the buffer.

        Parameters
        -----------
        data : list
            The received data [samples x channels].
        tslist : list
            The data's timestamps [samples].
        """
        self._data.extend(data)
        self._timestamps.extend(tslist)

        if len(self._timestamps) > self._bufsize:
            self._data = self._data[-self._bufsize:]
            self._timestamps = self._timestamps[-self._bufsize:]

    def reset_buffer(self):
        """
        Clear the buffer's data and timestamps.
        """
        self._data = []
        self._timestamps = []

    @property
    def winsize(self):
        """
        The window's size [samples].
        """
        return self._winsize

    @property
    def bufsize(self):
        """
        The buffer's size [samples].
        """
        return self._bufsize

    @property
    def data(self):
        """
        Buffer's data [samples x channels].
        """
        return self._data

    @property
    def timestamps(self):
        """
        Data's timestamps [samples].
        """
        return self._timestamps
