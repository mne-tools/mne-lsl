from neurodecode import logger

class Buffer():
    """
    Class representing the stream's buffer.
    
    Parameters
    ----------
    buffer_size : int
        Buffer's size [samples].
    window_size : int
        To extract the latest winsize samples from the buffer [samples].
    lsl_bufsize :
        The LSL buffer size (can be smaller than buffer_size because stream_receiver acquire at regular interval)
    """
    #----------------------------------------------------------------------
    def __init__(self, buffer_size, window_size, lsl_bufsize):
        
        self._winsize = window_size
        self._bufsize = buffer_size
        self._lsl_bufsize = lsl_bufsize
        
        self._data = []
        self._timestamps = []
    
    #----------------------------------------------------------------------
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
        
    #----------------------------------------------------------------------
    def reset_buffer(self):
        """
        Clear the buffer's data and timestamps.
        """
        self._data = []
        self._timestamps = []
        
    #----------------------------------------------------------------------
    @property
    def winsize(self):
        """
        The window's size [samples].
        """
        return self._winsize
    
    #----------------------------------------------------------------------
    @winsize.setter
    def winsize(self, winsize):
        logger.warning("This attribute cannot be changed")
    
    #----------------------------------------------------------------------
    @property
    def bufsize(self):
        """
        The buffer's size [samples].
        """
        return self._bufsize
    
    #----------------------------------------------------------------------
    @bufsize.setter
    def bufsize(self, bufsize):
        logger.warning("This attribute cannot be changed")
        
    #----------------------------------------------------------------------
    @property
    def data(self):
        """
        Buffer's data [samples x channels].
        """
        return self._data
    
    #----------------------------------------------------------------------
    @data.setter
    def data(self, data):
        logger.warning("This attribute cannot be changed.")
    
    #----------------------------------------------------------------------
    @property
    def timestamps(self):
        """
        Data's timestamps [samples].
        """
        return self._timestamps
    
    #----------------------------------------------------------------------    
    @timestamps.setter
    def timestamps(self, timestamps):
        logger.warning("This attribute cannot be changed.")