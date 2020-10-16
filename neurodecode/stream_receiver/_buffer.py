from neurodecode import logger

class Buffer():
    """
    Class representing the receiver's buffer
    
    Parameters
    ----------
    buffer_size : int
        Buffer's size in number of samples. 1-day is the maximum size.
        Large buffer may lead to a delay if not pulled frequently.
    window_size : int
        The latest window to extract from the buffer in number of samples.
    """
    #----------------------------------------------------------------------
    def __init__(self, buffer_size, window_size):
        
        self._winsize = window_size
        self._bufsize = buffer_size
        
        self._data = []
        self._timestamps = []
        self._lsl_time_offset = None
    
    #----------------------------------------------------------------------
    def fill(self, data, tslist, lsl_clock=None):
        """
        Fill the data and timestamp to the buffer.
        
        Parameters
        -----------
        data : list 
            The received data [samples x channels].
        tslist : list
            The data's timestamps [samples].
        lsl_clock : float
            The lsl clock when the last sample was acquired [secs]
        """
        # self._data.extend(data.tolist())
        self._data.extend(data)
        self._timestamps.extend(tslist)
        
        if len(self._timestamps) > self._bufsize:
            self._data = self._data[-self._bufsize:]
            self._timestamps = self._timestamps[-self._bufsize:]
        
        if lsl_clock:
            self._compute_offset(lsl_clock)
    
    #----------------------------------------------------------------------
    def _compute_offset(self, lsl_clock):
        """
        Compute the LSL offset coming from some devices.
        
        Parameters
        -----------
        lsl_clock : float
            The lsl clock when the last sample was acquired [secs]
        """
        self._lsl_time_offset = self.timestamps[-1] - lsl_clock
 
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
        The window's size in samples
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
        The buffer's size in samples
        """
        return self._bufsize
    
    #----------------------------------------------------------------------
    @bufsize.setter
    def bufsize(self, bufsize):
        logger.warning("This attribute cannot be changed")
    
    #----------------------------------------------------------------------
    @property
    def lsl_time_offset(self):
        """
        The difference between the local and the server LSL times.
        
        OpenVibe servers often have a bug of sending its own running time instead of LSL time.
        """
        return self._lsl_time_offset
    
    #----------------------------------------------------------------------
    @lsl_time_offset.setter
    def lsl_time_offset(self, lsl_time_offset):
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
    
    #----------------------------------------------------------------------
    #def _check_window_size(self, window_size):
        #"""
        #Check that buffer's window size is positive.
        
        #Parameters
        #-----------
        #window_size : float
            #The buffer size to verify [secs].
                
        #Returns
        #--------
        #secs
            #The verified buffer's size.
        #"""        
        #if window_size <= 0:
            #logger.error('Wrong window_size %d.' % window_size)
            #raise ValueError()
        #return window_size
    
    ##----------------------------------------------------------------------
    #def _check_buffer_size(self, bufsize, max_buffer_size=86400):
        #"""
        #Check that buffer's size is positive, smaller than max size and bigger than window's size.
        
        #Parameters
        #-----------
        #bufsize : float
            #The buffer size to verify [secs].
        #max_buffer_size : float
            #max buffer size allowed by StreamReceiver, default is 24h [secs].
        
        #Returns
        #--------
        #secs
            #The verified buffer size.
        #"""
        #if bufsize <= 0 or bufsize > max_buffer_size:
            #logger.error('Improper buffer size %.1f. Setting to %d.' % (bufsize, max_buffer_size))
            #bufsize = max_buffer_size
        
        #elif bufsize < self.winsize:
            #logger.error('Buffer size %.1f is smaller than window size. Setting to %.1f.' % (bufsize, self.winsec))
            #bufsize = self.winsize

        #return bufsize
    