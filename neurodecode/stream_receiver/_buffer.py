from neurodecode import logger
from abc import ABC, abstractmethod

class _Buffer(ABC):
    """
    Abstract class representing a receiver's stream.
    
    Parameters
    ----------
    """
    #----------------------------------------------------------------------
    @abstractmethod
    def __init__(self, sample_rate, buffer_size=1, window_size=1):
        
        self.sample_rate = sample_rate
        self.winsec = self.check_window_size(window_size)
        self.bufsec= self.check_buffer_size(buffer_size)        
    
    #----------------------------------------------------------------------
    def check_window_size(self, window_size):
        """
        Check that buffer's size is positive.
        
        Parameters
        -----------
        window_size : float
            The buffer size to verify [secs].
                
        Returns
        --------
        secs
            The verified buffer's size.
        """        
        if window_size <= 0:
            logger.error('Wrong window_size %d.' % window_size)
            raise ValueError()
        return window_size
    
    #----------------------------------------------------------------------
    def check_buffer_size(self, buffer_size, max_buffer_size=86400):
        """
        Check that buffer's size is positive, smaller than max size and bigger than window's size.
        
        Parameters
        -----------
        buffer_size : float
            The buffer size to verify [secs].
        max_buffer_size : float
            max buffer size allowed by StreamReceiver, default is 24h [secs].
        
        Returns
        --------
        secs
            The verified buffer size.
        """
        if buffer_size <= 0 or buffer_size > max_buffer_size:
            logger.error('Improper buffer size %.1f. Setting to %d.' % (buffer_size, max_buffer_size))
            buffer_size = max_buffer_size
        
        elif buffer_size < self.winsec:
            logger.error('Buffer size %.1f is smaller than window size. Setting to %.1f.' % (buffer_size, self.winsec))
            buffer_size = self.winsec

        return buffer_size
    
    #----------------------------------------------------------------------
    @abstractmethod
    def convert_sec_to_samples(self, winsec):
        """
        Convert a window's size from sec to samples
        
        Parameters
        -----------
        winsec : float
            The window's size [secs].
        
        Returns
        --------
        samples
            The converted window's size.
        """
        
                
class _BufferEEG(_Buffer):
    """
    Class representing the receiver's buffer.
    
    Parameters
    ----------
    sample_rate : Hertz
        The sampling rate of the corresponding stream.
    buffer_size : secs
        1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently.
    window_size : secs
        Extract the latest window_size seconds of the buffer.
    
    Attributes
    ----------
    data : list
        Buffer's data [samples x channels].
    timestamps : list
        Data's timestamps [samples].
    sample_rate : int
        The sampling rate of the receiving stream [Hz].
    bufsize : int
        Buffer's size [samples].
    winsize : int
        Window's size [samples].
    winsec: float
            To extract the latest window_size of the buffer [secs].
    bufsec : float
            Buffer's size [secs]
    """
    #----------------------------------------------------------------------
    def __init__(self, sample_rate, buffer_size=1, window_size=1):
        
        super().__init__()
        
        self.winsize = self.convert_sec_to_samples(self.winsec)
        self.bufsize = self.convert_sec_to_samples(self.bufsec)
        
        self.data = []
        self.timestamps = []
    
    #----------------------------------------------------------------------
    def convert_sec_to_samples(self, winsec):
        """
        Convert a window's size from sec to samples
        
        Parameters
        -----------
        winsec : float
            The window's size [secs].
        
        Returns
        --------
        samples
            The converted window's size.
        """
        return int(round(winsec * self.sample_rate))
    
    #----------------------------------------------------------------------
    def fill(self, data, tslist, lsl_clock=None):
        """
        Fill the data and timestamp to the buffer.
        
        Parameters
        -----------
        list 
            The received data [samples x channels].
        list
            The data's timestamps [samples].
        """
        self.data.extend(data.tolist())
        self.timestamps.extend(tslist)
        
        if self.bufsize > 0 and len(self.timestamps) > self.bufsize:
            self.data = self.data[-self.bufsize:]
            self.timestamps = self.timestamps[-self.bufsize:]
        
        if lsl_clock:
            self.compute_offset(lsl_clock)
    
    #----------------------------------------------------------------------
    def compute_offset(self, lsl_clock):
        """
        Compute the LSL offset coming from some devices.
        
        Parameters
        -----------
        lsl_clock : float
            The lsl clock when the last sample was acquired [secs]
        """
        logger.info('LSL timestamp = %s' % lsl_clock)
        logger.info('Server timestamp = %s' % self.timestamps[-1]) 
        self.lsl_time_offset = self.timestamps[-1] - lsl_clock
        logger.info('Offset = %.3f ' % (self.lsl_time_offset))
        
        if abs(self.lsl_time_offset) > 0.1:
            logger.warning('LSL server has a high timestamp offset.')
        else:
            logger.info_green('LSL time server synchronized')
        
    #----------------------------------------------------------------------
    def set_window_size(self, window_size):
        """
        Set the window's size.
        
        Parameters
        -----------
        window_size : int
            The window's size [samples].
        """
        self.window_size = self.check_window_size(window_size)
    
    #----------------------------------------------------------------------
    def get_buflen(self):
        """
        Return the buffer's length in seconds.

        Returns
        -------
        float
            Buffer's length [secs]
        """
        return (len(self.timestamps) / self.sample_rate)            
        
    #----------------------------------------------------------------------
    def get_sample_rate(self):
        """
        Return the sampling rate.

        Returns
        -------
        int
            Sampling rate [Hz]
        """
        return self.sample_rate
    
    #----------------------------------------------------------------------
    def get_lsl_offset(self, stream_index=0):
        """
        Get the time difference between the acquisition server's time and LSL time.

        OpenVibe servers often have a bug of sending its own running time instead of LSL time.
        
        Returns
        -------
        float
            The lsl offset [secs]
        """
        return self.lsl_time_offset
    
    #----------------------------------------------------------------------
    def reset_buffer(self, stream_index=0):
        """
        Clear the buffer's data and timestamps.
        """
        self.data = []
        self.timestamps = []