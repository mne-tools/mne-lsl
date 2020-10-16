import time
import pylsl
import numpy as np

from threading import Thread
from neurodecode import logger
from neurodecode.stream_receiver import StreamEEG, StreamMarker

import neurodecode.utils.q_common as qc

class StreamReceiver:
    """
    Class for data acquisition from LSL streams.
    
    It now supports eeg and markers streams.
        
    Parameters
    ----------
    window_size : float
        To extract the latest window_size seconds of the buffer [secs].
    buffer_size : float
        1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently [secs].
    amp_name : str
        Connect to a server named 'amp_name'. None: no constraint.
    eeg_only : bool
        If true, ignore non-EEG servers.
    """
    #----------------------------------------------------------------------
    def __init__(self, window_size=1, buffer_size=1, amp_name=None, eeg_only=False):
        
        self._streams = []
        self._is_connected = False
        
        self.connect(window_size, buffer_size, amp_name, eeg_only)
    
    #----------------------------------------------------------------------
    def connect(self, window_size=1, buffer_size=1, amp_name=None, eeg_only=False):
        """
        Search for the available streams on the LSL network and connect to the appropriate ones.
        If a LSL stream fullfills the requirements (name...), a connection is established.
        
        This function is called while instanciating a StreamReceiver and call be recall to connect
        to new LSL streams.
        
        Parameters
        ----------
        window_size : float
            To extract the latest window_size seconds of the buffer [secs].
        buffer_size : float
            1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently [secs].
        amp_name : list
            List of servers' name to connect to. None: no constraint.
        eeg_only : bool
            If true, ignore non-EEG servers.
        """
        self._streams = []
        
        self._is_connected = False        
        server_found = False
                
        while server_found == False:
            
            if amp_name is None:
                logger.info("Looking for available lsl streaming servers...")
            else:
                logger.info("Looking for server: {} ..." .format(amp_name))
                
            streamInfos = pylsl.resolve_streams()
            
            if len(streamInfos) > 0:
                for si in streamInfos:
                    
                    # EEG streaming server only?
                    if eeg_only and si.type() != 'EEG':
                        continue
                    # connect to a specific amp only?
                    if amp_name is not None and si.name() not in amp_name:
                        continue
                    
                    if si.type().lower() == "eeg":
                        self._streams.append(StreamEEG(si, buffer_size, window_size))
                    
                    elif si.nominal_srate() == 0:
                        self._streams.append(StreamMarker(si, buffer_size, window_size))
                    
                    server_found = True
            time.sleep(1)
        
        self.show_info()
        
        self.acquire()
        self._is_connected = True
        logger.info('Ready to receive data from the connected streams.')
    
    
    #----------------------------------------------------------------------
    def show_info(self):
        """
        Display the informations about all the connected streams
        """
        for i in range(len(self.streams)):
            logger.info("----------------------------------------------------------------")
            logger.info("The stream {} is connected to:".format(i))
            self.streams[i].show_info()
            # logger.info("----------------------------------------------------------------\n")
            
    #----------------------------------------------------------------------
    def acquire(self, blocking=True, timestamp_offset = False):
        """
        Read data from each streams and fill their buffer using threading.
        
        It is a blocking function as default.
          
        Parameters
        ----------
        blocking : bool
            If True, the streams wait to receive some data.
        """
        threads = []
        
        for i in range(len(self._streams)):
            t = Thread(target=self._acquire, args=[blocking, i, timestamp_offset])
            t.daemon = True
            t.start()
            threads.append(t)
        
        if blocking is True:
            self._wait_threads_to_finish(threads)
             
    #----------------------------------------------------------------------
    def get_window(self, stream_index=0):
        """
        Get the latest window of a stream in numpy format.

        Parameters
        ----------
        stream_index : int
            The index of the stream to get the window (default = 0).

        Returns
        -------
        np.array
             The window data [[samples_ch1],[samples_ch2]...]
        np.array
             The timestamps [samples]
        """
        self.is_connected
        window, timestamps = self._get_window_list()

        if len(timestamps) > 0:
            return np.array(window), np.array(timestamps)
        else:
            return np.array([]), np.array([])
        
    ##----------------------------------------------------------------------
    #def _get_buffer_list(self, stream_index=0):
        #"""
        #Get the entire buffer of a stream.
        
        #Parameters
        #----------
        #stream_index : int
            #The index of the stream to get the buffer.
        
        #Returns
        #--------
        #list
            #The buffer data [samples x channels]
        #list
            #Its timestamps [samples]
        #"""
        #self.is_connected()
        #return self._buffers[stream_index].data, self._buffers[stream_index].timestamps
    
    #----------------------------------------------------------------------
    def get_buffer(self, stream_index=0):
        """
        Get the entire buffer of a stream in numpy format.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer (default = 0).

        Returns
        -------
        np.array
            The data [[samples_ch1],[samples_ch2]...]
        np.array
            Its timestamps [samples]
        """
        self.is_connected
        
        if len(self.streams[stream_index].buffer.timestamps) > 0:
            return np.array(self.streams[stream_index].buffer.data), np.array(self.streams[stream_index].buffer.timestamps)
        else:
            return np.array([]), np.array([])
    
    #----------------------------------------------------------------------
    def reset_buffer(self, stream_index=0):
        """
        Clear the buffer of a stream.
                
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer (default = 0).
        """
        self.streams[stream_index].buffer.reset_buffer()
        
    #----------------------------------------------------------------------
    def reset_all_buffers(self):
        """
        Clear the buffer of all the streams.
        """
        for i in range(len(self._streams)):
            self.streams[i].buffer.reset_buffer()
            
    #----------------------------------------------------------------------
    def _get_window_list(self, stream_index=0):
        """
        Get the latest window from a buffer.
        
        Parameters
        -----------
        stream_index : int
            The index of the stream to get the window (default = 0).
        
        Returns
        --------
        list
            The window data [samples][channels]
        list
            timestamps [samples]
        """
        self.is_connected
        winsize = self.streams[stream_index].buffer.winsize
        
        try:
            window = self.streams[stream_index].buffer.data[-winsize:]
            timestamps = self.streams[stream_index].buffer.timestamps[-winsize:]
        except IndexError:
            logger.warning("The buffer of {} does not contain enough samples" .format(self._streams[stream_index].name))
            
        
        return window, timestamps    
    #----------------------------------------------------------------------
    def _acquire(self, blocking=True, stream_index=0, timestamp_offset = False):
        """
        Function called by acquire() in thread.
        
        Parameters
        ----------
        blocking : bool
            If True, the stream waits to receive some data. Othewise, return an empty list.
        stream_index : int
            The index of the stream to acquire the data.
        """        
        if len(self.streams[stream_index].buffer.timestamps) == 0:
            timestamp_offset = True

        data, tslist, lsl_clock = self._streams[stream_index].acquire(blocking, timestamp_offset)
        self.streams[stream_index].buffer.fill(data, tslist, lsl_clock)
                
        if lsl_clock is None:
            pass
        elif abs(self.streams[stream_index].buffer.lsl_time_offset) > 0.1:
            logger.warning('LSL server {}({}) has a high timestamp offset.'.format(self.streams[stream_index].name, self._streams[stream_index].serial))
        else:
            logger.info('LSL time server {}({}) synchronized [offset={:.3f}]'.format(self.streams[stream_index].name, self._streams[stream_index].serial, \
                                                                                    self.streams[stream_index].buffer.lsl_time_offset))          
    
    #----------------------------------------------------------------------
    def _wait_threads_to_finish(self, threads):
        """
        Wait that all the threads finish.
        
        Parameters
        ----------
        Threads : list
            List of all active threads
        """
        while len(threads) > 0:                
            for t in threads:
                if not t.isAlive():
                    t.handled = True
                else:
                    t.handled = False
                
            threads = [t for t in threads if not t.handled]          
    
    #----------------------------------------------------------------------
    @property
    def is_connected(self):
        """
        Check the connections status and automatically connect if not connected.
        """
        while not self._is_connected:
            logger.error('No LSL servers connected yet. Trying to connect automatically.')
            self.connect()
            time.sleep(1)
        
        self._is_connected = True
        
        #logger.info("Connected to the LSL servers: ")
        #for s in self.streams:
            #logger.info("- "+s.name)
            
        return self._is_connected
    
    #----------------------------------------------------------------------
    @is_connected.setter
    def is_connected(self, is_it):
        logger.warning("This attribute cannot be modified.")
    
    #----------------------------------------------------------------------
    @property
    def streams(self):
        """
        The list of connected streams. They can be eeg or markers type.
        """
        return self._streams
    
    #----------------------------------------------------------------------
    @streams.setter
    def streams(self, new_streams):
        logger.warning("The connected streams cannot be modified directly.")
    
    
"""
Example code for printing out raw values
"""
if __name__ == '__main__':
    import mne
    import os
    from neurodecode.utils.q_common import Timer 
    
    stream_index = 0                            # Stream of interest
    CH_INDEX = [1]                              # Channel of interest
    TIME_INDEX = None                           # integer or None. None = average of raw values of the current window
    SHOW_PSD = False
    
    mne.set_log_level('ERROR')
    os.environ['OMP_NUM_THREADS'] = '1'         # actually improves performance for multitaper
    
    # connect to LSL server
    sr = StreamReceiver(window_size=0.5, buffer_size=1, amp_name=None, eeg_only=False)
    
    
    stream_index = int(input("Provide the stream index of the stream you want to acquire \n>> "))
    
    sfreq = sr.streams[stream_index].sample_rate
    trg_ch = 0                                  # trg channels is always at index 0
    
    # PSD init
    if SHOW_PSD:
        psde = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=1, fmax=50, bandwidth=None, \
            adaptive=False, low_bias=True, n_jobs=1, normalization='length', verbose=None)

    tm = qc.Timer(autoreset=True)
    last_ts = 0
    
    tm_classify = Timer(autoreset=True)
    
    while True:
        
        
        sr.acquire(blocking=True, timestamp_offset=False)
        window, tslist = sr.get_window(stream_index=stream_index)       # window = [samples x channels]
        window = window.T                                               # channels x samples

        # print event values
        tsnew = np.where(np.array(tslist) > last_ts)[0]
        if len(tsnew) == 0:
            logger.warning('There seems to be delay in receiving data.')
            time.sleep(1)
            continue
        trigger = np.unique(window[trg_ch, tsnew[0]:])

        if TIME_INDEX is None:
            datatxt = qc.list2string(np.mean(window[CH_INDEX, :], axis=1), '%-15.6f')
            print('[%.3f : %.3f]' % (tslist[0], tslist[-1]) + ' data: %s' % datatxt)
        else:
            datatxt = qc.list2string(window[CH_INDEX, TIME_INDEX], '%-15.6f')
            print('[%.3f]' % tslist[TIME_INDEX] + ' data: %s' % datatxt)

        # show PSD
        if SHOW_PSD:
            psd = psde.transform(window.reshape((1, window.shape[0], window.shape[1])))
            psd = psd.reshape((psd.shape[1], psd.shape[2]))
            psdmean = np.mean(psd, axis=1)
            for p in psdmean:
                print('%.1f' % p, end=' ')

        last_ts = tslist[-1]
        tm.sleep_atleast(0.05)
