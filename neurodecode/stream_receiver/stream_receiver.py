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
        To extract the latest seconds of the buffer [secs].
    buffer_size : float
        1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently [secs].
    amp_name : str
        Connect to a specific stream. None: no constraint.
    eeg_only : bool
        If true, ignore non-EEG servers.
    """
    #----------------------------------------------------------------------
    def __init__(self, window_size=1, buffer_size=1, amp_name=None, eeg_only=False):
        
        self._streams = dict()
        self._is_connected = False
        
        self.connect(window_size, buffer_size, amp_name, eeg_only)
    
    #----------------------------------------------------------------------
    def connect(self, window_size=1, buffer_size=1, amp_name=None, eeg_only=False):
        """
        Search for the available streams on the LSL network and connect to the appropriate ones.
        If a LSL stream fullfills the requirements (name...), a connection is established.
        
        This function is called while instanciating a StreamReceiver and can be recall to reconnect
        to the LSL streams.
        
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
        self._streams = dict()
        
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
                    if eeg_only and si.type().lower() != 'eeg':
                        continue
                    # connect to a specific amp only?
                    if amp_name is not None and si.name() not in amp_name:
                        continue
                    # do not connect to StreamRecorderInfo
                    if si.name() == 'StreamRecorderInfo':
                        continue
                    # eeg stream
                    if si.type().lower() == "eeg":
                        self._streams[si.name()] = StreamEEG(si, buffer_size, window_size)
                    # marker stream
                    elif si.nominal_srate() == 0:
                        self._streams[si.name()] = StreamMarker(si, buffer_size, window_size)
                    
                    server_found = True
            time.sleep(1)
                
        self.acquire()
        
        self.show_info()
        self._is_connected = True
        logger.info('Ready to receive data from the connected streams.')
    
    
    #----------------------------------------------------------------------
    def show_info(self):
        """
        Display the informations about the connected streams.
        """
        for s in self.streams:
            logger.info("----------------------------------------------------------------")
            logger.info("The stream {} is connected to:".format(s))
            self.streams[s].show_info()
            
    #----------------------------------------------------------------------
    def acquire(self):
        """
        Read data from the streams and fill their buffer using threading.
        
        It will wait that the threads finish.
        """
        threads = []
        
        for s in self._streams:
            t = Thread(target=self._streams[s].acquire, args=[])
            t.daemon = True
            t.start()
            threads.append(t)
        
        self._wait_threads_to_finish(threads)
             
    #----------------------------------------------------------------------
    def get_window(self, stream_name=None):
        """
        Get the latest window from a stream's buffer.
        
        If several streams connected, specify the name.
        
        Parameters
        ----------
        stream_name : int
            The name of the stream to extract from.

        Returns
        -------
        np.array
             The data [samples x channels]
        np.array
             The timestamps [samples]
        """
        if len(list(self.streams)) == 1:
            stream_name = list(self.streams)[0]
        elif stream_name is None:
            raise IOError("Please provide a stream name to get its latest window.")
            
        self.is_connected
        winsize = self.streams[stream_name].buffer.winsize
        
        try:
            window = self.streams[stream_name].buffer.data[-winsize:]
            timestamps = self.streams[stream_name].buffer.timestamps[-winsize:]
        except IndexError:
            logger.warning("The buffer of {} does not contain enough samples" .format(self._streams[stream_name].name))

        if len(timestamps) > 0:
            return np.array(window), np.array(timestamps)
        else:
            return np.empty((0, len(self.streams[stream_name].ch_list))), np.array([])
    
    #----------------------------------------------------------------------
    def get_buffer(self, stream_name=None):
        """
        Get the entire buffer of a stream in numpy format.
        
        If several streams connected, specify the name.
        
        Parameters
        ----------
        stream_name : int
            The name of the stream to extract from.

        Returns
        -------
        np.array
            The data [samples x channels]
        np.array
            Its timestamps [samples]
        """
        if len(list(self.streams)) == 1:
            stream_name = list(self.streams)[0]
        elif stream_name is None:
            raise IOError("Please provide a stream name to get its buffer.")
        
        self.is_connected
        
        if len(self.streams[stream_name].buffer.timestamps) > 0:
            return np.array(self.streams[stream_name].buffer.data), np.array(self.streams[stream_name].buffer.timestamps)
        else:
            return np.array([]), np.array([])
    
    #----------------------------------------------------------------------
    def reset_buffer(self, stream_name):
        """
        Clear the stream's buffer.
        
        If several streams connected, specify the name.
                
        Parameters
        ----------
        stream_name : int
            The stream's name.
        """
        if len(list(self.streams)) == 1:
            stream_name = list(self.streams)[0]
        elif stream_name is None:
            raise IOError("Please provide a stream name to reset its buffer.")
        
        self.streams[stream_name].buffer.reset_buffer()
        
    #----------------------------------------------------------------------
    def reset_all_buffers(self):
        """
        Clear all the streams' buffer.
        """
        for i in self._streams:
            self.streams[i].buffer.reset_buffer()
                        
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
                if not t.is_alive():
                    t.handled = True
                else:
                    t.handled = False
                
            threads = [t for t in threads if not t.handled]          
    
    #----------------------------------------------------------------------
    @property
    def is_connected(self):
        """
        Check the connection status and automatically connect if not connected.
        """
        while not self._is_connected:
            logger.error('No LSL servers connected yet. Trying to connect automatically.')
            self.connect()
            time.sleep(1)
            
        return self._is_connected
    
    #----------------------------------------------------------------------
    @is_connected.setter
    def is_connected(self, is_it):
        logger.warning("This attribute cannot be modified.")
    
    #----------------------------------------------------------------------
    @property
    def streams(self):
        """
        The connected streams list.
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
    
    CH_INDEX = [1]                              # Channel of interest
    TIME_INDEX = None                           # integer or None. None = average of raw values of the current window
    SHOW_PSD = True
    
    mne.set_log_level('ERROR')
    os.environ['OMP_NUM_THREADS'] = '1'         # actually improves performance for multitaper
    
    # connect to LSL server
    sr = StreamReceiver(window_size=0.5, buffer_size=1, amp_name=None, eeg_only=False)
    
    
    # stream_name = input("Provide the name of the stream you want to acquire \n>> ")
    stream_name = None
    sfreq = sr.streams[list(sr.streams.keys())[0]].sample_rate
    trg_ch = 0                                  # trg channels is always at index 0
    
    # PSD init
    if SHOW_PSD:
        psde = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=1, fmax=50, bandwidth=None, \
            adaptive=False, low_bias=True, n_jobs=1, normalization='length', verbose=None)

    tm = qc.Timer(autoreset=True)
    last_ts = 0
    
    tm_classify = Timer(autoreset=True)
    
    while True:
        sr.acquire()
        window, tslist = sr.get_window(stream_name=stream_name)       # window = [samples x channels]
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
            print('[%.3f ]' % (tslist[0]-tslist[-1]) + ' data: %s' % datatxt)
        else:
            datatxt = qc.list2string(window[CH_INDEX, TIME_INDEX], '%-15.6f')
            print('[%.3f]' % tslist[TIME_INDEX] + ' data: %s' % datatxt)

        # show PSD
        if SHOW_PSD:
            psd = psde.transform(window.reshape((1, window.shape[0], window.shape[1])))
            psd = psd.reshape((psd.shape[1], psd.shape[2]))
            psdmean = np.mean(psd, axis=1)
            for p in psdmean:
                print('{:.2e}'.format(p), end=' ')
            print("")

        last_ts = tslist[-1]
        tm.sleep_atleast(0.05)
