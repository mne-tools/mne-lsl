import time
import pylsl
import numpy as np

import neurodecode.utils.pycnbi_utils as pu
from neurodecode import logger
from neurodecode.stream_receiver._buffer import _Buffer
from neurodecode.stream_receiver._stream import _StreamEEG, _StreamMarker

import neurodecode.utils.q_common as qc
from threading import Thread

class StreamReceiver:
    """
    Class for data acquisition from lsl streams.
        
    Parameters
    ----------
    window_size : secs
        To extract the latest window_size seconds of the buffer.
    buffer_size : secs
        1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently.
    amp_name : str
        Connect to a server named 'amp_name'. None: no constraint.
    eeg_only : bool
        If true, ignore non-EEG servers.
    """    
    #----------------------------------------------------------------------
    def __init__(self, window_size=1, buffer_size=1, amp_name=None, eeg_only=False):
        
        self._buffers = []
        self._streams = []
        self._ready = False
        
        self.connect(window_size, buffer_size, amp_name, eeg_only)
    
    #----------------------------------------------------------------------
    def connect(self, window_size=1, buffer_size=1, amp_name=None, eeg_only=False):
        """
        Search for the available streams on the lsl network and connect to the appropriate ones.
        
        If a lsl stream fullfills the requirements (name, serial...), a connection is done.
        
        Parameters
        ----------
        window_size : secs
            To extract the latest window_size seconds of the buffer.
        buffer_size : secs
            1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently.
        amp_name : str
            Connect to a server named 'amp_name'. None: no constraint.
        eeg_only : bool
            If true, ignore non-EEG servers.
        """
        self._buffers = []
        self._streams = []
        
        self._connected = False
        self._ready = False
        
        server_found = False
                
        while server_found == False:
            
            if amp_name is None:
                logger.info("Looking for a streaming server...")
            else:
                logger.info("Looking for %s ..." % (amp_name))
                
            streamInfos = pylsl.resolve_streams()
            
            if len(streamInfos) > 0:
                # For now, only 1 amp is supported by a single StreamReceiver object.
                for si in streamInfos:
                    
                    # EEG streaming server only?
                    if eeg_only and si.type() != 'EEG':
                        continue
                    # connect to a specific amp only?
                    if amp_name is not None and si.name() not in amp_name:
                        continue
                    
                    if si.type() == "eeg":
                        self._streams.append(_StreamEEG(si, buffer_size))
                    elif si.type() == "marker":
                        self._streams.append(_StreamMarker(si, buffer_size))
                    self._buffers.append(_Buffer(si.nominal_srate(), buffer_size, window_size))        
                    server_found = True
            time.sleep(1)
            
        self._connected = True

        # pre-fill in initial buffers
        self.pre_acquire()
        self._ready = True
        logger.info('Start receiving stream data.')
    
    #----------------------------------------------------------------------
    def _acquire(self, blocking=True, stream_index=0):
        """
        Read data from one stream and fill its buffer.
        
        It is a blocking function as default.
        
        Parameters
        ----------
        blocking : bool
            If True, the stream waits to receive some data. Othewise, return an empty list.
        stream_index : int
            The index of the stream to acquire the data.
        """
        timestamp_offset = False
        
        if len(self._buffers[stream_index].timestamps) == 0:
            logger.info('Acquisition from: %s (%s)' % (self._streams[stream_index].amp_name, self._streams[stream_index].amp_serial))
            timestamp_offset = True

        data, tslist, lsl_clock = self._streams[stream_index].acquire(blocking, timestamp_offset)
        self._buffers[stream_index].fill(data, tslist, lsl_clock)
            
    #----------------------------------------------------------------------
    def acquire(self, blocking=True):
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
            t = Thread(target=self._acquire, args=[blocking, i])
            t.daemon = True
            t.start()
        threads.append(t)
        
        if blocking is True:
            self._wait_threads_to_finish(threads)
            
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
    def _pre_acquire(self, stream_index=0):
        """
        Prefill the buffer with at least winsize elements.
        
        It is a blocking function as default.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to acquire the data.
        """
        logger.info('Waiting to fill initial buffer for stream {}({})'.format(self._streams[stream_index].amp_name, self._streams[stream_index].amp_serial))
        
        while len(self._buffers[stream_index].timestamps) < self._buffers[stream_index].winsize:
            self._acquire(blocking=True, stream_index=stream_index)
    
    #----------------------------------------------------------------------
    def pre_acquire(self):
        """
        Prefill all the buffers with at least winsize elements.
        
        It is a blocking function.
        """        
        for i in range(len(self._streams)):
            t = Thread(target=self._pre_acquire, args=[i])
            t.daemon = True
            t.start()
        
    #----------------------------------------------------------------------
    def check_connect(self):
        """
        Check the connections status and automatically connect if not connected.
        """
        while not self._connected:
            logger.error('LSL server not _connected yet. Trying to connect automatically.')
            self.connect()
            time.sleep(1)
    
    #----------------------------------------------------------------------
    def set_window_size(self, window_size, stream_index=0):
        """
        Set the window size for a buffer.
        
        Parameters
        -----------
        window_size : secs
            The new window's size.
        stream_index : int
            The index of the stream to modify its buffer parameter (default = 0).
        """
        self.check_connect()
        window_size = self._buffers[stream_index].check_window_size(window_size)
        
        self._buffers[stream_index].winsize = window_size
        self._buffers[stream_index].winsec = self._streams[stream_index].convert_sec_to_samples(window_size)
            
    #----------------------------------------------------------------------
    def get_channel_names(self, stream_index=0):
        """
        Get the channels' list of a stream.
        
        Parameters
        -----------
        stream_index : int
            The index of the stream to get the channels (default = 0).
        """
        return self._streams[stream_index].ch_list
    
    #----------------------------------------------------------------------
    def get_window_list(self, stream_index=0):
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
        self.check_connect()
        winsize = self._buffers[stream_index].winsize
        
        window = self._buffers[stream_index].data[-winsize:]
        timestamps = self._buffers[stream_index].timestamps[-winsize:]
        
        return window, timestamps    
    #----------------------------------------------------------------------
    def get_window(self, stream_index=0):
        """
        Get the latest window and timestamps  of a stream in numpy format.

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
        self.check_connect()
        window, timestamps = self.get_window_list()

        if len(timestamps) > 0:
            return np.array(window), np.array(timestamps)
        else:
            return np.array([]), np.array([])    
    #----------------------------------------------------------------------
    def get_buffer_list(self, stream_index=0):
        """
        Get the entire buffer of a stream.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer.
        
        Returns
        --------
        list
            The buffer data [samples x channels]
        list
            Its timestamps [samples]
        """
        self.check_connect()
        return self._buffers[stream_index].data, self._buffers[stream_index].timestamps    
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
        self.check_connect()
        
        if len(self._buffers[stream_index].timestamps) > 0:
            return np.array(self._buffers[stream_index].data), np.array(self._buffers[stream_index].timestamps)
        else:
            return np.array([]), np.array([])
    
    #----------------------------------------------------------------------
    def get_buflen(self, stream_index=0):
        """
        Return buffer length in seconds.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer's length(default = 0).

        Returns
        -------
        int
            Buffer's length
        """
        return self._buffers[stream_index].get_buflen()
    #----------------------------------------------------------------------
    def get_sample_rate(self, stream_index=0):
        """
        Get the sampling rate of a stream.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer (default = 0).

        Returns
        -------
        int
            The sampling rate
        """
        return self._buffers[stream_index].get_sample_rate()
    #----------------------------------------------------------------------
    def get_num_channels(self, stream_index=0):
        """
        Get the total number of channels including the trigger channel.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer (default = 0).

        Returns
        -------
        int
            The number of channels
        """
        return self._streams[stream_index].get_num_channels()
    #----------------------------------------------------------------------
    def get_eeg_channels(self, stream_index=0):
        """
        Get the eeg channels list.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the eeg channels (default = 0).

        Returns
        -------
        list
            The channels list
        """
        return self._streams[stream_index].get_eeg_channels()    
    
    #----------------------------------------------------------------------
    def get_trigger_channel(self, stream_index=0):
        """
        Get trigger channel index (0-based index).
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer (default = 0).
            
        Returns
        -------
        int
            The trigger channel index
        """        
        return self._streams[stream_index].get_trigger_channel()
    
    #----------------------------------------------------------------------
    def get_lsl_offset(self, stream_index=0):
        """
        Get the time difference between the acquisition server's time and LSL time.

        OpenVibe servers often have a bug of sending its own running time instead of LSL time.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer (default = 0).
        
        Returns
        -------
        float
            The lsl offset
        """
        return self._buffers[stream_index].get_lsl_offset()
    
    #----------------------------------------------------------------------
    def reset_buffer(self, stream_index=0):
        """
        Clear the buffer of a stream.
        
        After call pre-acquire() to fill the buffer with at least winsize elements.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the buffer (default = 0).
        """
        self._buffers[stream_index].reset_buffer()
        
    #----------------------------------------------------------------------
    def reset_all_buffers(self):
        """
        Clear the buffer of all the streams.
        
        Call pre-acquire() after to fill the buffers with at least winsize elements.
        """
        for i in range(len(self._streams)):
            self._buffers[i].reset_buffer()
            
    #----------------------------------------------------------------------
    def is_ready(self):
        """
        Get the receiver's status. True when buffers are prefilled.
        
        Returns
        -------
        bool
            True if the buffer is not empty.
        """
        return self._ready
        
    
"""
Example code for printing out raw values
"""
if __name__ == '__main__':
    import mne
    import os

    CH_INDEX = [1] # channel to monitor
    TIME_INDEX = None # integer or None. None = average of raw values of the current window
    SHOW_PSD = False
    mne.set_log_level('ERROR')
    os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

    # connect to LSL server
    amp_name, amp_serial = pu.search_lsl()
    sr = StreamReceiver(window_size=1, buffer_size=1, amp_name=amp_name, eeg_only=False)
    sfreq = sr.get_sample_rate()
    trg_ch = sr.get_trigger_channel()
    logger.info('Trigger channel = %d' % trg_ch)

    # PSD init
    if SHOW_PSD:
        psde = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=1, fmax=50, bandwidth=None, \
            adaptive=False, low_bias=True, n_jobs=1, normalization='length', verbose=None)

    watchdog = qc.Timer()
    tm = qc.Timer(autoreset=True)
    last_ts = 0
    while True:
        sr.acquire()
        window, tslist = sr.get_window() # window = [samples x channels]
        window = window.T # chanel x samples

        qc.print_c('LSL Diff = %.3f' % (pylsl.local_clock() - tslist[-1]), 'G')

        # print event values
        tsnew = np.where(np.array(tslist) > last_ts)[0]
        if len(tsnew) == 0:
            logger.warning('There seems to be delay in receiving data.')
            time.sleep(1)
            continue
        trigger = np.unique(window[trg_ch, tsnew[0]:])

        # for Biosemi
        # if sr.amp_name=='BioSemi':
        #    trigger= set( [255 & int(x-1) for x in trigger ] )

        if len(trigger) > 0:
            logger.info('Triggers: %s' % np.array(trigger))

        logger.info('[%.1f] Receiving data...' % watchdog.sec())

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
