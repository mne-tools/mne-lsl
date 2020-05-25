import time
import pylsl
import numpy as np

import neurodecode.utils.pycnbi_utils as pu
from neurodecode import logger
from neurodecode.stream_receiver._buffer import _Buffer
from neurodecode.stream_receiver._stream import _Stream

import neurodecode.utils.q_common as qc

class StreamReceiver:
    """
    Facade class for data acquisition from lsl streams.
        
    Parameters
    ----------
    window_size : secs
        To extract the latest window_size seconds of the buffer.
    buffer_size : secs
        1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently.
    amp_name : str
        Connect to a server named 'amp_name'. None: no constraint.
    amp_serial : str
        Connect to a server with serial number 'amp_serial'. None: no constraint.
    eeg_only : bool
        If true, ignore non-EEG servers.
    find_any : bool
        If True, look for any kind of streams. If False, look only for USBamp, BioSemi, SmartBCI, openvibeSignal, openvibeMarkers, StreamPlayer.
    
    Attributes
    ----------
    streams : list
        List of all the streams the receiver is connected to.
    buffers : list
        List containing the buffers associated to each connected streams.
    ready : bool
        The receiver's status, True when the buffers have been prefilled.
             
    """    
    #----------------------------------------------------------------------
    def __init__(self, window_size=1, buffer_size=1, amp_serial=None, eeg_only=False, amp_name=None, find_any=True):

        self.connect(window_size, buffer_size, amp_name, amp_serial, eeg_only, find_any)
    
    #----------------------------------------------------------------------
    def connect(self, window_size=1, buffer_size=1, amp_name=None, amp_serial=None, eeg_only=False, find_any=True):
        """
        Search for the available streams on the lsl network and connect to the appropriate ones.
        
        If a lsl stream fullfills the requirements (name, serial...), it is saved in streams list and its buffer is created.
        
        Parameters
        ----------
        window_size : secs
            To extract the latest window_size seconds of the buffer.
        buffer_size : secs
            1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently.
        amp_name : str
            Connect to a server named 'amp_name'. None: no constraint.
        amp_serial : str
            Connect to a server with serial number 'amp_serial'. None: no constraint.
        eeg_only : bool
            If true, ignore non-EEG servers.
        find_any : bool
            If True, look for any kind of streams. If False, look only for "USBamp", "BioSemi", "SmartBCI", "openvibeSignal", "openvibeMarkers", "StreamPlayer".
        """

        self._connected = False
        self.ready = False        
        self.buffers = []
        self.streams = []
        
        server_found = False
        
        while server_found == False:
            
            if amp_name is None and amp_serial is None:
                logger.info("Looking for a streaming server...")
            else:
                logger.info("Looking for %s (Serial %s) ..." % (amp_name, amp_serial))
                
            streamInfos = pylsl.resolve_streams()
            
            if len(streamInfos) > 0:
                # For now, only 1 amp is supported by a single StreamReceiver object.
                for si in streamInfos:
                    s = _Stream(si, buffer_size)
                    # connect to a specific amp only?
                    if amp_serial is not None and amp_serial != s.amp_serial:
                        continue
                    # connect to a specific amp only?
                    if amp_name is not None and amp_name != s.amp_name:
                        continue
                    # EEG streaming server only?
                    if eeg_only and s.type != 'EEG':
                        continue
                    self.streams.append(s)
                    self.buffers.append(_Buffer(si.nominal_srate(), buffer_size, window_size))        
                    server_found = True
            time.sleep(1)
            
        self._connected = True

        # pre-fill in initial buffer
        self.pre_acquire()
        self.ready = True
        logger.info('Start receiving stream data.')
    
    #----------------------------------------------------------------------
    def acquire(self, blocking=True, stream_index=0):
        """
        Read data from one stream and fill its buffer.
        
        It is a blocking function as default.
        
        Parameters
        ----------
        blocking : bool
            If True, the stream waits to receive some data. Othewise, return an empty list.
        stream_index : int
            The index of the stream to acquire the data.

        Returns
        --------
        list
             The acquired data [samples x channels]
        list
             Its timestamps [samples]
        """
        timestamp_offset = False
        
        if len(self.buffers[stream_index].timestamps) == 0:
            logger.info('Acquisition from: %s (%s)' % (self.streams[stream_index].amp_name, self.streams[stream_index].amp_serial))
            timestamp_offset = True

        data, tslist, lsl_clock = self.streams[stream_index].acquire(blocking, timestamp_offset)
        self.buffers[stream_index].fill(data, tslist, lsl_clock)
        
        return (data, tslist)
    
    #----------------------------------------------------------------------
    def acquire_all_streams(self, blocking=True):
        """
        Read data from all the streams and fill their buffer.
        
        It is a blocking function as default.
          
        Parameters
        ----------
        blocking : bool
            If True, the streams wait to receive some data.
        """
        for i in range(len(self.streams)):
            self.acquire(blocking=True, stream_index=i)
    
    #----------------------------------------------------------------------
    def pre_acquire(self, blocking=True, stream_index=0):
        """
        Prefill the buffer with at least winsize elements.
        
        It is a blocking function as default.
        
        Parameters
        ----------
        blocking : bool
            If True, the stream wait to receive some data.
        """
        logger.info('Waiting to fill initial buffers')
        
        while len(self.buffers[stream_index].timestamps) < self.buffers[stream_index].winsize:
            self.acquire(blocking=True, stream_index=stream_index)
    
    #----------------------------------------------------------------------
    def pre_acquire_all(self, blocking=True):
        """
        Prefill all the buffers with at least winsize elements.
        
        It is a blocking function as default.
        
        Parameters
        ----------
        blocking : bool
            If True, the streams wait to receive some data.
        """
        logger.info('Waiting to fill initial buffers')
        
        for i in range(len(self.streams)):
            self.pre_acquire(i)
    
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
        Set the window size for a buffer of stream_index.
        
        Parameters
        -----------
        window_size : secs
            The new window's size.
        stream_index : int
            The index of the stream to modify its buffer parameter (default = 0).
        """
        self.check_connect()
        window_size = self.buffers[stream_index].check_window_size(window_size)
        
        self.buffers[stream_index].winsize = window_size
        self.buffers[stream_index].winsec = self.streams[stream_index].convert_sec_to_samples(window_size)
            
    #----------------------------------------------------------------------
    def get_channel_names(self, stream_index=0):
        """
        Get the channels' list of a stream.
        
        Parameters
        -----------
        stream_index : int
            The index of the stream to get the channels (default = 0).
        """
        return self.streams[stream_index].ch_list
    
    #----------------------------------------------------------------------
    def get_window_list(self, stream_index=0):
        """
        Get the latest window from the buffer of a stream.
        
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
        winsize = self.buffers[stream_index].winsize
        
        window = self.buffers[stream_index].data[-winsize:]
        timestamps = self.buffers[stream_index].timestamps[-winsize:]
        
        return window, timestamps    
    #----------------------------------------------------------------------
    def get_window(self, decim=1, stream_index=0):
        """
        Get the latest window and timestamps  of a stream in numpy format.

        Parameters
        ----------
        decim : int
            Decimation factor for unit conversion.
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
        return self.buffers[stream_index].data, self.buffers[stream_index].timestamps    
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
        
        if len(self.buffers[stream_index].timestamps) > 0:
            return np.array(self.buffers[stream_index].data), np.array(self.buffers[stream_index].timestamps)
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
        return self.buffers[stream_index].get_buflen()
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
        return self.buffers[stream_index].get_sample_rate()
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
        return self.streams[stream_index].get_num_channels
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
        return self.streams[stream_index].ch_list[1:]    
    
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
        return self.streams[stream_index].get_trigger_channel()
    
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
        return self.buffers[stream_index].get_lsl_offset()
    
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
        self.buffers[stream_index].reset_buffer()
        
    #----------------------------------------------------------------------
    def reset_all_buffers(self):
        """
        Clear the buffer of all the streams.
        
        After call pre-acquire() to fill the buffers with at least winsize elements.
        """
        for i in range(len(self.streams)):
            self.buffers[i].reset_buffer()
            
    #----------------------------------------------------------------------
    def is_ready(self):
        """
        Get the receiver's status.
        
        Returns
        -------
        bool
            True if the buffer is not empty.
        """
        return self.ready    

"""
Example code for printing out raw values
"""
def test_receiver():
    import mne
    import os

    CH_INDEX = [1] # channel to monitor
    TIME_INDEX = None # integer or None. None = average of raw values of the current window
    SHOW_PSD = False
    mne.set_log_level('ERROR')
    os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

    # connect to LSL server
    amp_name, amp_serial = pu.search_lsl()
    sr = StreamReceiver(window_size=1, buffer_size=1, amp_name=amp_name, amp_serial=amp_serial, eeg_only=False)
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

if __name__ == '__main__':
    test_receiver()
