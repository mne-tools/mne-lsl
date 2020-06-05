from __future__ import print_function, division

"""
Stream Player

Stream signals from a recorded file on LSL network.

For Windows users, make sure to use the provided time resolution
tweak tool to set to 500us time resolution of the OS.

Kyuhwa Lee, 2015

"""
import time
import pylsl
import numpy as np
import neurodecode.utils.q_common as qc
import neurodecode.utils.pycnbi_utils as pu
from neurodecode.triggers.trigger_def import trigger_def
from neurodecode import logger
from builtins import input
from multiprocessing import Process

class StreamPlayer:
    """
    Class for playing recorded file on LSL network in another process.
     
    Parameters
    ----------
    server_name : str
        The stream's name, displayed on LSL network.
    fif_file : str
        The absolute path to the .fif file to play.
    chunk_size : int
        The number of samples to send at once (usually 16-32 is good enough).
    trigger_file : str
        The absolute path to the file containing the table converting event numbers
        into event strings.
        
    Notes
    -----
    It instances a Streamer in a new process and call Streamer.stream().
    """

    def __init__(self, server_name, fif_file, chunk_size, trigger_file=None):

        self._server_name = server_name
        self._fif_file = fif_file
        self._chunk_size = chunk_size
        self._trigger_file = trigger_file
        
        self._process = None
        
    #----------------------------------------------------------------------
    def start(self, repeat=np.float('inf'), high_resolution=False, auto_restart=False):
        """
        Start streaming data on LSL network in a new process by calling stream().
        
        Parameters
        ----------
        repeat : float
            The number of times to replay the data.
        high_resolution : bool
            If True, it uses perf_counter() instead of sleep() for higher time resolution.
            However, it uses much more cpu due to polling.
        auto_restart : bool
            If True, it replays from beginning after reaching the end.
        """
        self._process = Process(target=self._stream, args=(repeat, high_resolution, auto_restart))
        self._process.start()
    
    #----------------------------------------------------------------------
    def wait(self, timeout=None):
        """
        Wait that the data streaming finishes.
        
        Attributes
        ----------
        timeout : float
            Block until timeout is reached. If None, block until streaming is finished.
        
        Notes
        -----
        If auto_restart = True, the streaming will only finish by calling stop()
        """
        self._process.join()
        
    #----------------------------------------------------------------------
    def stop(self):
        """
        Stop the streaming, by terminating the process.
        """
        if self._process:
            logger.info("Stop streaming data from: {}" .format(server_name))
            self._process.terminate()
        
    #----------------------------------------------------------------------
    def _stream(self, repeat, high_resolution, auto_restart):
        """
        The function called in the new process.
        
        Instance a Streamer and start streaming.
        """
        s = Streamer(self.server_name, self.fif_file, self.chunk_size, self.trigger_file)
        s.stream(repeat, high_resolution, auto_restart) 
    
    #----------------------------------------------------------------------
    @property
    def server_name(self):
        """
        The stream's name, displayed on LSL network.
        
        Returns
        -------
        str
        """
        return self._server_name
    
    #----------------------------------------------------------------------
    @property
    def fif_file(self):
        """
        The absolute path to the .fif file to play.
        
        Returns
        -------
        str
        """
        return self._fif_file       
    
    #----------------------------------------------------------------------
    @property
    def chunk_size(self):
        """
        The size of a chunk of data.
        
        Returns
        -------
        int
        """
        return self._chunk_size      

    #----------------------------------------------------------------------
    @property
    def trigger_file(self):
        """
        The absolute path to the file containing the table converting event numbers
        into event strings.
        
        Returns
        -------
        int
        """
        return self._trigger_file
    #----------------------------------------------------------------------
    @property
    def process(self):
        """
        The launched process
        
        Returns
        -------
        multiprocessing.Process
        """
        return self._process
        
class Streamer:
    """
    Class for playing recorded file on LSL network.
    
    Parameters
    ----------
    server_name : str
        The stream's name, displayed on LSL network.
    fif_file : str
        The absolute path to the .fif file to play.
    chunk_size : int
        The number of samples to send at once (usually 16-32 is good enough).
    trigger_file : str
        The absolute path to the file containing the table converting event numbers
        into event strings.
    
    Notes
    -----
    Run neurodecode.set_log_level('DEBUG') to print out the relative time stamps since started.
    """
    #----------------------------------------------------------------------
    def __init__(self, server_name, fif_file, chunk_size, trigger_file=None):
        
        self._raw = None
        self._events = None
        self._server_name = server_name
        self._chunk_size = chunk_size

        self._thread = None
        self._tdef = None
        
        if trigger_file is not None:
            self._tdef = trigger_def(trigger_file)
        
        self.load_data(fif_file)
        sinfo = self.set_lsl_info(server_name)
        self._outlet = pylsl.StreamOutlet(sinfo, chunk_size=chunk_size)
        self.get_info()
    
    #----------------------------------------------------------------------
    def stream(self, repeat, high_resolution, auto_restart):
        """
        Stream data on LSL network.
        
        Parameters
        ----------
        repeat : int
            The number of times to replay the data.
        high_resolution : bool
            If True, it uses perf_counter() instead of sleep() for higher time resolution.
            However, it uses much more cpu due to polling.
        auto_restart : bool
            If True, it replays from beginning after reaching the end.
        """
        logger.info('Streaming started')
        
        idx_chunk = 0
        t_chunk = self.chunk_size / self.get_sample_rate()
        finished = False
        
        if high_resolution:
            t_start = time.perf_counter()
        else:
            t_start = time.time()
    
        # start streaming
        played = 1
        while played < repeat:
        
            idx_current = idx_chunk * self.chunk_size
            chunk = self.raw._data[:, idx_current:idx_current + self.chunk_size]
            data = chunk.transpose().tolist()
        
            if idx_current >= self.raw._data.shape[1] - self.chunk_size:
                finished = True
            
            self._sleep(high_resolution, idx_chunk, t_start, t_chunk)
            
            self._outlet.push_chunk(data)
            logger.debug('[%8.3fs] sent %d samples (LSL %8.3f)' % (time.perf_counter(), len(data), pylsl.local_clock()))
            print('[%8.3fs] sent %d samples (LSL %8.3f)' % (time.perf_counter(), len(data), pylsl.local_clock()))
            
            self._log_event(chunk)
            idx_chunk += 1
            
            if auto_restart is True:
                logger.info('Reached the end of data. Restarting.')
            
            if finished:
                idx_chunk = 0
                finished = False
                if high_resolution:
                    t_start = time.perf_counter()
                else:
                    t_start = time.time()
                played += 1
        
    #----------------------------------------------------------------------
    def set_lsl_info(self, server_name):
        """
        Set the lsl server's infos needed to create the LSL stream.
        
        Parameters
        ----------
        server_name : str
            The stream's name, displayed on LSL network.
        
        Returns
        -------
        pylsl.StreamInfo
            The info to create the stream on LSL network.
        
        """
        sinfo = pylsl.StreamInfo(server_name, channel_count=self.get_nb_ch(), channel_format='float32',\
            nominal_srate=self.get_sample_rate(), type='EEG', source_id=server_name)
        
        desc = sinfo.desc()
        channel_desc = desc.append_child("channels")
        for ch in self.raw.ch_names:
            channel_desc.append_child('channel').append_child_value('label', str(ch))\
                .append_child_value('type','EEG').append_child_value('unit','microvolts')
        
        desc.append_child('amplifier').append_child('settings').append_child_value('is_slave', 'false')
        desc.append_child('acquisition').append_child_value('manufacturer', 'NeuroDecode').append_child_value('serial_number', 'N/A')
        
        return sinfo
                
    #----------------------------------------------------------------------
    def load_data(self, fif_file):
        """
        Load the data to play from a fif file.
        
        Parameters
        ----------
        fif_file : str
            The absolute path to the .fif file to play.
        """
        self._raw, self._events = pu.load_raw(fif_file)
                
        if self.raw is not None:
            logger.info_green('Successfully loaded %s' % fif_file)        
        else:
            raise RuntimeError('Error while loading %s' % fif_file)
    
    #----------------------------------------------------------------------  
    def get_sample_rate(self):
        """
        Get the sample rate
        
        Returns
        -------
        float
            The sampling rate [Hz]
        """
        return self.raw.info['sfreq']
        
    #----------------------------------------------------------------------
    def get_nb_ch(self):
        """
        Get the number of channels.
        
        Returns
        -------
        int
            The number of channels.
        """
        return len(self.raw.ch_names)
    
    #----------------------------------------------------------------------   
    def get_trg_index(self):
        """
        Return the index of the trigger channel.
        
        Returns
        -------
        int
            The trigger channel's index.
        """
        try:
            event_ch = self.raw.ch_names.index('TRIGGER')
        except ValueError:
            event_ch = None
        return event_ch
    
    #----------------------------------------------------------------------   
    def get_info(self):
        """
        Log the info about the created LSL stream.
        """
        logger.info('Server name: %s' % self.server_name)
        logger.info('Sampling frequency %.3f Hz' % self.get_sample_rate())
        logger.info('Number of channels : %d' % self.get_nb_ch())
        logger.info('Chunk size : %d' % self.chunk_size)
        for i, ch in enumerate(self.raw.ch_names):
            logger.info('%d %s' % (i, ch))
        logger.info('Trigger channel : %s' % self.get_trg_index())
        
    #----------------------------------------------------------------------
    def _sleep(self, high_resolution, idx_chunk, t_start, t_chunk):
        """
        Determine the time to sleep to fit simulate the 
        """
        if high_resolution:
            # if a resolution over 2 KHz is needed
            t_sleep_until = t_start + idx_chunk * t_chunk
            while time.perf_counter() < t_sleep_until:
                pass
        else:
            # time.sleep() can have 500 us resolution using the tweak tool provided.
            t_wait = t_start + idx_chunk * t_chunk - time.time()
            print(t_wait)
            if t_wait > 0.001:
                time.sleep(t_wait)

    #----------------------------------------------------------------------
    def _log_event(self, chunk):
        """
        Look for an event on the data chunk and log it.
        """
        event_ch = self.get_trg_index()
        
        if event_ch is not None:
            event_values = set(chunk[event_ch]) - set([0])
        
            if len(event_values) > 0:
                if self._tdef is None:
                    logger.info('Events: %s' % event_values)
                else:
                    for event in event_values:
                        if event in self._tdef.by_value:
                            logger.info('Events: %s (%s)' % (event, self._tdef.by_value[event]))
                        else:
                            logger.info('Events: %s (Undefined event)' % event)    
    
    #----------------------------------------------------------------------
    @property
    def raw(self):
        """
        The raw data to stream on LSL network.
        
        Returns
        -------
        mne.io.RawArray
        """
        return self._raw
    
    #----------------------------------------------------------------------
    @property
    def events(self):
        """
        The mne-compatible events (N x [frame, 0, type]).
        
        Returns
        -------
        np.array
        """
        return self._events
    
    #----------------------------------------------------------------------
    @property
    def server_name(self):
        """
        The stream's name, displayed on LSL network.
        
        Returns
        -------
        str
        """
        return self._server_name
    
    #----------------------------------------------------------------------
    @property
    def chunk_size(self):
        """
         The size of a chunk of data.
        
        Returns
        -------
        int
        """
        return self._chunk_size        

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 5:
        raise RuntimeError("Too many arguments provided, maximum is 4.")
    
    if len(sys.argv) > 4:
        trigger_file = sys.argv[4]
    
    if len(sys.argv) > 3:
        chunk_size = sys.argv[3]
    
    if len(sys.argv) > 2:
        fif_file = sys.argv[2]
    
    if len(sys.argv) > 1:
        server_name = sys.argv[1]
    
    if len(sys.argv) == 1:
        server_name = 'StreamPlayer'
        chunk_size = 16
        fif_file = str(Path(input("Provide the path to the .fif file to play: \n")))
        trigger_file = None
    
    sp = StreamPlayer(server_name, fif_file, chunk_size, trigger_file)
    sp.wait(timeout=10)
    sp.stop()
