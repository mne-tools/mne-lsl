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
    Class for playing recorded file on LSL network.
    
    Parameters
    ----------
    server_name : str
        LSL server name.
    fif_file : str
        fif file to replay.
    chunk_size : int
        number of samples to send at once (usually 16-32 is good enough).
    auto_restart : bool
        play from beginning again after reaching the end.
    wait_start : bool
        wait for user to start in the beginning.
    repeat : np.float
        number of loops to play.
    high_resolution :bool
        use perf_counter() instead of sleep() for higher time resolution
        but uses much more cpu due to polling.
    trigger_file : str
        used to convert event numbers into event strings for readability.
        
    Attributes
    ----------
    raw : mne.io.RawArray
        The raw data to stream on LSL network.
    events : np.array
        mne-compatible events (N x [frame, 0, type]).
    server_name : str
        The name of the stream on the LSL network.
    chunk_size :
         The size of a chunk of data.
    
    Note: Run neurodecode.set_log_level('DEBUG') to print out the relative time stamps since started.
    """
    #----------------------------------------------------------------------
    def __init__(self, server_name, fif_file, chunk_size, trigger_file=None):
        
        self.raw = None
        self.events = None
        self.server_name = server_name
        self.chunk_size = chunk_size

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
        
        repeat : int
            The number of loops to play.
        high_resolution : bool
            Use perf_counter() instead of sleep() for higher time resolution
            but uses much more cpu due to polling.
        auto_restart : bool
            Replay from the beginning after reaching the end.
        """
        logger.info('Streaming started')
        
        idx_chunk = 0
        t_chunk = chunk_size / self.get_sample_rate()
        finished = False
        
        if high_resolution:
            t_start = time.perf_counter()
        else:
            t_start = time.time()
    
        # start streaming
        played = 1
        while played < repeat:
        
            idx_current = idx_chunk * chunk_size
            chunk = self.raw._data[:, idx_current:idx_current + chunk_size]
            data = chunk.transpose().tolist()
        
            if idx_current >= self.raw._data.shape[1] - chunk_size:
                finished = True
            
            print("idx: {}, tstart: {}, t_chunk: {}" .format(idx_chunk, t_start, t_chunk))
            self._sleep(high_resolution, idx_chunk, t_start, t_chunk)
            
            self._outlet.push_chunk(data)
            logger.debug('[%8.3fs] sent %d samples (LSL %8.3f)' % (time.perf_counter(), len(data), pylsl.local_clock()))
            
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
        
        return finished
    
    #----------------------------------------------------------------------
    def start(self, repeat=np.float('inf'), high_resolution=False, auto_restart=False):
        """
        Start streaming data on LSL network in a new process
        
        repeat : float
            The number of chunks to stream.
        high_resolution : bool
            use perf_counter() instead of sleep() for higher time resolution
            but uses much more cpu due to polling.
        """
        self.thread = Process(target=self.stream, args=(repeat, high_resolution))
        self.thread.start()
    
    #----------------------------------------------------------------------   
    def stop(self):
        """
        Stop the streaming of the data on LSL network when launched with start().
        """
        if self._thread:
            self._thread.terminate()
        
    #----------------------------------------------------------------------
    def set_lsl_info(self, server_name):
        """
        Set the lsl server infos.
        
        Parameters
        ----------
        server_name : str
            The name to display on LSL network.
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
            The fif file containing the raw data to play.
        """
        self.raw, self.events = pu.load_raw(fif_file)
                
        if self.raw is not None:
            logger.info_green('Successfully loaded %s' % fif_file)        
        else:
            raise RuntimeError('Error while loading %s' % fif_file)
    
    #----------------------------------------------------------------------  
    def get_sample_rate(self):
        """
        Get the sample rate 
        """
        return self.raw.info['sfreq']
        
    #----------------------------------------------------------------------
    def get_nb_ch(self):
        """
        Get the number of channels.
        """
        return len(self.raw.ch_names)
    
    #----------------------------------------------------------------------   
    def get_trg_index(self):
        """
        Return the index of the trigger channel.
        """
        try:
            event_ch = self.raw.ch_names.index('TRIGGER')
        except ValueError:
            event_ch = None
        return event_ch
    
    #----------------------------------------------------------------------   
    def get_info(self):
        """
        Log the info about the created LSL stream 
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
        Sleep for
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
        
        Parameters
        ----------
        chunk : mne.io.RawArray
            Chunk of data to stream on LSL network
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
    sp.stream(repeat=np.float('inf'), high_resolution=False, auto_restart=False)
    # stream_player(server_name, fif_file, chunk_size)
