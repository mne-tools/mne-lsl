import os
import time
import datetime
from pathlib import Path
import multiprocessing as mp

from neurodecode.utils.timer import Timer
from neurodecode.utils.lsl import start_server
from neurodecode.utils.io import save_obj, pcl2fif, make_dirs
from neurodecode.stream_receiver import StreamReceiver, StreamEEG

class _Recorder:
    """
    Base class for recording signals coming from an lsl stream.
    """
    #----------------------------------------------------------------------
    def __init__(self, record_dir, logger, state=mp.Value('i', 0)):
        self._MAX_BUFSIZE = 86400 
        
        self.record_dir = record_dir
        
        self.sr = None
        self.state = state
        self.logger = logger
        
    #----------------------------------------------------------------------
    def connect(self, amp_name=None, eeg_only=False):
        """
        Instance a StreamReceiver connecting to the appropriate lsl stream.
        
        Parameters
        ----------
        amp_name : str
                Connect to a server named 'amp_name'. None: no constraint.
        eeg_only : bool
            If true, ignore non-EEG servers.
        """   
        self.sr = StreamReceiver(buffer_size=self._MAX_BUFSIZE, amp_name=amp_name, eeg_only=eeg_only)
            
    #----------------------------------------------------------------------
    def create_files(self, record_dir):
        """
        Create the files to save the data.
        
        Parameters
        ----------
        record_dir : str
            The directory where to create the recording file.
            
        Returns
        -------
        str
            The data file's name (pickle format)
        str
            The software events' file (txt format)
        """
        data_files, eve_file = self.create_filenames(record_dir)
        self.test_writability(record_dir, data_files)
        
        self.logger.info('Record to files:')
        print(*data_files.values(), sep='\n')
        
        return data_files, eve_file
        
    #----------------------------------------------------------------------
    def create_filenames(self, record_dir):
        """
        Create the filenames to save the data and events
        
        Parameters
        ----------
        record_dir : str
            The directory where to save the data
            
        Returns
        -------
        str
            The data file's name (pickle format)
        str
            The software events' file (txt format)
        """
        if record_dir == '.':
            record_dir = os.getcwd()
        
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())        
        eve_file = Path('%s/%s-eve.txt' % (record_dir, timestamp))

        data_files = dict()
        
        for s in self.sr.streams:
            data_files[s] = Path("%s/%s-%s-raw.pcl" % (record_dir, timestamp, s))
        
        return data_files, eve_file    

    #----------------------------------------------------------------------
    def test_writability(self, record_dir, pcl_files):
        """
        Test the file's writability.
        
        Parameters
        ----------
        record_dir : str
            The directory where the file will be created
        pcl_files : str
            The pickle files to create to write to.
        """
        make_dirs(record_dir)

        for pcl_f in pcl_files:
            try:
                open(pcl_files[pcl_f], 'w').write('The data will written when the recording is finished.')
            except:
                raise RuntimeError('Problem writing to %s. Check permission.' % pcl_f)
    
    #----------------------------------------------------------------------
    def create_events_server(self, name='StreamRecorderInfo', source_id=None):
        """
        Start a lsl server for sending out events when software trigger is used.
        
        Parameters
        ----------
        name : str
            The server's name displayed on the network.
        source_id : str
            The software events' file (txt format).
        """
        return start_server(name, channel_format='string', source_id=source_id, stype='Markers')
    
    #----------------------------------------------------------------------
    def get_buffer(self, stream_name=None):
        """
        Get all the data collected in the buffer for a stream.
        
        Parameters
        ----------
        stream_name : str
            The name of the stream to get the data.
            
        Returns
        -------
        np.array
            The data [[samples_ch1],[samples_ch2]...]
        np.array
            Its timestamps [samples]
        """
        
        data, timestamps = self.sr.get_buffer(stream_name)
        
        return data, timestamps
            
    #----------------------------------------------------------------------
    def save_to_file(self, pcl_files, eve_file):
        """
        Save the acquired data to pcl and fif format.
        """
        self.logger.info('Saving raw data ...')
        
        for s in self.sr.streams:
            
            signals, timestamps = self.get_buffer(s)
            
            if isinstance(self.sr.streams[s], StreamEEG):
                signals[:, 1:] *= 1E-6
                    
            data = self.create_dict_to_save(signals, timestamps, s)
            
            save_obj(pcl_files[s], data)
            self.logger.info('Saved to %s\n' % pcl_files[s])
            self.logger.info('Converting raw files into fif.')
            
            if os.path.exists(eve_file):
                self.logger.info('Found matching event file, adding events.')
                pcl2fif(pcl_files[s], external_event=eve_file)
            else:
                pcl2fif(pcl_files[s], external_event=None)
                
    #----------------------------------------------------------------------
    def create_dict_to_save(self, signals, timestamps, stream_name):
        """
        Create a dictionnary containing the signals, its timestamps, the sampling rate,
        the channels' names and the lsl offset for each acquired streams.
        
        Parameters
        ----------
        signal: np.array
            The data to save.
        timestamps: np.array
            The associated timestamps
        stream_name : str
            The name of the stream to extract from.

        Returns
        -------
        np.array
            The data [[samples_ch1],[samples_ch2]...]
        np.array
            Its timestamps [samples]
        """
        data = {'signals':signals,
                'timestamps':timestamps,
                'events':None,
                'sample_rate': self.sr.streams[stream_name].sample_rate,
                'channels':len(self.sr.streams[stream_name].ch_list),
                'ch_names':self.sr.streams[stream_name].ch_list,
                'lsl_time_offset':self.sr.streams[stream_name].lsl_time_offset}
        
        return data
    
    #----------------------------------------------------------------------
    def record(self, verbose=False):
        """
        Start the recording and save to files the data in pickle and fif format.
        """
        data_files, eve_file = self.create_files(self.record_dir)
        
        self.outlet = self.create_events_server(source_id=str(eve_file))
        self.logger.info('>> Recording started (PID %d).' % os.getpid())
        self.acquire(verbose)
        
        self.logger.info('>> Stop requested. Copying buffer')
        self.save_to_file(data_files, eve_file)
        
    #----------------------------------------------------------------------
    def acquire(self, verbose=False):
        """
        Acquire the data from the connected lsl stream.
        """
        with self.state.get_lock():
            self.state.value = 1
            
        tm = Timer(autoreset=True)
        
        #  Extract name of the first recorded stream
        first_stream = list(self.sr.streams.keys())[0]
        
        while self.state.value == 1:
            self.sr.acquire()
            
            bufsec = len(self.sr.streams[first_stream].buffer.data) / self.sr.streams[first_stream].sample_rate
            
            if verbose:
                duration = str(datetime.timedelta(seconds=int(bufsec)))
                self.logger.info('RECORDING %s' % duration)
            
            tm.sleep_atleast(1)    
            