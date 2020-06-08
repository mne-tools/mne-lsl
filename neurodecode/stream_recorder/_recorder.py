import os
import time
import datetime

import neurodecode.utils.q_common as qc
from neurodecode.utils.convert2fif import pcl2fif
from neurodecode.utils.cnbi_lsl import start_server
from neurodecode.stream_receiver import StreamReceiver

class _Recorder:
    """
    Base class for recording signals coming from an lsl stream.
    """
    #----------------------------------------------------------------------
    def __init__(self, amp_name, amp_serial, record_dir, eeg_only, logger, state):
        
        self.sr = None
        self.state = state
        self.logger = logger
        self.pcl_file, self.eve_file = self.create_files(record_dir)

        self.connect(amp_name, amp_serial, eeg_only)
        
        self.outlet = self.create_events_server(source_id=self.eve_file)
        
    #----------------------------------------------------------------------
    def connect(self, amp_name, amp_serial, eeg_only):
        """
        Instance a StreamReceiver connecting to the appropriate lsl stream.
        
        Parameters
        ----------
        amp_name : str
                Connect to a server named 'amp_name'. None: no constraint.
        amp_serial : str
            Connect to a server with serial number 'amp_serial'. None: no constraint.
        eeg_only : bool
            If true, ignore non-EEG servers.
        """   
        self.sr = StreamReceiver(buffer_size=0, amp_name=amp_name, amp_serial=amp_serial, eeg_only=eeg_only)
            
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
        pcl_file, eve_file = self.create_filenames(record_dir)
        self.test_writability(record_dir, pcl_file)
        self.logger.info('>> Record to file: %s' % (pcl_file))
        
        return pcl_file, eve_file
        
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
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        pcl_file = "%s/%s-raw.pcl" % (record_dir, timestamp)
        eve_file = '%s/%s-eve.txt' % (record_dir, timestamp)
        
        return pcl_file, eve_file    

    #----------------------------------------------------------------------
    def test_writability(self, record_dir, pcl_file):
        """
        Test the file's writability.
        
        Parameters
        ----------
        record_dir : str
            The directory where the file will be created
        pcl_file : str
            The pickle file to create and write to.
        """
        try:
            qc.make_dirs(record_dir)
            open(pcl_file, 'w').write('The data will written when the recording is finished.')
        except:
            raise RuntimeError('Problem writing to %s. Check permission.' % pcl_file)    
    
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
    def get_buffer(self):
        """
        Get all the data contained in the buffer.
        
        Returns
        -------
        np.array
            The data [[samples_ch1],[samples_ch2]...]
        np.array
            Its timestamps [samples]
        """
        data, timestamps = self.sr.get_buffer()
        
        return data, timestamps
    
    #----------------------------------------------------------------------
    def save_to_file(self):
        """
        Save the acquired date to pickle and also fif formats.
        """
        data = self.create_dict_to_save()
        
        self.logger.info('Saving raw data ...')
        qc.save_obj(self.pcl_file, data)
        self.logger.info('Saved to %s\n' % self.pcl_file)
        
        if os.path.exists(eve_file):
            self.logger.info('Found matching event file, adding events.')
        else:
            eve_file = None
        
        self.logger.info('Converting raw file into fif.')
        pcl2fif(self.pcl_file, external_event=self.eve_file)
        
    #----------------------------------------------------------------------
    def create_dict_to_save(self):
        """
        Create a dictionnary containing the signals, its timestamps, the sampling rate,
        the channels' names and the lsl offset.
        """
        signals, timestamps = self.get_buffer()
        
        data = {'signals':signals, 'timestamps':timestamps, 'events':None,
                'sample_rate':self.sr.get_sample_rate(), 'channels':self.sr.get_num_channels(),
                'ch_names':self.sr.get_channel_names(), 'lsl_time_offset':self.sr.buffers[-1].lsl_time_offset}
        
        return data
    
    #----------------------------------------------------------------------
    def record(self):
        """
        Start the recording and save to files the data in pickle and fif format.
        """
        self.logger.info('\n>> Recording started (PID %d).' % os.getpid())
        self.acquire()
        
        self.logger.info('>> Stop requested. Copying buffer')
        self.save_to_file()
    
    #----------------------------------------------------------------------
    def acquire(self):
        """
        Acquire the data from the connected lsl stream.
        """
        with self.state.get_lock():
            self.state.value = 1
            
        tm = qc.Timer(autoreset=True)
        next_sec = 1
        
        while self.state.value == 1:
            self.sr.acquire()
            
            if self.sr.get_buflen() > next_sec:
                duration = str(datetime.timedelta(seconds=int(self.sr.get_buflen())))
                self.logger.info('RECORDING %s' % duration)
                next_sec += 1
            tm.sleep_atleast(0.001)    
        
