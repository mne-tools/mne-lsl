import os
import time
import datetime
from mne.io import read_raw_fif
from mne_bids import make_bids_basename, write_raw_bids

import neurodecode.utils.q_common as qc
from neurodecode.utils.convert2fif import pcl2fif, add_events_from_txt
from neurodecode.utils.cnbi_lsl import start_server
from neurodecode.stream_receiver import StreamReceiver
from neurodecode.triggers.trigger_def import trigger_def

class _Recorder:
    """
    Base class for recording signals coming from an lsl stream.
    """
    #----------------------------------------------------------------------
    def __init__(self, record_dir, bids_info, logger, state):
        self._MAX_BUFSIZE = 86400 
        
        self.record_dir = record_dir
        self.bids_info = bids_info
        
        self.sr = None
        self.nb_amps = 0
        self.amp_names = None
        self.state = state
        self.logger = logger
        
    #----------------------------------------------------------------------
    def connect(self, amp_name, eeg_only):
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
        self.extract_connected_amp_names()
    
    #----------------------------------------------------------------------
    def extract_connected_amp_names(self):
        """
        Extract from the stream receiver the names of the connected amplifier.
        
        Returns
        -------
        list
            The connected amplifiers' name.
        """
        self.nb_amps = len(self.sr._buffers)
        self.amp_names = []
        
        for s in self.sr._streams:
            self.amp_names.append(s.name) 
            
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
        
        self.logger.info('>> Record to file: %s', *data_files, sep='\n')
        
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
        timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        eve_file = '%s/%s-eve.txt' % (record_dir, timestamp)
        
        data_files = []
        
        for i in range(len(self.sr._buffers)):
            data_files.append("%s/%s-%s-raw.pcl" % (record_dir, timestamp, self.sr._streams[i].name))
        
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
        qc.make_dirs(record_dir)

        for i in range(self.nb_amps):
            try:
                open(pcl_files[i], 'w').write('The data will written when the recording is finished.')
            except:
                raise RuntimeError('Problem writing to %s. Check permission.' % pcl_files[i])
    
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
    def get_buffer(self, stream_index=0):
        """
        Get all the data collected in the buffer for a stream.
        
        Parameters
        ----------
        stream_index : int
            The index of the stream to get the data.
            
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
    def save_to_bids(self, data_files, eve_file):
        """
        Save the data to BIDS structure using brainvision .eeg files.
        """
        self.logger.info('Saving data to BIDS...')
        event_id = trigger_def(self.bids_info["trigger_file"])
        
        for i in range(len(data_files)):
            
            fif_file = os.path.splitext(data_files[i])[0] + ".fif"
            raw = read_raw_fif(fif_file)
            bids_file_name = make_bids_basename(subject=self.bids_info["subj_idx"], \
                                               session=self.bids_info["session"], \
                                               task=self.bids_info["task"], \
                                               run=self.bids_info["run_idx"], \
                                               recording=self.amp_names[i])             
            
            if os.path.exists(eve_file):
                add_events_from_txt(raw, eve_file)
                        
            write_raw_bids(raw, bids_file_name, self.record_dir, event_id=event_id.by_name, \
                        events_data=None, overwrite=True)        
        
    #----------------------------------------------------------------------
    def save_to_file(self, pcl_files, eve_file):
        """
        Save the acquired data to pcl and fif format.
        """
        self.logger.info('Saving raw data ...')
        
        for i in range(self.nb_amps):
            
            signals, timestamps = self.get_buffer(i)
            data = self.create_dict_to_save(signals, timestamps, i)
            
            qc.save_obj(pcl_files[i], data)
            self.logger.info('Saved to %s\n' % pcl_files[i])
            self.logger.info('Converting raw files into fif.')
            
            if os.path.exists(eve_file):
                self.logger.info('Found matching event file, adding events.')
                pcl2fif(pcl_files[i], external_event=eve_file)
            else:
                pcl2fif(pcl_files[i], external_event=None)
            
            os.remove(pcl_files[i])
 
    #----------------------------------------------------------------------
    def save(self, data_files, eve_file):
        """
        Save the acquired data in .fif or bids format.
        
        Save to bids only if the bids info were provided.
        """
        self.save_to_file(data_files, eve_file)
        
        # if self.bids_info:
        #    self.save_to_bids(data_files, eve_file)
                    
    #----------------------------------------------------------------------
    def create_dict_to_save(self, signals, timestamps, stream_index):
        """
        Create a dictionnary containing the signals, its timestamps, the sampling rate,
        the channels' names and the lsl offset for each acquired streams.
        """
        data = {'signals':signals,
                'timestamps':timestamps,
                'events':None,
                'sample_rate': self.sr.get_sample_rate(stream_index),
                'channels':self.sr.get_num_channels(stream_index),
                'ch_names':self.sr.get_channel_names(stream_index),
                'lsl_time_offset':self.sr.get_lsl_offset(stream_index)}
        
        return data
    
    #----------------------------------------------------------------------
    def record(self):
        """
        Start the recording and save to files the data in pickle and fif format.
        """
        data_files, eve_file = self.create_files(self.record_dir)
        
        # self.outlet = self.create_events_server(source_id=eve_file)
        self.logger.info('\n>> Recording started (PID %d).' % os.getpid())
        self.acquire()
        
        self.logger.info('>> Stop requested. Copying buffer')
        self.save(data_files, eve_file)
        
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
            