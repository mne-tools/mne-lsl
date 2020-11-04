import time
import pylsl
import numpy as np

import neurodecode.utils.q_common as qc
from abc import ABC, abstractmethod
from neurodecode import logger
from neurodecode.stream_receiver import Buffer
from neurodecode.utils.pycnbi_utils import find_event_channel, lsl_channel_list


class _Stream(ABC):
    """
    Abstract class representing a base receiver's stream.
    
    Parameters
    ----------
    streamInfo : lsl streamInfo
        Contain all the info from the lsl stream to connect to.
    bufsize : int
        The buffer's size [secs]
    winsize : int
        To extract the latest winsize samples from the buffer [secs]
    """
    #----------------------------------------------------------------------
    @abstractmethod
    def __init__(self, streamInfo, bufsize, winsize):
        
        self._ch_list = []
        self._streamInfo = streamInfo
        self._watchdog = qc.Timer()
        self._sample_rate = streamInfo.nominal_srate()
        
        self._blocking = True
        self._blocking_time = 5
        self._lsl_time_offset = None
        
        self._inlet = pylsl.StreamInlet(streamInfo, max_buflen=bufsize)
        
        winsize = _Stream._check_window_size(winsize)
        bufsize = _Stream._check_buffer_size(bufsize, winsize)
    
        if self._sample_rate:
            samples_per_sec = self.sample_rate
        else:
            samples_per_sec = 100
            
        bufsize = _Stream._convert_sec_to_samples(bufsize, samples_per_sec)
        winsize = _Stream._convert_sec_to_samples(winsize, samples_per_sec)
        
        self._buffer = Buffer(bufsize, winsize)
        
        self._extract_amp_info()
        self._create_ch_name_list()
    
    #----------------------------------------------------------------------
    def show_info(self):
        """
        Display the stream's info.
        """
        logger.info('Server: {name}({serial}) / type:{type} @ {host} (v{version}).'.format(name=self.name, serial=self.serial, type=self.streamInfo.type(), host=self.streamInfo.hostname(), version=self.streamInfo.version()))
        logger.info('Source sampling rate: {}'.format(self.streamInfo.nominal_srate()))
        logger.info('Channels: {}'.format(self.streamInfo.channel_count()))
        logger.info('{}'.format(self._ch_list))
        
        # Check for high lsl offset
        if self.lsl_time_offset is None:
            logger.warning('No LSL timestamp offset computed, no data received yet.')
        elif abs(self.lsl_time_offset) > 0.1:
            logger.warning('LSL server {}({}) has a high timestamp offset [offset={:.3f}].'.format(self.name, self.serial, self.buffer.lsl_time_offset))
        else:
            logger.info('LSL server {}({}) synchronized [offset={:.3f}]'.format(self.name, self.serial, self.lsl_time_offset))           
    
    #----------------------------------------------------------------------
    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.
        
        Returns
        --------
        list
            data [samples x channels]
        list
            timestamps [samples]
        """
        chunk = []
        tslist = []
        received = False
        self._watchdog.reset()
        
        # If first data acquisition, compute lsl offset
        if len(self.buffer.timestamps) == 0:
            timestamp_offset = True
        else:
            timestamp_offset = False
        
        # Acquire the data
        while not received:
            while self._watchdog.sec() < self._blocking_time:    
                if len(tslist) == 0:
                    chunk, tslist = self._inlet.pull_chunk(max_samples=self.buffer._bufsize)    # chunk = [frames]x[ch], tslist = [frames]
                    if self._blocking == False and len(tslist) == 0:
                        received = True
                        break
                if len(tslist) > 0:  
                    if timestamp_offset is True:
                        self._compute_offset(tslist)
                    received = True                  
                    tslist = self._correct_lsl_offset(tslist)
                    break
                time.sleep(0.0005)
            else:
                # give up and return empty values to avoid deadlock
                logger.warning('Timeout occurred [{}secs] while acquiring data from {}({}). Amp driver bug?'.format(self._blocking_time, self.name, self.serial))

        return chunk, tslist
    
    #----------------------------------------------------------------------
    def _correct_lsl_offset(self, timestamps):
        """
        Correct the timestamps if there is a high lsl offset.
        
        Parameters
        -----------
        timestamps : list
            The timestamps from the last 
        """
        if abs(self._lsl_time_offset) > 0.1:
            timestamps = [t - self._lsl_time_offset for t in timestamps]
        
        return timestamps
    
    #----------------------------------------------------------------------
    def _compute_offset(self, timestamps):
        """
        Compute the LSL offset coming from some devices.
        
        It has to be called just after acquiring the data/timestamps
        in order to be valid.
        
        Parameters
        -----------
        timestamps : list
            The last acquired timestamps.
        """
        self._lsl_time_offset = timestamps[-1] - pylsl.local_clock()
        
        return self._lsl_time_offset
        
    #----------------------------------------------------------------------
    def _extract_amp_info(self):
        
        """
        Extract the name and serial number and if it is slave.
        """
        self._name = self.streamInfo.name()
        self._serial = self._inlet.info().desc().child('acquisition').child_value('serial_number')
        
        if self._serial == '':
            self._serial = 'N/A'
            
        self.is_slave= ('true'==pylsl.StreamInlet(self.streamInfo).info().desc().child('amplifier').child('settings').child('is_slave').first_child().value())
    
    #----------------------------------------------------------------------   
    def _create_ch_name_list(self):
        """
        Create the channels' name list.
        """
        self._ch_list = lsl_channel_list(self._inlet)
        
        if not self._ch_list:
            self._ch_list = ["ch_"+str(i+1) for i in range(self.streamInfo.channel_count())]
            
    #----------------------------------------------------------------------
    @staticmethod
    def _check_window_size(window_size):
        """
        Check that buffer's window size is positive.
        
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
    @staticmethod
    def _check_buffer_size(buffer_size, window_size):
        """
        Check that buffer's size is positive and bigger than the window's size.
        
        Parameters
        -----------
        buffer_size : float
            The buffer size to verify [secs].
        window_size : float
            The window's size to compare to buffer_size [secs].
        
        Returns
        --------
        secs
            The verified buffer size.
        """
        MAX_BUF_SIZE=86400       # 24h max buffer length
        
        if buffer_size <= 0 or buffer_size > MAX_BUF_SIZE:
            logger.error('Improper buffer size %.1f. Setting to %d.' % (buffer_size, MAX_BUF_SIZE))
            buffer_size = MAX_BUF_SIZE
        
        elif buffer_size < window_size:
            logger.error('Buffer size %.1f is smaller than window size. Setting to %.1f.' % (buffer_size, window_size))
            buffer_size = window_size

        return buffer_size
    
    #----------------------------------------------------------------------
    @staticmethod
    def _convert_sec_to_samples(bufsec, sample_rate):
        """
        Convert a buffer's size from sec to samples.
        
        Parameters
        -----------
        bufsec : float
            The buffer_size's size [secs].
        sample_rate : float
             The sampling rate of the LSL server. If irregular sampling rate, empirical number of samples per sec.
        Returns
        --------
        samples
            The converted buffer_size's size.
        """
        return int(round(bufsec * sample_rate))
        
    #----------------------------------------------------------------------
    @property
    def name(self):
        """
        The stream's name.
        """
        return self._name

    #----------------------------------------------------------------------
    @name.setter
    def name(self, name):
        logger.warning("The stream's name cannot be changed")
    
    #----------------------------------------------------------------------
    @property
    def serial(self):
        """
        The stream's serial number.
        """
        return self._serial

    #----------------------------------------------------------------------
    @serial.setter
    def serial(self, serial):
        logger.warning("The stream's name cannot be changed")
        
    #----------------------------------------------------------------------
    @property
    def sample_rate(self):
        """
        The stream's sampling rate.
        """
        return self._sample_rate

    #----------------------------------------------------------------------
    @sample_rate.setter
    def sample_rate(self, sr):
        logger.warning("The stream's sampling rate cannot be changed")
    
    #----------------------------------------------------------------------
    @property
    def streamInfo(self):
        """
        The stream info received from the LSL inlet.
        """
        return self._streamInfo
    
    #----------------------------------------------------------------------
    @streamInfo.setter
    def streamInfo(self, si):
        logger.warning("The stream's info cannot be changed")    
    
    #----------------------------------------------------------------------
    @property
    def ch_list(self):
        """
        The channels' name list.
        """
        return self._ch_list
    
    #----------------------------------------------------------------------
    @ch_list.setter
    def ch_list(self, ch_list):
        logger.warning("The channels' names list cannot be changed")
        
    #----------------------------------------------------------------------
    @property
    def buffer(self):
        """
        The buffer containing the data and the timestamps.
        """
        return self._buffer
    
    #----------------------------------------------------------------------
    @buffer.setter
    def buffer(self, buf):
        logger.warning("The buffer cannot be changed")
    #----------------------------------------------------------------------
    @property
    def blocking(self):
        """
        If True, the stream wait to receive data.
        """
        return self._buffer
    
    #----------------------------------------------------------------------
    @blocking.setter
    def blocking(self, block):
        self._blocking = block
    
    #----------------------------------------------------------------------
    @property
    def blocking_time(self):
        """
        If blocking is True, how long to wait to receive data [secs].
        """
        return self._blocking_time 
    
    #----------------------------------------------------------------------
    @blocking_time.setter
    def blocking_time(self, block_time):
        self._blocking_time  = block_time
    
    #----------------------------------------------------------------------
    @property
    def lsl_time_offset(self):
        """
        The difference between the local and the stream's LSL clocks, used for timestamps correction.
        
        Some LSL servers (like OpenVibe) often have a bug of sending its own running time instead of LSL time.
        """
        return self._lsl_time_offset
    
    #----------------------------------------------------------------------
    @lsl_time_offset.setter
    def lsl_time_offset(self, lsl_time_offset):
        logger.warning("This attribute cannot be changed")
    
    
#----------------------------------------------------------------------       
class StreamMarker(_Stream):
    """
    Class representing a receiver's markers stream.
    
    Notice the irregular sampling rate.
    This stream is instanciated as non-blocking. 
    
    Parameters
    -----------
    streamInfo : lsl streamInfo
        Contain all the info from the lsl stream to connect to.
    buffer_size : float
        The buffer's size [secs]. 1-day is the maximum size.
        Large buffer may lead to a delay if not pulled frequently.
    window_size : float
        To extract the latest seconds of the buffer [secs].
    """
    #----------------------------------------------------------------------
    def __init__(self, streamInfo, buffer_size=1, window_size=1):
          
        super().__init__(streamInfo, buffer_size, window_size)
        
        self._blocking = False
        self._blocking_time = np.Inf
    
    #----------------------------------------------------------------------
    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.
        """
        chunk, tslist = super().acquire()
        
        # Fill its buffer
        self.buffer.fill(chunk, tslist)
    
#----------------------------------------------------------------------
class StreamEEG(_Stream):
    """
    Class representing a receiver's eeg stream.
    
    This stream is instanciated as blocking. 
    
    Parameters
    -----------
    streamInfo : lsl streamInfo
        Contain all the info from the lsl server to connect to.
    buffer_size : float
        The buffer's size [secs].
    window_size : float
        Extract the latest window_size seconds of the buffer [secs].
    """
    #----------------------------------------------------------------------
    def __init__(self, streamInfo, buffer_size=1, window_size=1):
                
        super().__init__(streamInfo, buffer_size, window_size)
        
        self._multiplier = 1  # 10**6 for uV unit (automatically updated for openvibe servers)

    #----------------------------------------------------------------------   
    def acquire(self):
        """
        Pull data from the stream's inlet and fill the buffer.
        """
        chunk, tslist = super().acquire()
        
        if not chunk:
            return
        
        data = np.array(chunk)
        
        # BioSemi has pull-up resistor instead of pull-down
        if 'BioSemi' in self.streamInfo.name()and self._lsl_tr_channel is not None:
            datatype = data.dtype
            data[:, self._lsl_tr_channel] = (np.bitwise_and(255, data[:, self._lsl_tr_channel].astype(int)) - 1).astype(datatype)
        
        # multiply values (to change unit)
        if self._multiplier != 1:
            data[:, self._lsl_eeg_channels] *= self._multiplier
    
        if self._lsl_tr_channel is not None:
            # move trigger channel to 0 and add back to the buffer
            data = np.concatenate((data[:, self._lsl_tr_channel].reshape(-1, 1),
                                       data[:, self._lsl_eeg_channels]), axis=1)
        else:
            # add an empty channel with zeros to channel 0
            data = np.concatenate((np.zeros((data.shape[0],1)),
                                       data[:, self._lsl_eeg_channels]), axis=1)
        data = data.tolist()
        
        # Fill its buffer
        self.buffer.fill(data, tslist)

    #----------------------------------------------------------------------     
    def _find_lsl_trig_ch(self):
        """
        Look for the trigger channel index at the lsl inlet level.
        """
        if 'USBamp' in self.name:
            self._lsl_tr_channel = 16
        
        elif 'BioSemi' in self.name:
            self._lsl_tr_channel = 0  # or subtract -6684927? (value when trigger==0)
        
        elif 'SmartBCI' in self.name:
            self._lsl_tr_channel = 23
        
        elif 'StreamPlayer' in self.name:
            self._lsl_tr_channel = 0
        
        elif 'openvibeSignal' in self.name:
            self._multiplier = 10**6 # change V -> uV unit for OpenVibe sources
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)
        
        elif 'openvibeMarkers' in self.name:
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)
        
        else:
            self._lsl_tr_channel = find_event_channel(ch_names=self._ch_list)
    
    #----------------------------------------------------------------------   
    def _create_ch_name_list(self):
        """
        Create the channel info.
        
        Trigger channel will always move to the first position.
        """
        super()._create_ch_name_list()
        
        self._find_lsl_trig_ch()
        
        self._lsl_eeg_channels = list(range(len(self.ch_list)))
        
        if self._lsl_tr_channel is not None:
            self._ch_list.pop(self._lsl_tr_channel)
            self._lsl_eeg_channels.pop(self._lsl_tr_channel)
        
        self._ch_list = ['TRIGGER'] + self._ch_list
    