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
    Abstract class representing a receiver's stream.
    
    Parameters
    ----------
    streamInfo : lsl streamInfo
        Contain all the info from the lsl server to connect to.
    bufsize : int
        The buffer's size [samples]
    winsize : int
        Extract the latest winsize samples from the buffer.
    """
    #----------------------------------------------------------------------
    @abstractmethod
    def __init__(self, streamInfo, bufsize, winsize):
        
        self._ch_list = []
        self._streamInfo = streamInfo
        self._watchdog = qc.Timer()
        self._sample_rate = streamInfo.nominal_srate()
        
        self._inlet = pylsl.StreamInlet(streamInfo, max_buflen=bufsize)
        self._buffer = Buffer(bufsize, winsize)
        
        self._extract_amp_info()
        self._create_ch_name_list()
    
    #----------------------------------------------------------------------
    def show_info(self):
        """
        Display  the stream's info.
        """
        logger.info('Server: {name}({serial}) / {type} @ {host} (v{version}).'.format(name=self.name, serial=self.serial, type=self.streamInfo.type(), host=self.streamInfo.hostname(), version=self.streamInfo.version()))
        logger.info('Source sampling rate: {}'.format(self.streamInfo.nominal_srate()))
        logger.info('Channels: {}'.format(self.streamInfo.channel_count()))
        logger.info('{}'.format(self._ch_list))
    
    #----------------------------------------------------------------------
    def acquire(self, blocking, timestamp_offset, blocking_time):
        """
        Pull data from the stream's inlet.
        
        Parameters
        -----------
        blocking : bool
            True for a blocking stream.
        timestamp_offset : bool
            True if wrong timestamps and require to compute offset.
        blocking_time : float
            The time allowed to wait for receiving data. [secs]
        
        Returns
        --------
        list
            data [samples x channels]
        list
            timestamps [samples]
        """
        self._watchdog.reset()
        tslist = []
        received = False
        chunk = None
        lsl_clock = None
        
        while not received:
            while self._watchdog.sec() < blocking_time:    
                # chunk = [frames]x[ch], tslist = [frames] 
                if len(tslist) == 0:
                    chunk, tslist = self._inlet.pull_chunk(max_samples=self.buffer._bufsize)
                    if blocking == False and len(tslist) == 0:
                        return np.empty((0, len(self.ch_list))), [], None
                if len(tslist) > 0:
                    if timestamp_offset is True:
                        lsl_clock = pylsl.local_clock()
                    received = True
                    break
                time.sleep(0.0005)
            
            else:
                logger.warning('Timeout occurred [{}secs] while acquiring data from {}({}). Amp driver bug?'.format(blocking_time, self.name, self.serial))
                # give up and return empty values to avoid deadlock
                return np.empty((0, len(self._ch_list))), [], None
        
        data = np.array(chunk)
        
        return data, tslist, lsl_clock
    
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
        Create the channel info.
        """
        self._ch_list = lsl_channel_list(self.inlet)
        
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
    def _check_buffer_size(buffer_size, window_size, max_buffer_size=86400):
        """
        Check that buffer's size is positive, smaller than max size and bigger than window's size.
        
        Parameters
        -----------
        buffer_size : float
            The buffer size to verify [secs].
        window_size : float
            The window's size to compare to buffer_size [secs].
        max_buffer_size : float
            max buffer size allowed by StreamReceiver, default is 24h [secs].
        
        Returns
        --------
        secs
            The verified buffer size.
        """
        if buffer_size <= 0 or buffer_size > max_buffer_size:
            logger.error('Improper buffer size %.1f. Setting to %d.' % (buffer_size, max_buffer_size))
            buffer_size = max_buffer_size
        
        elif buffer_size < window_size:
            logger.error('Buffer size %.1f is smaller than window size. Setting to %.1f.' % (buffer_size, window_size))
            buffer_size = window_size

        return buffer_size
    
    #----------------------------------------------------------------------
    @abstractmethod
    def _convert_sec_to_samples(self):
        """
        Convert a buffer's size from sec to samples.
        """
        pass
        
    #----------------------------------------------------------------------
    @property
    def name(self):
        """
        The LSL stream's name.
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
        The LSL stream's serial number.
        """
        return self._serial

    #----------------------------------------------------------------------
    @serial.setter
    def serial(self):
        logger.warning("The stream's name cannot be changed")
        
    #----------------------------------------------------------------------
    @property
    def sample_rate(self):
        """
        The LSL stream's sampling rate.
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
        The channels' list from the LSL server
        """
        return self._ch_list
    
    #----------------------------------------------------------------------
    @ch_list.setter
    def ch_list(self, ch_list):
        logger.warning("The channels' name list cannot be changed")
        
    #----------------------------------------------------------------------
    @property
    def inlet(self):
        """
        The LSL inlet created to acquire from a stream
        """
        return self._inlet
    
    #----------------------------------------------------------------------
    @inlet.setter
    def inlet(self, inlet):
        logger.warning("The LSL inlet cannot be changed")
    #----------------------------------------------------------------------
    @property
    def buffer(self):
        """
        The buffer containing the data and the timestamps.
        """
        return self._buffer
    
    #----------------------------------------------------------------------
    @buffer.setter
    def buffer(self):
        logger.warning("The buffer cannot be changed")    
    
#----------------------------------------------------------------------       
class StreamMarker(_Stream):
    """
    Class representing a receiver's markers stream.
    
    Notice the irregular sampling rate.
    
    Parameters
    -----------
    streamInfo : lsl streamInfo
        Contain all the info from the lsl server to connect to.
    buffer_size : float
        The buffer's size [secs]. 1-day is the maximum size.
        Large buffer may lead to a delay if not pulled frequently.
    window_size : float
        To extract the latest window_size seconds of the buffer [secs].
    nb_sample_per_sec : int
        Due to the irregular sampling rate, a number of samples per second
        needs to be defined.
    """
    #----------------------------------------------------------------------
    def __init__(self, streamInfo, buffer_size=1, window_size=1, nb_sample_per_sec=100):
        
        _Stream._check_window_size(window_size)
        _Stream._check_buffer_size(buffer_size, window_size)
        
        bufsize = self._convert_sec_to_samples(buffer_size, nb_sample_per_sec)
        winsize = self._convert_sec_to_samples(window_size, nb_sample_per_sec)
        
        super().__init__(streamInfo, bufsize, winsize)
    
    #----------------------------------------------------------------------
    def acquire(self, blocking=False, timestamp_offset=False, blocking_time=np.Inf):
        """
        Pull data from the stream's inlet.
        
        Parameters
        -----------
        blocking : bool
            True if the stream is blocking (wait until data is received).
        timestamp_offset : bool
            True if wrong timestamps and require to compute offset.
        blocking_time : float
            The time allowed to wait for receiving data, if blocking mode. [secs]
            
        Returns
        --------
        list
            data [samples x channels]
        list
            timestamps [samples]
        """
        super().acquire(blocking, timestamp_offset, blocking_time)

    #----------------------------------------------------------------------
    def _convert_sec_to_samples(self, bufsec, nb_samples_per_sec):
        """
        Convert a buffer's size from sec to samples.
        
        Parameters
        -----------
        bufsec : float
            The buffer_size's size [secs].
        nb_samples_per_sec: int
            Empirical number of samples per sec due to irregular sampling rate.
            
        Returns
        --------
        samples
            The converted buffer_size's size.
        """
        return int(round(bufsec * nb_samples_per_sec)) 
    
#----------------------------------------------------------------------
class StreamEEG(_Stream):
    """
    Class representing a receiver's eeg stream.
    
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
        
        _Stream._check_window_size(window_size)
        _Stream._check_buffer_size(buffer_size, window_size)
        
        bufsize = self._convert_sec_to_samples(buffer_size, streamInfo.nominal_srate())
        winsize = self._convert_sec_to_samples(window_size, streamInfo.nominal_srate())
        
        self._buffer = Buffer(bufsize, winsize)
        
        super().__init__(streamInfo, bufsize, winsize)
        
        self._multiplier = 1  # 10**6 for uV unit (automatically updated for openvibe servers)

    #----------------------------------------------------------------------   
    def acquire(self, blocking=True, timestamp_offset=False, blocking_time=5):
        """
        Pull data from the stream's inlet.
        
        Parameters
        -----------
        blocking : bool
            True if the stream is blocking (wait until data is received).
        timestamp_offset : bool
            True if wrong timestamps and require to compute offset.
        blocking_time : float
            The time allowed to wait for receiving data, if blocking mode. [secs]
        
        Returns
        --------
        list
            data [samples x channels]
        list
            timestamps [samples]
        """
        data, tslist, lsl_clock = super().acquire(blocking, timestamp_offset, blocking_time)
        
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
            
        return data, tslist, lsl_clock    

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
        
        if self._lsl_tr_channel is None:
            logger.warning('Trigger channel not found. Adding an empty channel at index 0.')
        else:
            self._ch_list.pop(self._lsl_tr_channel)
            self._lsl_eeg_channels.pop(self._lsl_tr_channel)
            logger.info_yellow('Trigger channel found at index %d. Moving to index 0.' % self._lsl_tr_channel)
        
        self._ch_list = ['TRIGGER'] + self._ch_list
        
    #----------------------------------------------------------------------
    def _convert_sec_to_samples(self, bufsec, sample_rate):
        """
        Convert a buffer's size from sec to samples.
        
        Parameters
        -----------
        bufsec : float
            The buffer_size's size [secs].
        sample_rate : float
             The sampling rate of the LSL server
        Returns
        --------
        samples
            The converted buffer_size's size.
        """
        return int(round(bufsec * sample_rate))
    