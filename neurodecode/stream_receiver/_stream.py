import math
import time
import pylsl
import numpy as np

import neurodecode.utils.q_common as qc
from abc import ABC, abstractmethod
from neurodecode import logger
from neurodecode.utils.pycnbi_utils import find_event_channel, lsl_channel_list


class _Stream(ABC):
    """
    Abstract class representing a receiver's stream.
    
    Parameters
    ----------
    
    """
    #----------------------------------------------------------------------
    @abstractmethod
    def __init__(self, streamInfo):
        """
        Constructor
        """
        
        self._streamInfo = streamInfo
        self._inlet = None
        self._ch_list = None        
        self._inlet_bufsize = None
        
        self.watchdog = qc.Timer()
    
    #----------------------------------------------------------------------
    def show_stream_info(self):
        """
        Display  the stream's info.
        """
        logger.info('Found streaming server: %s (type %s) @ %s.' % (self.streamInfo.name(), self.streamInfo.type(), self.streamInfo.hostname()))
        logger.info('LSL Protocol version: %s' % self.streamInfo.version())
        logger.info('Source sampling rate: %.1f' % self.streamInfo.nominal_srate())
        logger.info('Channels: %d' % self.streamInfo.channel_count())
        # logger.info('Unit multiplier: %.1f' % self.multiplier)
    
    #----------------------------------------------------------------------
    def extract_amp_info(self):
        """
        Extract the name and serial number and if it is slave.
        """
        self.name = self.streamInfo.name()
        self.serial = self._inlet.info().desc().child('acquisition').child_value('serial_number')
        
        if self.serial == '':
            self.serial = 'N/A'
            
        self.is_slave= ('true'==pylsl.StreamInlet(self.streamInfo).info().desc().child('amplifier').child('settings').child('is_slave').first_child().value())
        
    #----------------------------------------------------------------------
    def acquire(self, blocking=True, timestamp_offset=False):
        """
        Pull data from the stream's inlet.
        
        Parameters
        -----------
        blocking : bool
            True if the stream is blocking (wait until data is received).
        timestamp_offset : bool
            True if wrong timestamps and require to compute offset.
        
        Returns
        --------
        list
            data [samples x channels]
        list
            timestamps [samples]
        """
        self.watchdog.reset()
        tslist = []
        received = False
        chunk = None
        lsl_clock = None
        
        while not received:
            while self.watchdog.sec() < 5:    
                # chunk = [frames]x[ch], tslist = [frames] 
                if len(tslist) == 0:
                    chunk, tslist = self._inlet.pull_chunk(max_samples=self._inlet_bufsize)
                    if blocking == False and len(tslist) == 0:
                        return np.empty((0, len(self.ch_list))), [], None
                if len(tslist) > 0:
                    if timestamp_offset is True:
                        lsl_clock = pylsl.local_clock()
                    received = True
                    break
                time.sleep(0.0005)
            
            else:
                logger.warning('Timeout occurred while acquiring data from {}({}). Amp driver bug?'.format(self.name, self.serial))
                # give up and return empty values to avoid deadlock
                return np.empty((0, len(self.ch_list))), [], None
        
        data = np.array(chunk)
        
        return data, tslist, lsl_clock
    
    #----------------------------------------------------------------------
    def get_num_channels(self):
        """
        Get the total number of channels including the trigger channel.

        Returns
        -------
        int
            The number of channels 
        """
        return len(self.ch_list)
    
    #----------------------------------------------------------------------
    @property
    def name(self):
        """
        The LSL stream's name.
        """
        return self.streamInfo.name()

    #----------------------------------------------------------------------
    @property
    def streamInfo(self):
        """
        The stream info received from the LSL inlet.
        """
        return self._streamInfo
    
    #----------------------------------------------------------------------
    @property
    def ch_list(self):
        """
        The channels' list from the LSL inlet
        """
        return self._ch_list
        
    #----------------------------------------------------------------------
    @property
    def inlet(self):
        """
        The LSL inlet created to acquire from a stream
        """
        return self._inlet
        
        
#----------------------------------------------------------------------       
class _StreamMarker(_Stream):
    """
    Class representing a receiver's markers stream.
    
    Parameters
    -----------
    streamInfo : lsl streamInfo
        Contain all the info from the connected lsl inlet.
    bufsec : secs
        The buffer's size in seconds.
    """
    #----------------------------------------------------------------------
    def __init__(self, streamInfo, bufsec):
        """Constructor"""
        
        super().__init__(streamInfo)
        
        inlet_bufsec = int(round(bufsec * 100))     #  due to irregular sr, defined to 100x.
        self._inlet_bufsize = inlet_bufsec
        self._inlet = pylsl.StreamInlet(streamInfo, inlet_bufsec)
        
        self._ch_list = ["Markers"]
        self.extract_amp_info()
        self.show_stream_info()
   
    
#----------------------------------------------------------------------
class _StreamEEG(_Stream):
    """
    Class representing a receiver's eeg stream.
    
    Parameters
    -----------
    streamInfo : lsl streamInfo
        Contain all the info from the connected lsl inlet.
    bufsec : secs
        The buffer's size in seconds.
        
    Attributes
    ----------
    inlet : lsl inlet
        The lsl inlet the stream connect to
    streamInfo : lsl streamInfo
        All availbale info from the inlet.
    name : str
        The name of the amplifier connected to the inlet.
    serial : str
        The serial number of the amplifier.
    is_slave : bool
        True if the amplifier is the slave (more than one amplifier).
    multiplier : float
        Unit conversion.
    ch_list : list
        List of the channels.
    """
    #----------------------------------------------------------------------
    def __init__(self, streamInfo, bufsec):
        
        super().__init__(streamInfo)
    
        self._lsl_tr_channel = None
        self._lsl_eeg_channels = None
        
        inlet_bufsec = bufsec
        self._inlet_bufsize = int(round(inlet_bufsec * streamInfo.nominal_srate())) 
        self.inlet = pylsl.StreamInlet(streamInfo, inlet_bufsec)
        
        self.multiplier = 1  # 10**6 for uV unit (automatically updated for openvibe servers)
        
        self.find_trig_channel()
        self.define_ch_indices()
        self.create_ch_name_list()
        self.show_stream_info()
        
    #----------------------------------------------------------------------     
    def find_trig_channel(self):
        """
        Look for the trigger channel index at the lsl inlet level.
        """
        self.ch_list = lsl_channel_list(self.inlet)
        
        if 'USBamp' in self.name:
            self._lsl_tr_channel = 16
        
        elif 'BioSemi' in self.name:
            self._lsl_tr_channel = 0  # or subtract -6684927? (value when trigger==0)
        
        elif 'SmartBCI' in self.name:
            self._lsl_tr_channel = 23
        
        elif 'StreamPlayer' in self.name:
            self._lsl_tr_channel = 0
        
        elif 'openvibeSignal' in self.name:
            self.multiplier = 10**6 # change V -> uV unit for OpenVibe sources
            self._lsl_tr_channel = find_event_channel(ch_names=self.ch_list)
        
        elif 'openvibeMarkers' in self.name:
            self._lsl_tr_channel = find_event_channel(ch_names=self.ch_list)
        
        else:
            self._lsl_tr_channel = find_event_channel(ch_names=self.ch_list)
    
    #----------------------------------------------------------------------      
    def define_ch_indices(self):
        """
        Define trigger and EEG channel indices at lsl inlet level.
        
        Trigger channel will always move to the first position.
        """
        channels = self.streamInfo.channel_count()
        self._lsl_eeg_channels = list(range(channels))
        if self._lsl_tr_channel is None:
            logger.warning('Trigger channel not found. Adding an empty channel at index 0.')
        else:
            if self._lsl_tr_channel != 0:
                logger.info_yellow('Trigger channel found at index %d. Moving to index 0.' % self._lsl_tr_channel)
            self._lsl_eeg_channels.pop(self._lsl_tr_channel)
        
        self._lsl_eeg_channels = np.array(self._lsl_eeg_channels)
        # self.tr_channel = 0                         # trigger channel is always set to 0.
        # self.eeg_channels = np.arange(1, channels)  # signal channels start from 1.
    
    #----------------------------------------------------------------------   
    def create_ch_name_list(self):
        """
        Create the channel info.
        """        
        if self._lsl_tr_channel is None:
            self.ch_list = ['TRIGGER'] + self.ch_list
        else:
            for i, chn in enumerate(self.ch_list):
                if chn == 'TRIGGER' or chn == 'TRG' or 'STI ' in chn:
                    self.ch_list.pop(i)
                    self.ch_list = ['TRIGGER'] + self.ch_list
                    break
        logger.info('Channels list %s(%s): ' % (self.name, self.serial))
        logger.info(self.ch_list)
    
    #----------------------------------------------------------------------   
    def acquire(self, blocking=True, timestamp_offset=False):
        """
        Pull data from the stream's inlet.
        
        Parameters
        -----------
        blocking : bool
            True if the stream is blocking (wait until data is received).
        timestamp_offset : bool
            True if wrong timestamps and require to compute offset.
        
        Returns
        --------
        list
            data [samples x channels]
        list
            timestamps [samples]
        """
        data, tslist, lsl_clock = super().acquire()
        
        # BioSemi has pull-up resistor instead of pull-down
        if 'BioSemi' in self.streamInfo.name()and self._lsl_tr_channel is not None:
            datatype = data.dtype
            data[:, self._lsl_tr_channel] = (np.bitwise_and(255, data[:, self._lsl_tr_channel].astype(int)) - 1).astype(datatype)
        
        # multiply values (to change unit)
        if self.multiplier != 1:
            data[:, self._lsl_eeg_channels] *= self.multiplier
    
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
    def get_eeg_channels(self):
        """
        Get the eeg channels indexes.

        Returns
        -------
        list
            Channels' index list 
        """
        return np.arange(1, self.get_num_channels())
    
    #----------------------------------------------------------------------
    def get_trigger_channel(self):
        """
        Get trigger channel index (0-based index).
        
        Returns
        -------
        int
            The trigger channel index 
        """
        tr_channel = 0
        
        return tr_channel
    