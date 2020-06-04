import math
import time
import pylsl
import numpy as np

import neurodecode.utils.q_common as qc
from neurodecode import logger
from neurodecode.utils.pycnbi_utils import find_event_channel, lsl_channel_list

class _Stream:
    """
    Class representing a receiver's stream.
    
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
    amp_name : str
        The name of the amplifier connected to the inlet.
    amp_serial : str
        The serial number of the amplifier.
    type : str
        The data's type (EEG...).
    is_slave : bool
        True if the amplifier is the slave (more than one amplifier).
    multiplier : float
        Unit conversion.
    ch_list : list
        List of the channels.
    """
    #----------------------------------------------------------------------
    def __init__(self, streamInfo, bufsec):
        
        self.streamInfo = streamInfo
        
        _MAX_PYLSL_STREAM_BUFSIZE = 360     # max buffer size for pylsl.StreamInlet
        inlet_bufsec = int(math.ceil(min(_MAX_PYLSL_STREAM_BUFSIZE, bufsec)))
        self._inlet_bufsize = int(round(inlet_bufsec * streamInfo.nominal_srate()))
        self.inlet = pylsl.StreamInlet(streamInfo, inlet_bufsec)
        
        self.amp_name = streamInfo.name()
        self.amp_serial = self.inlet.info().desc().child('acquisition').child_value('serial_number')
        
        if self.amp_serial == '':
            self.amp_serial = 'N/A'
            
        self.type = streamInfo.type()
        self.is_slave= ('true'==pylsl.StreamInlet(streamInfo).info().desc().child('amplifier').child('settings').child('is_slave').first_child().value() )    
        self.multiplier = 1  # 10**6 for uV unit (automatically updated for openvibe servers)
        self.watchdog = qc.Timer()
        
        self.find_trig_channel()
        self.define_ch_indices()
        self.create_ch_name_list()
        self.show_info(streamInfo)

    #----------------------------------------------------------------------     
    def show_info(self, streamInfo):
        """
        Display  the stream's info.
        """
        logger.info('Found streaming server: %s (type %s) @ %s.' % (streamInfo.name(), streamInfo.type(), streamInfo.hostname()))
        logger.info('LSL Protocol version: %s' % streamInfo.version())
        logger.info('Source sampling rate: %.1f' % streamInfo.nominal_srate())
        logger.info('Channels: %d' % streamInfo.channel_count())
        logger.info('Unit multiplier: %.1f' % self.multiplier)
    
    #----------------------------------------------------------------------     
    def find_trig_channel(self):
        """
        Look for the trigger channel index at the lsl inlet level.
        """
        self.ch_list = lsl_channel_list(self.inlet)
        
        if 'USBamp' in self.amp_name:
            self._lsl_tr_channel = 16
        
        elif 'BioSemi' in self.amp_name:
            self._lsl_tr_channel = 0  # or subtract -6684927? (value when trigger==0)
        
        elif 'SmartBCI' in self.amp_name:
            self._lsl_tr_channel = 23
        
        elif 'StreamPlayer' in self.amp_name:
            self._lsl_tr_channel = 0
        
        elif 'openvibeSignal' in self.amp_name:
            self.multiplier = 10**6 # change V -> uV unit for OpenVibe sources
            self._lsl_tr_channel = find_event_channel(ch_names=self.ch_list)
        
        elif 'openvibeMarkers' in self.amp_name:
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
        logger.info('Channels list %s(%s): ' % (self.amp_name, self.amp_serial))
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
        self.watchdog.reset()
        tslist = []
        received = False
        chunk = None
        lsl_clock = None
        
        while not received:
            while self.watchdog.sec() < 5:    
                # chunk = [frames]x[ch], tslist = [frames] 
                if len(tslist) == 0:
                    chunk, tslist = self.inlet.pull_chunk(max_samples=self._inlet_bufsize)
                    if blocking == False and len(tslist) == 0:
                        return np.empty((0, len(self.ch_list))), [], None
                if len(tslist) > 0:
                    if timestamp_offset is True:
                        lsl_clock = pylsl.local_clock()
                    received = True
                    break
                time.sleep(0.0005)
            
            else:
                logger.warning('Timeout occurred while acquiring data. Amp driver bug?')
                # give up and return empty values to avoid deadlock
                return np.empty((0, len(self.ch_list))), []
        
        data = np.array(chunk)
        
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