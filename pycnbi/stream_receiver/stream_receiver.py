from __future__ import print_function, division

"""
stream_receiver.py

Acquires signals from LSL server and save into buffer.

Note:
- Trigger channel is always 0 and EEG channels start from 1. When there is no
  known trigger channel, channel 0 is added with zero values for consistency.

- BioSemi's Trigger values (Ch.0) should be taken with care because all
  trigger pins are pulled to 1 and the data is written with 32 bits. Since
  the usual parallel port supports only 8 bits, remove the highest 24 bits
  by masking to 0 and subtract 1. A quick and dirty way is to subtract the
  most commonly occurring value, which usually corresponds to zero value.
  It only works when 0's are majority.

- Some LSL servers, especially OpenVibe-based servers, send wrong LSL timestamps.
  Most of the time, it does not matter but when you use software trigger, you will
  need this offset to synchronize the event timings.

TODO:
   Restrict buffer size.

Kyuhwa Lee, 2019
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import sys
import pdb
import math
import time
import pylsl
import numpy as np
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
from pycnbi.utils.pycnbi_utils import find_event_channel
from pycnbi import logger

class StreamReceiver:
    def __init__(self, window_size=1, buffer_size=1, amp_serial=None, eeg_only=False, amp_name=None):
        """
        Params:
            window_size (in seconds): keep the latest window_size seconds of the buffer.
            buffer_size (in seconds): 1-day is the maximum size. Large buffer may lead to a delay if not pulled frequently.
            amp_name: connect to a server named 'amp_name'. None: no constraint.
            amp_serial: connect to a server with serial number 'amp_serial'. None: no constraint.
            eeg_only: ignore non-EEG servers
        """
        _MAX_BUFFER_SIZE = 86400 # max buffer size allowed by StreamReceiver (24 hours)
        _MAX_PYLSL_STREAM_BUFSIZE = 360 # max buffer size for pylsl.StreamInlet

        if window_size <= 0:
            logger.error('Wrong window_size %d.' % window_size)
            raise ValueError()
        self.winsec = window_size
        if buffer_size == 0:
            buffer_size = _MAX_BUFFER_SIZE
        elif buffer_size < 0 or buffer_size > _MAX_BUFFER_SIZE:
            logger.error('Improper buffer size %.1f. Setting to %d.' % (buffer_size, _MAX_BUFFER_SIZE))
            buffer_size = _MAX_BUFFER_SIZE
        elif buffer_size < self.winsec:
            logger.error('Buffer size %.1f is smaller than window size. Setting to %.1f.' % (buffer_size, self.winsec))
            buffer_size = self.winsec
        self.bufsec = buffer_size
        self.bufsize = 0 # to be calculated using sampling rate
        self.stream_bufsec = int(math.ceil(min(_MAX_PYLSL_STREAM_BUFSIZE, self.bufsec)))
        self.stream_bufsize = 0 # to be calculated using sampling rate
        self.amp_serial = amp_serial
        self.eeg_only = eeg_only
        self.amp_name = amp_name
        self.tr_channel = None  # trigger indx used by StreamReceiver class
        self.eeg_channels = []  # signal indx used by StreamReceiver class
        self._lsl_tr_channel = None  # raw trigger indx in pylsl.pull_chunk()
        self._lsl_eeg_channels = []  # raw signal indx in pylsl.pull_chunk()
        self.ready = False  # False until the buffer is filled for the first time
        self.connected = False
        self.buffers = []
        self.timestamps = []
        self.watchdog = qc.Timer()
        self.multiplier = 1  # 10**6 for uV unit (automatically updated for openvibe servers)

        self.connect()

    def connect(self, find_any=True):
        """
        Run in child process
        """
        server_found = False
        amps = []
        channels = 0
        while server_found == False:
            if self.amp_name is None and self.amp_serial is None:
                logger.info("Looking for a streaming server...")
            else:
                logger.info("Looking for %s (Serial %s) ..." % (self.amp_name, self.amp_serial))
            streamInfos = pylsl.resolve_streams()
            if len(streamInfos) > 0:
                # For now, only 1 amp is supported by a single StreamReceiver object.
                for si in streamInfos:
                    # is_slave= ('true'==pylsl.StreamInlet(si).info().desc().child('amplifier').child('settings').child('is_slave').first_child().value() )
                    inlet = pylsl.StreamInlet(si)
                    # LSL XML parser has a bug which crashes so do not use for now
                    #amp_serial = inlet.info().desc().child('acquisition').child_value('serial_number')
                    amp_serial = 'N/A'
                    amp_name = si.name()

                    # connect to a specific amp only?
                    if self.amp_serial is not None and self.amp_serial != amp_serial:
                        continue

                    # connect to a specific amp only?
                    if self.amp_name is not None and self.amp_name != amp_name:
                        continue

                    # EEG streaming server only?
                    if self.eeg_only and si.type() != 'EEG':
                        continue

                    if 'USBamp' in amp_name:
                        logger.info('Found USBamp streaming server %s (type %s, amp_serial %s) @ %s.' % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 16
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'BioSemi' in amp_name:
                        logger.info('Found BioSemi streaming server %s (type %s, amp_serial %s) @ %s.' % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 0  # or subtract -6684927? (value when trigger==0)
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'SmartBCI' in amp_name:
                        logger.info('Found SmartBCI streaming server %s (type %s, amp_serial %s) @ %s.' % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 23
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'StreamPlayer' in amp_name:
                        logger.info('Found StreamPlayer streaming server %s (type %s, amp_serial %s) @ %s.' % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 0
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'openvibeSignal' in amp_name:
                        logger.info('Found an Openvibe signal streaming server %s (type %s, amp_serial %s) @ %s.' % (amp_name, si.type(), amp_serial, si.hostname()))
                        ch_list = pu.lsl_channel_list(inlet)
                        self._lsl_tr_channel = find_event_channel(ch_names=ch_list)
                        channels += si.channel_count()
                        amps.append(si)
                        server_found = True
                        # OpenVibe standard unit is Volts, which is not ideal for some numerical computations
                        self.multiplier = 10**6 # change V -> uV unit for OpenVibe sources
                        break
                    elif 'openvibeMarkers' in amp_name:
                        logger.info('Found an Openvibe markers server %s (type %s, amp_serial %s) @ %s.' % (amp_name, si.type(), amp_serial, si.hostname()))
                        ch_list = pu.lsl_channel_list(inlet)
                        self._lsl_tr_channel = find_event_channel(ch_names=ch_list)
                        channels += si.channel_count()
                        amps.append(si)
                        server_found = True
                        break
                    elif find_any:
                        logger.info('Found a streaming server %s (type %s, amp_serial %s) @ %s.' % (amp_name, si.type(), amp_serial, si.hostname()))
                        ch_list = pu.lsl_channel_list(inlet)
                        self._lsl_tr_channel = find_event_channel(ch_names=ch_list)
                        channels += si.channel_count()
                        amps.append(si)
                        server_found = True
                        break
            time.sleep(1)

        self.amp_name = amp_name

        # define EEG channel indices
        self._lsl_eeg_channels = list(range(channels))
        if self._lsl_tr_channel is None:
            logger.warning('Trigger channel not fonud. Adding an empty channel 0.')
        else:
            if self._lsl_tr_channel != 0:
                logger.info_yellow('Trigger channel found at index %d. Moving to index 0.' % self._lsl_tr_channel)
            self._lsl_eeg_channels.pop(self._lsl_tr_channel)
        self._lsl_eeg_channels = np.array(self._lsl_eeg_channels)
        self.tr_channel = 0  # trigger channel is always set to 0.
        self.eeg_channels = np.arange(1, channels)  # signal channels start from 1.

        # create new inlets to read from the stream
        inlets_master = []
        inlets_slaves = []
        for amp in amps:
            # data type of the 2nd argument (max_buflen) is int according to LSL C++ specification!
            inlet = pylsl.StreamInlet(amp, max_buflen=self.stream_bufsec)
            inlets_master.append(inlet)
            self.buffers.append([])
            self.timestamps.append([])

        inlets = inlets_master + inlets_slaves
        sample_rate = amps[0].nominal_srate()
        logger.info('Channels: %d' % channels)
        logger.info('LSL Protocol version: %s' % amps[0].version())
        logger.info('Source sampling rate: %.1f' % sample_rate)
        logger.info('Unit multiplier: %.1f' % self.multiplier)

        #self.winsize = int(self.winsec * sample_rate)
        #self.bufsize = int(self.bufsec * sample_rate)
        self.winsize = int(round(self.winsec * sample_rate))
        self.bufsize = int(round(self.bufsec * sample_rate))
        self.stream_bufsize = int(round(self.stream_bufsec * sample_rate))
        self.sample_rate = sample_rate
        self.connected = True
        self.ch_list = ch_list
        self.inlets = inlets  # Note: not picklable!

        # TODO: check if there's any problem with multiple inlets
        if len(self.inlets) > 1:
            logger.warning('Merging of multiple acquisition servers is not supported yet. Only %s will be used.' % amps[0].name())
            '''
            for i in range(1, len(self.inlets)):
                chunk, tslist = self.inlets[i].pull_chunk(max_samples=self.stream_bufsize)
                self.buffers[i].extend(chunk)
                self.timestamps[i].extend(tslist)
                if self.bufsize > 0 and len(self.buffers[i]) > self.bufsize:
                    self.buffers[i] = self.buffers[i][-self.bufsize:]
            '''

        # create channel info
        if self._lsl_tr_channel is None:
            self.ch_list = ['TRIGGER'] + self.ch_list
        else:
            for i, chn in enumerate(self.ch_list):
                if chn == 'TRIGGER' or chn == 'TRG' or 'STI ' in chn:
                    self.ch_list.pop(i)
                    self.ch_list = ['TRIGGER'] + self.ch_list
                    break
        logger.info('self.ch_list %s' % self.ch_list)

        # fill in initial buffer
        logger.info('Waiting to fill initial buffer of length %d' % (self.winsize))
        while len(self.timestamps[0]) < self.winsize:
            self.acquire()
            time.sleep(0.1)
        self.ready = True
        logger.info('Start receiving stream data.')

    def acquire(self, blocking=True):
        """
        Reads data into buffer. It is a blocking function as default.

        Fills the buffer and return the current chunk of data and timestamps.

        Returns:
            data [samples x channels], timestamps [samples]
        """
        timestamp_offset = False
        if len(self.timestamps[0]) == 0:
            timestamp_offset = True

        self.watchdog.reset()
        tslist = []
        received = False
        chunk = None
        while not received:
            while self.watchdog.sec() < 5:
                # chunk = [frames]x[ch], tslist = [frames]
                if len(tslist) == 0:
                    chunk, tslist = self.inlets[0].pull_chunk(max_samples=self.stream_bufsize)
                    if blocking == False and len(tslist) == 0:
                        return np.empty((0, len(self.ch_list))), []
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
        if self.amp_name == 'BioSemi' and self._lsl_tr_channel is not None:
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

        # add data to buffer
        chunk = data.tolist()
        self.buffers[0].extend(chunk)
        self.timestamps[0].extend(tslist)
        if self.bufsize > 0 and len(self.timestamps[0]) > self.bufsize:
            self.buffers[0] = self.buffers[0][-self.bufsize:]
            self.timestamps[0] = self.timestamps[0][-self.bufsize:]

        if timestamp_offset is True:
            timestamp_offset = False
            logger.info('LSL timestamp = %s' % lsl_clock)
            logger.info('Server timestamp = %s' % self.timestamps[-1][-1])
            self.lsl_time_offset = self.timestamps[-1][-1] - lsl_clock
            logger.info('Offset = %.3f ' % (self.lsl_time_offset))
            if abs(self.lsl_time_offset) > 0.1:
                logger.warning('LSL server has a high timestamp offset.')
            else:
                logger.info_green('LSL time server synchronized')

        ''' TODO: test the merging of multiple streams
        # if we have multiple synchronized amps
        if len(self.inlets) > 1:
            for i in range(1, len(self.inlets)):
                chunk, tslist = self.inlets[i].pull_chunk(max_samples=len(tslist))  # [frames][channels]
                self.buffers[i].extend(chunk)
                self.timestamps[i].extend(tslist)
                if self.bufsize > 0 and len(self.buffers[i]) > self.bufsize:
                    self.buffers[i] = self.buffers[i][-self.bufsize:]
        '''

        # data= array[samples, channels], tslist=[samples]
        return (data, tslist)

    def check_connect(self):
        """
        Check connection and automatically connect if not connected
        """
        while not self.connected:
            logger.error('LSL server not connected yet. Trying to connect automatically.')
            self.connect()
            time.sleep(1)

    def set_window_size(self, window_size):
        """
        Set window size (in seconds)
        """
        self.check_connect()
        #self.winsize = int(window_size * self.sample_rate) + 1
        self.winsize = int(round(window_size * self.sample_rate)) + 1

    def get_channel_names(self):
        """
        Get a list of channels
        """
        return self.ch_list

    def get_window_list(self):
        """
        Get the latest window
        IT ONLY RETURNS list[amps][samples][channels]
        """
        self.check_connect()
        window = self.buffers[0][-self.winsize:]
        timestamps = self.timestamps[0][-self.winsize:]
        return window, timestamps

    def get_window(self, decim=1):
        """
        Get the latest window and timestamps in numpy format

        input
        -----
        decim (int): decimation factor

        output
        ------
        [samples x channels], [samples]
        """
        self.check_connect()
        window, timestamps = self.get_window_list()

        if len(timestamps) > 0:
            # window = array[[samples_ch1],[samples_ch2]...]
            return np.array(window), np.array(timestamps)
        else:
            return np.array([]), np.array([])

    def get_buffer_list(self):
        """
        Get entire buffer
        Returns the raw list: amps x samples x channels
        """
        self.check_connect()
        return self.buffers, self.timestamps

    def get_buffer(self):
        """
        Returns the entire buffer: samples x channels

        If multiple amps, signals are concatenated along the channel axis.
        """
        self.check_connect()
        try:
            if len(self.timestamps[0]) > 0:
                w = np.concatenate(self.buffers, axis=1) # samples x channels
                t = np.array(self.timestamps).reshape(-1, 1) # samples x 1
                return w, t
            else:
                return np.array([]), np.array([])
        except:
            logger.exception('Sorry! Unexpected error occurred in get_buffer(). Dropping into a shell for debugging.')
            pdb.pm()

    def get_buflen(self):
        """
        Return buffer length in seconds
        """
        return (len(self.timestamps[0]) / self.sample_rate)

    def get_sample_rate(self):
        """
        Sampling rate
        """
        return self.sample_rate

    def get_num_channels(self):
        """
        Total number of channels includingi trigger channel
        """
        return len(self.ch_list)

    def get_eeg_channels(self):
        """
        Returns indices of eeg channels excluding trigger channel
        """
        return self.eeg_channels

    def get_trigger_channel(self):
        """
        Return trigger channel (0-based index)
        """
        return self.tr_channel

    def get_lsl_offset(self):
        """
        Return time difference of acquisition server's time and LSL time

        OpenVibe servers often have a bug of sending its own running time instead of LSL time.
        """
        return self.lsl_time_offset

    def reset_buffer(self):
        """
        Clear buffers
        """
        self.buffers = []

    def is_ready(self):
        """
        Returns True if the buffer is not empty.
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
    sr = StreamReceiver(window_size=1, buffer_size=1, amp_serial=amp_serial, eeg_only=False, amp_name=amp_name)
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
