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


Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import pycnbi_config  # from global common folder
import pycnbi_utils as pu
import time, sys
import pylsl
import numpy as np
import q_common as qc

def find_trigger_channel(ch_list):
    if 'TRIGGER' in ch_list:
        return ch_list.index('TRIGGER')
    elif 'TRG' in ch_list:
        return ch_list.index('TRG')
    else:
        for i, chn in enumerate(ch_list):
            if chn is None:
                continue
            # usually STI 014 for many trigger boxes
            if 'STI ' in chn:
                return i
        return None

class StreamReceiver:
    def __init__(self, window_size=1.0, buffer_size=0, amp_serial=None, eeg_only=False, amp_name=None):
        """
        Params:
            window_size (in seconds): keep the latest window_size seconds of the buffer.
            buffer_size (in seconds): keep everything if buffer_size=0.
            amp_name: connect to a server named 'amp_name'. None: no constraint.
            amp_serial: connect to a server with serial number 'amp_serial'. None: no constraint.
            eeg_only: ignore non-EEG servers
        """
        self.winsec = window_size
        self.bufsec = buffer_size
        self.amp_serial = amp_serial
        self.eeg_only = eeg_only
        self.amp_name = amp_name
        self.tr_channel = None  # trigger indx used by StreamReceiver class
        self.eeg_channels = []  # signal indx used by StreamReceiver class
        self._lsl_tr_channel = None  # raw trigger indx in pylsl.pull_chunk()
        self._lsl_eeg_channels = []  # raw signal indx in pylsl.pull_chunk()
        self.ready = False  # False until the buffer is filled for the first time

        self.bufsize = 0  # to be calculated using sampling rate
        self.connected = False
        self.buffers = []
        self.timestamps = []
        self.watchdog = qc.Timer()
        self.multiplier = 1  # 10**6 for uV unit (automatically updated for openvibe servers)

        self.connect()

    def print(self, msg, color='W'):
        qc.print_c('[StreamReceiver] %s' % msg, color)

    def connect(self, find_any=True):
        """
        Run in child process
        """
        server_found = False
        amps = []
        channels = 0
        while server_found == False:
            if self.amp_name is None and self.amp_serial is None:
                self.print("Looking for a streaming server...")
            else:
                self.print("Looking for %s (Serial %s) ..." % (self.amp_name, self.amp_serial))
            streamInfos = pylsl.resolve_streams()
            # print(streamInfos)
            if len(streamInfos) > 0:
                # For now, only 1 amp is supported by a single StreamReceiver object.
                for si in streamInfos:
                    # is_slave= ('true'==pylsl.StreamInlet(si).info().desc().child('amplifier').child('settings').child('is_slave').first_child().value() )
                    inlet = pylsl.StreamInlet(si)
                    amp_serial = inlet.info().desc().child('acquisition').child_value('serial_number')
                    amp_name = si.name()
                    # qc.print_c('Found %s (%s)'% (amp_name,amp_serial), 'G')

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
                        self.print('Found USBamp streaming server %s (type %s, amp_serial %s) @ %s.' \
                                   % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 16
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'BioSemi' in amp_name:
                        self.print('Found BioSemi streaming server %s (type %s, amp_serial %s) @ %s.' \
                                   % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 0  # or subtract -6684927? (value when trigger==0)
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'SmartBCI' in amp_name:
                        self.print('Found SmartBCI streaming server %s (type %s, amp_serial %s) @ %s.' \
                                   % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 23
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'StreamPlayer' in amp_name:
                        self.print('Found StreamPlayer streaming server %s (type %s, amp_serial %s) @ %s.' \
                                   % (amp_name, si.type(), amp_serial, si.hostname()))
                        self._lsl_tr_channel = 0
                        channels += si.channel_count()
                        ch_list = pu.lsl_channel_list(inlet)
                        amps.append(si)
                        server_found = True
                        break
                    elif 'openvibeSignal' in amp_name:
                        self.print('Found an Openvibe signal streaming server %s (type %s, amp_serial %s) @ %s.' \
                                   % (amp_name, si.type(), amp_serial, si.hostname()))
                        ch_list = pu.lsl_channel_list(inlet)
                        self._lsl_tr_channel = find_trigger_channel(ch_list)
                        channels += si.channel_count()
                        amps.append(si)
                        server_found = True
                        break
                    elif 'openvibeMarkers' in amp_name:
                        self.print('Found an Openvibe markers server %s (type %s, amp_serial %s) @ %s.' \
                                   % (amp_name, si.type(), amp_serial, si.hostname()))
                        ch_list = pu.lsl_channel_list(inlet)
                        self._lsl_tr_channel = find_trigger_channel(ch_list)
                        channels += si.channel_count()
                        amps.append(si)
                        server_found = True
                        break
                    elif find_any:
                        self.print('Found a streaming server %s (type %s, amp_serial %s) @ %s.' \
                                   % (amp_name, si.type(), amp_serial, si.hostname()))
                        ch_list = pu.lsl_channel_list(inlet)
                        self._lsl_tr_channel = find_trigger_channel(ch_list)
                        channels += si.channel_count()
                        amps.append(si)
                        server_found = True
                        break
            time.sleep(1)

        self.amp_name = amp_name

        # define EEG channel indices
        self._lsl_eeg_channels = range(channels)
        if self._lsl_tr_channel is None:
            self.print('Trigger channel not fonud. Adding an empty channel 0.', 'Y')
        else:
            self.print('Trigger channel found at index %d. Moving to index 0.' % self._lsl_tr_channel, 'Y')
            self._lsl_eeg_channels.pop(self._lsl_tr_channel)
        self._lsl_eeg_channels = np.array(self._lsl_eeg_channels)
        self.tr_channel = 0  # trigger channel is always set to 0.
        self.eeg_channels = np.arange(1, channels)  # signal channels start from 1.

        # create new inlets to read from the stream
        inlets_master = []
        inlets_slaves = []
        for amp in amps:
            inlet = pylsl.StreamInlet(amp)
            inlets_master.append(inlet)
            self.buffers.append([])
            self.timestamps.append([])

        inlets = inlets_master + inlets_slaves
        sample_rate = amps[0].nominal_srate()
        self.print('Channels: %d' % channels)
        self.print('LSL Protocol version: %s' % amps[0].version())
        self.print('Source sampling rate: %.1f' % sample_rate)
        self.print('Unit multiplier: %.1f' % self.multiplier)

        self.winsize = int(round(self.winsec * sample_rate))
        self.bufsize = int(round(self.bufsec * sample_rate))
        self.sample_rate = sample_rate
        self.connected = True
        self.inlets = inlets  # NOTE: not picklable!
        self.ch_list = ch_list

        # create channel info
        if self._lsl_tr_channel is None:
            self.ch_list = ['TRIGGER'] + self.ch_list
        else:
            for i, chn in enumerate(self.ch_list):
                if chn == 'TRIGGER' or chn == 'TRG' or 'STI ' in chn:
                    self.ch_list.pop(i)
                    self.ch_list = ['TRIGGER'] + self.ch_list
                    break
        qc.print_c('self.ch_list %s' % self.ch_list, 'Y')

        # fill in initial buffer
        self.print('Waiting to fill initial buffer of length %d' % (self.winsize))
        while len(self.timestamps[0]) < self.winsize:
            self.acquire()
            time.sleep(0.1)
        self.ready = True
        self.print('Start receiving stream data.')

    def acquire(self, blocking=True):
        """
        Reads data into buffer. Itis a blocking function as default.

        Fills the buffer and return the current chunk of data and timestamps.

        Returns:
            (data, timestamps) where
            data: [samples, channels]
            timestamps: [samples]

        TODO: add a parameter to set to non-blocking mode.
        """
        self.watchdog.reset()
        tslist = []
        while self.watchdog.sec() < 5:
            # retrieve chunk in [frame][ch]
            if len(tslist) == 0:
                chunk, tslist = self.inlets[0].pull_chunk()  # [frames][channels]
                if blocking == False and len(tslist) == 0:
                    return np.zeros((0, len(self.ch_list))), []
            if len(tslist) > 0:
                break
            time.sleep(0.001)
        else:
            self.print('Warning: Timeout occurred while acquiring data. Amp driver bug ?')
            return np.zeros((0, len(self.ch_list))), []
        data = np.array(chunk)

        # BioSemi has pull-up resistor instead of pull-down
        # import pdb; pdb.set_trace()
        if self.amp_name == 'BioSemi' and self._lsl_tr_channel is not None:
            datatype = data.dtype
            data[:, self._lsl_tr_channel] = (np.bitwise_and(255, data[:, self._lsl_tr_channel].astype(int)) - 1).astype(
                datatype)

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

        ########### TEST ###############
        timestamp_offset = False
        if len(self.timestamps[0]) == 0:
            timestamp_offset = True
        ########### TEST ###############

        # add data to buffer
        chunk = data.tolist()
        self.buffers[0].extend(chunk)
        self.timestamps[0].extend(tslist)
        if self.bufsize > 0 and len(self.timestamps) > self.bufsize:
            self.buffers[0] = self.buffers[0][-self.bufsize:]
            self.timestamps[0] = self.timestamps[0][-self.bufsize:]

        ########### TEST ###############
        if timestamp_offset is True:
            timestamp_offset = False
            print( 'LSL timestamp =', pylsl.local_clock() )
            print( 'OV timestamp =', self.timestamps[0][0] )
            self.lsl_time_offset = pylsl.local_clock() - self.timestamps[0][0]
            print( 'Offset = %.1f' % (self.lsl_time_offset) )
        ########### TEST ###############
            

        # if we have multiple synchronized amps
        if len(self.inlets) > 1:
            for i in range(1, len(self.inlets)):
                chunk, tslist = self.inlets[i].pull_chunk(max_samples=len(tslist))  # [frames][channels]
                self.buffers[i].extend(chunk)
                self.timestamps[i].extend(tslist)
                if self.bufsize > 0 and len(self.buffers[i]) > self.bufsize:
                    self.buffers[i] = self.buffers[i][-self.bufsize:]

        # data= array[samples, channels], tslist=[samples]
        return (data, tslist)

    def check_connect(self):
        """
        Check connection and automatically connect if not connected
        """
        while not self.connected:
            self.print('ERROR: LSL server not connected yet. Trying to connect automatically.')
            self.connect()
            time.sleep(1)

    def set_window_size(self, window_size):
        """
        Set window size (in seconds)
        """
        self.check_connect()
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

    def get_window(self):
        """
        Get the latest window and timestamps in numpy format
        """
        self.check_connect()
        window, timestamps = self.get_window_list()

        if len(timestamps) > 0:
            # window= array[[samples_ch1],[samples_ch2]...]
            # return (samples x channels, samples)
            return np.array(window), np.array(timestamps)
        else:
            return np.array([]), np.array([])

    def get_buffer_list(self):
        """
        Get entire buffer
        Returns the raw list: [amps][samples][channels]
        """
        self.check_connect()
        return self.buffers, self.timestamps

    def get_buffer(self):
        """
        Get entire buffer in numpy format: samples x channels
        """
        self.check_connect()
        try:
            if len(self.timestamps[0]) > 0:
                w = np.array(zip(*self.buffers))
                t = np.array(zip(*self.timestamps))
                return w.reshape((w.shape[0], w.shape[1] * w.shape[2])), t
            else:
                return np.array([]), np.array([])
        except:
            self.print('Sorry! Unexpected error occurred in get_buffer(). Dropping into a shell.')
            qc.shell()

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
if __name__ == '__main__':

    # settings
    CH_INDEX = [1]  # zero-baesd
    TIME_INDEX = None # integer or None. None = average of raw values of the current window
    SHOW_PSD = False


    import q_common as qc
    import mne

    amp_name, amp_serial = pu.search_lsl()
    sr = StreamReceiver(window_size=1, buffer_size=1, amp_serial=amp_serial, eeg_only=False, amp_name=amp_name)
    sfreq = sr.get_sample_rate()
    watchdog = qc.Timer()
    tm = qc.Timer(autoreset=True)
    trg_ch = sr.get_trigger_channel()
    last_ts = 0
    qc.print_c('Trigger channel: %d' % trg_ch, 'G')

    if SHOW_PSD:
        psde = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=1, fmax=50, bandwidth=None, \
            adaptive=False, low_bias=True, n_jobs=1, normalization='length', verbose=None)

    while True:
        sr.acquire()
        window, tslist = sr.get_window() # window = [samples x channels]
        window = window.T

        # print event values
        tsnew = np.where(np.array(tslist) > last_ts)[0][0]
        trigger = np.unique(window[trg_ch, tsnew:])

        # for Biosemi
        # if sr.amp_name=='BioSemi':
        #    trigger= set( [255 & int(x-1) for x in trigger ] )

        if len(trigger) > 0:
            qc.print_c('Triggers: %s' % np.array(trigger), 'G')

        print('[%.1f] Receiving data...' % watchdog.sec())
        
        if TIME_INDEX is None:
            datatxt = qc.list2string(np.mean(window[CH_INDEX, :], axis=1), '%-15.6f')
            print('[%.3f : %.3f]' % (tslist[0], tslist[1]) + ' data: %s' % datatxt)
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
