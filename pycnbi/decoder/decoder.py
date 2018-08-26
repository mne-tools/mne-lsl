from __future__ import print_function, division

"""
Online decoder using frequency features.

Interleaved parallel decoding is supported to achieve high frequency decoding.
For example, if a single decoder takes 32ms to compute the likelihoods of a single window,
100 Hz decoding rate can be achieved reliably using 4 cpu cores.

TODO:
Allow self.ref_new to be a list.

TODO:
Use Pathos to overcome non-picklable class object limitations.

Auhtor:
Kyuhwa Lee
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import time
import math
import random
import os
import sys
import psutil
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
import numpy as np
import multiprocessing as mp
import multiprocessing.sharedctypes as sharedctypes
import mne
from pycnbi.stream_receiver.stream_receiver import StreamReceiver
from pycnbi.triggers.trigger_def import trigger_def
from numpy import ctypeslib
mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

def get_decoder_info(classifier):
    """
    Get only the classifier information without connecting to a server

    Params
    ------
        classifier: model file

    Returns
    -------
        info dictionary object
    """

    model = qc.load_obj(classifier)
    if model == None:
        print('>> Error loading %s' % model)
        sys.exit(-1)

    cls = model['cls']
    psde = model['psde']
    labels = list(cls.classes_)
    w_seconds = model['w_seconds']
    w_frames = model['w_frames']
    wstep = model['wstep']
    sfreq = model['sfreq']
    psd_temp = psde.transform(np.zeros((1, len(model['picks']), w_frames)))
    psd_shape = psd_temp.shape
    psd_size = psd_temp.size

    info = dict(labels=labels, cls=cls, psde=psde, w_seconds=w_seconds, w_frames=w_frames,\
                wstep=wstep, sfreq=sfreq, psd_shape=psd_shape, psd_size=psd_size)
    return info


class BCIDecoder(object):
    """
    Decoder class

    The label order of self.labels and self.label_names match likelihood orders computed by get_prob()

    """

    def __init__(self, classifier=None, buffer_size=1.0, fake=False, amp_serial=None, amp_name=None):
        """
        Params
        ------
        classifier: classifier file
        spatial: spatial filter to use
        buffer_size: length of the signal buffer in seconds
        """

        self.classifier = classifier
        self.buffer_sec = buffer_size
        self.fake = fake
        self.amp_serial = amp_serial
        self.amp_name = amp_name

        if self.fake == False:
            model = qc.load_obj(self.classifier)
            if model == None:
                self.print('Error loading %s' % model)
                sys.exit(-1)
            self.cls = model['cls']
            self.psde = model['psde']
            self.labels = list(self.cls.classes_)
            self.label_names = [model['classes'][k] for k in self.labels]
            self.spatial = model['spatial']
            self.spectral = model['spectral']
            self.notch = model['notch']
            self.w_seconds = model['w_seconds']
            self.w_frames = model['w_frames']
            self.wstep = model['wstep']
            self.sfreq = model['sfreq']
            if not int(self.sfreq * self.w_seconds) == self.w_frames:
                raise RuntimeError('sfreq * w_sec %d != w_frames %d' % (int(self.sfreq * self.w_seconds), self.w_frames))

            if 'multiplier' in model:
                self.multiplier = model['multiplier']
            else:
                self.multiplier = 1

            # Stream Receiver
            self.sr = StreamReceiver(window_size=self.w_seconds, amp_name=self.amp_name, amp_serial=self.amp_serial)
            if self.sfreq != self.sr.sample_rate:
                raise RuntimeError('Amplifier sampling rate (%.1f) != model sampling rate (%.1f). Stop.' % (
                    self.sr.sample_rate, self.sfreq))

            # Map channel indices based on channel names of the streaming server
            self.spatial_ch = model['spatial_ch']
            self.spectral_ch = model['spectral_ch']
            self.notch_ch = model['notch_ch']
            self.ref_new = model['ref_new']
            self.ref_old = model['ref_old']
            self.ch_names = self.sr.get_channel_names()
            mc = model['ch_names']
            self.picks = [self.ch_names.index(mc[p]) for p in model['picks']]
            if self.spatial_ch is not None:
                self.spatial_ch = [self.ch_names.index(mc[p]) for p in model['spatial_ch']]
            if self.spectral_ch is not None:
                self.spectral_ch = [self.ch_names.index(mc[p]) for p in model['spectral_ch']]
            if self.notch_ch is not None:
                self.notch_ch = [self.ch_names.index(mc[p]) for p in model['notch_ch']]
            if self.ref_new is not None:
                self.ref_new = self.ch_names.index(mc[model['ref_new']])
            if self.ref_old is not None:
                self.ref_old = self.ch_names.index(mc[model['ref_old']])

            # PSD buffer
            psd_temp = self.psde.transform(np.zeros((1, len(self.picks), self.w_frames)))
            self.psd_shape = psd_temp.shape
            self.psd_size = psd_temp.size
            self.psd_buffer = np.zeros((0, self.psd_shape[1], self.psd_shape[2]))
            self.ts_buffer = []

        else:
            # Fake left-right decoder
            model = None
            self.psd_shape = None
            self.psd_size = None
            # TODO: parameterize directions using fake_dirs
            self.labels = [11, 9]
            self.label_names = ['LEFT_GO', 'RIGHT_GO']

    def print(self, *args):
        if len(args) > 0: print('[BCIDecoder] ', end='')
        print(*args)

    def get_labels(self):
        """
        Returns
        -------
        Class labels numbers in the same order as the likelihoods returned by get_prob()
        """
        return self.labels

    def get_label_names(self):
        """
        Returns
        -------
        Class label names in the same order as get_labels()
        """
        return self.label_names

    def start(self):
        pass

    def stop(self):
        pass

    def get_prob(self):
        """
        Read the latest window

        Returns
        -------
        The likelihood P(X|C), where X=window, C=model
        """
        if self.fake:
            # fake deocder: biased likelihood for the first class
            probs = [random.uniform(0.0, 1.0)]
            # others class likelihoods are just set to equal
            p_others = (1 - probs[0]) / (len(self.labels) - 1)
            for x in range(1, len(self.labels)):
                probs.append(p_others)
            time.sleep(0.0625)  # simulated delay for PSD + RF
        else:
            self.sr.acquire()
            w, ts = self.sr.get_window()  # w = times x channels
            w = w.T  # -> channels x times

            # apply filters. Important: maintain the original channel order at this point.
            pu.preprocess(w, sfreq=self.sfreq, spatial=self.spatial, spatial_ch=self.spatial_ch,
                          spectral=self.spectral, spectral_ch=self.spectral_ch, notch=self.notch,
                          notch_ch=self.notch_ch, multiplier=self.multiplier)

            # select the same channels used for training
            w = w[self.picks]

            # debug: show max - min
            # c=1; print( '### %d: %.1f - %.1f = %.1f'% ( self.picks[c], max(w[c]), min(w[c]), max(w[c])-min(w[c]) ) )

            # psd = channels x freqs
            psd = self.psde.transform(w.reshape((1, w.shape[0], w.shape[1])))

            # update psd buffer ( < 1 msec overhead )
            self.psd_buffer = np.concatenate((self.psd_buffer, psd), axis=0)
            self.ts_buffer.append(ts[0])
            if ts[0] - self.ts_buffer[0] > self.buffer_sec:
                # search speed comparison for ordered arrays:
                # http://stackoverflow.com/questions/16243955/numpy-first-occurence-of-value-greater-than-existing-value
                t_index = np.searchsorted(self.ts_buffer, ts[0] - 1.0)
                self.ts_buffer = self.ts_buffer[t_index:]
                self.psd_buffer = self.psd_buffer[t_index:, :, :]  # numpy delete is slower
            # assert ts[0] - self.ts_buffer[0] <= self.buffer_sec

            # make a feautre vector and classify
            feats = np.concatenate(psd[0]).reshape(1, -1)

            # compute likelihoods
            probs = self.cls.predict_proba(feats)[0]

        return probs

    def get_prob_unread(self):
        return self.get_prob()

    def get_psd(self):
        """
        Returns
        -------
        The latest computed PSD
        """
        return self.psd_buffer[-1].reshape((1, -1))

    def is_ready(self):
        """
        Ready to decode? Returns True if buffer is not empty.
        """
        return self.sr.is_ready()


class BCIDecoderDaemon(object):
    """
    BCI Decoder daemon class

    Some codes are redundant because BCIDecoder class object cannot be pickled.
    BCIDecoder object must be created inside the child process.
    Set parallel parameter to achieve high-frequency decoding using multiple cores.

    """

    def __init__(self, classifier=None, buffer_size=1.0, fake=False, amp_serial=None,\
                 amp_name=None, fake_dirs=None, parallel=None, alpha_new=None):
        """
        Params
        ------
        classifier: file name of the classifier
        buffer_size: buffer window size in seconds
        fake:
            False: Connect to an amplifier LSL server and decode
            True: Create a mock decoder (fake probabilities biased to 1.0)
        buffer_size: Buffer size in seconds.
        parallel: dict(period, stride, num_strides)
            period: Decoding period length for a single decoder in seconds.
            stride: Time step between decoders in seconds.
            num_strides: Number of decoders to run in parallel.
        alpha_new: exponential smoothing factor, real value in [0, 1].
            p_new = p_new * alpha_new + p_old * (1 - alpha_new)

        Example: If the decoder runs 32ms per cycle, we can set
                 period=0.04, stride=0.01, num_strides=4
                 to achieve 100 Hz decoding.
        """

        self.classifier = classifier
        self.buffer_sec = buffer_size
        self.startmsg = 'Decoder daemon started.'
        self.stopmsg = 'Decoder daemon stopped.'
        self.fake = fake
        self.amp_serial = amp_serial
        self.amp_name = amp_name
        self.parallel = parallel
        if alpha_new is None:
            alpha_new = 1
        if not 0 <= alpha_new <= 1:
            raise ValueError('alpha_new must be a real number between 0 and 1.')
        self.alpha_new = alpha_new
        self.alpha_old = 1 - alpha_new

        if fake == False or fake is None:
            self.model = qc.load_obj(self.classifier)
            if self.model == None:
                raise IOError('Error loading %s' % self.model)
            else:
                self.labels = self.model['cls'].classes_
                self.label_names = [self.model['classes'][k] for k in self.labels]
        else:
            # create a fake decoder with LEFT/RIGHT classes
            self.model = None
            tdef = trigger_def('triggerdef_16.ini')
            if type(fake_dirs) is not list:
                raise RuntimeError('Decoder(): wrong argument type of fake_dirs: %s.' % type(fake_dirs))
            self.labels = [tdef.by_key[t] for t in fake_dirs]
            self.label_names = [tdef.by_value[v] for v in self.labels]
            self.startmsg = '** WARNING: FAKE ' + self.startmsg
            self.stopmsg = 'FAKE ' + self.stopmsg

        self.psdlock = mp.Lock()
        self.reset()
        self.start()

    def print(self, *args):
        if len(args) > 0: print('[DecoderDaemon] ', end='')
        print(*args)

    def reset(self):
        """
        Reset classifier to the initial state
        """
        # share numpy array self.psd between processes.
        # to compute the shared memory size, we need to create a temporary decoder object.
        if self.fake == True:
            psd_size = None
            psd_shape = None
            psd_ctypes = None
            self.psd = None
        else:
            info = get_decoder_info(self.classifier)
            psd_size = info['psd_size']
            psd_shape = info['psd_shape'][1:]  # we get only the last window
            psd_ctypes = sharedctypes.RawArray('d', np.zeros(psd_size))
            self.psd = np.frombuffer(psd_ctypes, dtype=np.float64, count=psd_size)

        self.probs = mp.Array('d', [1.0 / len(self.labels)] * len(self.labels))
        self.probs_smooth = mp.Array('d', [1.0 / len(self.labels)] * len(self.labels))
        self.pread = mp.Value('i', 1)
        self.t_problast = mp.Value('d', 0)
        self.return_psd = mp.Value('i', 0)
        self.procs = []
        mp.freeze_support()

        if self.parallel:
            num_strides = self.parallel['num_strides']
            period = self.parallel['period']
            self.running = [mp.Value('i', 0)] * num_strides
            if num_strides > 1:
                stride = period / num_strides
            else:
                stride = 0
            t_start = time.time()
            for i in range(num_strides):
                self.procs.append(mp.Process(target=self.daemon, args=\
                    [self.classifier, self.probs, self.probs_smooth, self.pread, self.t_problast,\
                     self.running[i], self.return_psd, psd_ctypes, self.psdlock,\
                     dict(t_start=(t_start+i*stride), period=period)]))
        else:
            self.running = [mp.Value('i', 0)]
            self.procs = [mp.Process(target=self.daemon, args=\
                [self.classifier, self.probs, self.probs_smooth, self.pread, self.t_problast,\
                 self.running[0], self.return_psd, psd_ctypes, self.psdlock, None])]

    def daemon(self, classifier, probs, probs_smooth, pread, t_problast, running, return_psd, psd_ctypes, lock, interleave=None):
        """
        Runs Decoder class as a daemon.

        BCIDecoder object cannot be pickled but Pathos library may solve this problem and simplify the code.

        Input
        -----
        interleave: None or dict with the following keys:
        - t_start:double (seconds, same as time.time() format)
        - period:double (seconds)

        """

        pid = os.getpid()
        ps = psutil.Process(pid)
        ps.nice(psutil.HIGH_PRIORITY_CLASS)
        print('[DecodeWorker-%-6d] Decoder worker process started' % (pid))
        decoder = BCIDecoder(classifier, buffer_size=self.buffer_sec, fake=self.fake,\
                             amp_serial=self.amp_serial, amp_name=self.amp_name)
        if self.fake == False:
            psd = ctypeslib.as_array(psd_ctypes)
        else:
            psd = None

        if interleave is None:
            # single-core decoding
            running.value = 1
            while running.value == 1:
                # compute features and likelihoods
                probs[:] = decoder.get_prob()
                for i in range(len(probs_smooth)):
                    probs_smooth[i] = probs_smooth[i] * self.alpha_old + probs[i] * self.alpha_new
                pread.value = 0
                # copy back PSD values only when requested
                if self.fake == False and return_psd.value == 1:
                    lock.acquire()
                    psd[:] = decoder.psd_buffer[-1].reshape((1, -1))
                    lock.release()
                    return_psd.value = 0
        else:
            # interleaved parallel decoding
            t_start = interleave['t_start']
            period = interleave['period']
            running.value = 1
            t_next = t_start + math.ceil(((time.time() - t_start) / period)) * period

            while running.value == 1:
                # end of the current time slot
                t_next += period

                # compute likelihoods
                t_prob = time.time()
                probs_local = decoder.get_prob()
                
                # update the probs only if the current value is the latest
                if t_prob > t_problast.value:
                    lock.acquire()
                    probs[:] = probs_local
                    for i in range(len(probs_smooth)):
                        probs_smooth[i] = probs_smooth[i] * self.alpha_old + probs[i] * self.alpha_new
                    pread.value = 0
                    t_problast.value = t_prob
                    lock.release()
                
                # copy back PSD values only when requested
                if self.fake == False and return_psd.value == 1:
                    lock.acquire()
                    psd[:] = decoder.psd_buffer[-1].reshape((1, -1))
                    lock.release()
                    return_psd.value = 0

                # get the next time slot if didn't finish in the current slot
                if time.time() > t_next:
                    t_next_new = t_start + math.ceil(((time.time() - t_start) / period)) * period
                    print('[DecodeWorker-%-6d] High decoding delay (%.1f ms): t_next = %.3f -> %.3f' %\
                          (pid, (time.time() - t_next + period) * 1000, t_next, t_next_new))
                    t_next = t_next_new

                # sleep until the next time slot
                t_sleep = t_next - time.time()
                if t_sleep > 0.001:
                    time.sleep(t_sleep)
                #print('[DecodeWorker-%-6d] Woke up at %.3f' % (pid, time.time()))

    def start(self):
        """
        Start the daemon
        """
        if self.is_running() > 0:
            msg = 'ERROR: Cannot start. Daemon already running. (PID' +\
                ', '.join(['%d' % proc.pid for proc in self.procs]) + ')'
            self.print(msg)
            return
        for proc in self.procs:
            proc.start()
        for running in self.running:
            while running.value == 0:
                time.sleep(0.001)
        self.print(self.startmsg)

    def stop(self):
        """
        Stop the daemon
        """
        if self.is_running() == 0:
            self.print('Warning: Decoder already stopped.')
            return
        for running in self.running:
            running.value = 0
        for proc in self.procs:
            proc.join()
        self.reset()
        self.print(self.stopmsg)

    def get_labels(self):
        """
        Returns
        -------
        Class labels numbers in the same order as the likelihoods returned by get_prob()
        """
        return self.labels

    def get_label_names(self):
        """
        Returns
        -------
        Class label names in the same order as get_labels()
        """
        return self.label_names

    def get_prob(self):
        """
        Returns
        -------
        The last computed probability.
        """
        self.pread.value = 1
        return self.probs[:]

    def get_prob_unread(self):
        """
        Returns
        -------
        Probability if it's not read previously.
        None otherwise.
        """
        if self.pread.value == 0:
            return self.get_prob()
        else:
            return None

    def get_prob_smooth(self):
        """
        Returns
        -------
        The last computed probability.
        """
        self.pread.value = 1
        return self.probs_smooth[:]

    def get_prob_smooth_unread(self):
        """
        Returns
        -------
        Probability if it's not read previously.
        None otherwise.
        """
        if self.pread.value == 0:
            return self.get_prob_smooth()
        else:
            return None

    def get_psd(self):
        """
        Returns
        -------
        The latest computed PSD
        """
        self.return_psd.value = 1
        while self.return_psd.value == 1:
            time.sleep(0.001)
        return self.psd

    def is_running(self):
        """
        Returns
        -------
        The number of daemons running
        """
        return sum([v.value for v in self.running])

# test decoding speed
def check_speed(decoder, max_count=float('inf')):
    tm = qc.Timer()
    count = 0
    mslist = []
    while count < max_count:
        while decoder.get_prob_unread() is None:
            pass
        count += 1
        if tm.sec() > 1:
            t = tm.sec()
            ms = 1000.0 * t / count
            # show time per classification and its reciprocal
            print('%.0f ms/c   %.1f Hz' % (ms, count/t))
            mslist.append(ms)
            count = 0
            tm.reset()
    print('mean = %.1f ms' % np.mean(mslist))

# decoding example
def sample_decoding(decoder):
    # load trigger definitions for labeling
    labels = decoder.get_label_names()
    tm_watchdog = qc.Timer(autoreset=True)
    tm_cls = qc.Timer()
    while True:
        praw = decoder.get_prob_unread()
        psmooth = decoder.get_prob_smooth()
        if praw is None:
            # watch dog
            if tm_cls.sec() > 5:
                print('WARNING: No classification was done in the last 5 seconds. Are you receiving data streams?')
                tm_cls.reset()
            tm_watchdog.sleep_atleast(0.001)
            continue

        txt = '[%8.1f msec]' % (tm_cls.msec())
        for i, label in enumerate(labels):
            txt += '   %s %.3f (raw %.3f)' % (label, psmooth[i], praw[i])
        maxi = qc.get_index_max(psmooth)
        txt += '   %s' % labels[maxi]
        print(txt)
        tm_cls.reset()

# sample code
if __name__ == '__main__':
    model_file = r'D:\data\STIMO_EEG\DM002\offline\all\classifier_GB\classifier-64bit.pkl'

    if len(sys.argv) == 2:
        amp_name = sys.argv[1]
        amp_serial = None
    elif len(sys.argv) == 3:
        amp_name, amp_serial = sys.argv[1:3]
    else:
        amp_name, amp_serial = pu.search_lsl(ignore_markers=True)
    if amp_name == 'None':
        amp_name = None
    print('Connecting to a server %s (Serial %s).' % (amp_name, amp_serial))

    # run on background
    #parallel = None # no process interleaving
    parallel = dict(period=0.1, num_strides=5)
    decoder = BCIDecoderDaemon(model_file, buffer_size=1.0, fake=False, amp_name=amp_name,\
        amp_serial=amp_serial, parallel=parallel, alpha_new=0.1)

    # run on foreground
    #decoder = BCIDecoder(model_file, buffer_size=1.0, amp_name=amp_name, amp_serial=amp_serial)

    # run a fake classifier on background
    #decoder= BCIDecoderDaemon(fake=True, fake_dirs=['L','R'])

    check_speed(decoder, 5000)

    #sample_decoding(decoder)

    decoder.stop()
