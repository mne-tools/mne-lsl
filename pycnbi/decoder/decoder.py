from __future__ import print_function, division

"""
Online decoder using frequency features.

Interleaved parallel decoding is supported to achieve high frequency decoding.
For example, if a single decoder takes 40ms to compute the likelihoods of a single window,
100 Hz decoding rate can be achieved reliably using 4 cpu cores.

TODO:
Allow self.ref_new to be a list.

TODO:
Use Pathos to overcome non-picklable class object limitations.

Auhtor:
Kyuhwa Lee
Swiss Federal Institute of Technology Lausanne (EPFL)

"""

import os
import sys
import mne
import time
import math
import pylsl
import random
import psutil
import numpy as np
import multiprocessing as mp
import multiprocessing.sharedctypes as sharedctypes
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from numpy import ctypeslib
from pycnbi import logger
from pycnbi.triggers.trigger_def import trigger_def
from pycnbi.stream_receiver.stream_receiver import StreamReceiver
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
    if model is None:
        logger.error('>> Error loading %s' % model)
        raise ValueError

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
            if model is None:
                logger.error('Classifier model is None.')
                raise ValueError
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
            if 'decim' not in model:
                model['decim'] = 1
            self.decim = model['decim']
            if not int(round(self.sfreq * self.w_seconds)) == self.w_frames:
                logger.error('sfreq * w_sec %d != w_frames %d' % (int(round(self.sfreq * self.w_seconds)), self.w_frames))
                raise RuntimeError

            if 'multiplier' in model:
                self.multiplier = model['multiplier']
            else:
                self.multiplier = 1

            # Stream Receiver
            self.sr = StreamReceiver(window_size=self.w_seconds, amp_name=self.amp_name, amp_serial=self.amp_serial)
            if self.sfreq != self.sr.sample_rate:
                logger.error('Amplifier sampling rate (%.3f) != model sampling rate (%.3f). Stop.' % (self.sr.sample_rate, self.sfreq))
                raise RuntimeError

            # Map channel indices based on channel names of the streaming server
            self.spatial_ch = model['spatial_ch']
            self.spectral_ch = model['spectral_ch']
            self.notch_ch = model['notch_ch']
            #self.ref_ch = model['ref_ch'] # not supported yet
            self.ch_names = self.sr.get_channel_names()
            mc = model['ch_names']
            self.picks = [self.ch_names.index(mc[p]) for p in model['picks']]
            if self.spatial_ch is not None:
                self.spatial_ch = [self.ch_names.index(mc[p]) for p in model['spatial_ch']]
            if self.spectral_ch is not None:
                self.spectral_ch = [self.ch_names.index(mc[p]) for p in model['spectral_ch']]
            if self.notch_ch is not None:
                self.notch_ch = [self.ch_names.index(mc[p]) for p in model['notch_ch']]

            # PSD buffer
            #psd_temp = self.psde.transform(np.zeros((1, len(self.picks), self.w_frames // self.decim)))
            #self.psd_shape = psd_temp.shape
            #self.psd_size = psd_temp.size
            #self.psd_buffer = np.zeros((0, self.psd_shape[1], self.psd_shape[2]))
            #self.psd_buffer = None

            self.ts_buffer = []

            logger.info_green('Loaded classifier %s (sfreq=%.3f, decim=%d)' % (' vs '.join(self.label_names), self.sfreq, self.decim))
        else:
            # Fake left-right decoder
            model = None
            self.psd_shape = None
            self.psd_size = None
            # TODO: parameterize directions using fake_dirs
            self.labels = [11, 9]
            self.label_names = ['LEFT_GO', 'RIGHT_GO']

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

    def get_prob(self, timestamp=False):
        """
        Read the latest window

        Input
        -----
        timestamp: If True, returns LSL timestamp of the leading edge of the window used for decoding.

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
            time.sleep(0.0625)  # simulated delay
            t_prob = pylsl.local_clock()
        else:
            self.sr.acquire(blocking=True)
            w, ts = self.sr.get_window()  # w = times x channels
            t_prob = ts[-1]
            w = w.T  # -> channels x times

            # re-reference channels
            # TODO: add re-referencing function to preprocess()

            # apply filters. Important: maintain the original channel order at this point.
            w = pu.preprocess(w, sfreq=self.sfreq, spatial=self.spatial, spatial_ch=self.spatial_ch,
                          spectral=self.spectral, spectral_ch=self.spectral_ch, notch=self.notch,
                          notch_ch=self.notch_ch, multiplier=self.multiplier, decim=self.decim)

            # select the same channels used for training
            w = w[self.picks]

            # debug: show max - min
            # c=1; print( '### %d: %.1f - %.1f = %.1f'% ( self.picks[c], max(w[c]), min(w[c]), max(w[c])-min(w[c]) ) )

            # psd = channels x freqs
            psd = self.psde.transform(w.reshape((1, w.shape[0], w.shape[1])))

            # make a feautre vector and classify
            feats = np.concatenate(psd[0]).reshape(1, -1)

            # compute likelihoods
            probs = self.cls.predict_proba(feats)[0]

            # update psd buffer ( < 1 msec overhead )
            '''
            if self.psd_buffer is None:
                self.psd_buffer = psd
            else:
                self.psd_buffer = np.concatenate((self.psd_buffer, psd), axis=0)
                # TODO: CHECK THIS BLOCK
                self.ts_buffer.append(ts[0])
                if ts[0] - self.ts_buffer[0] > self.buffer_sec:
                    # search speed comparison for ordered arrays:
                    # http://stackoverflow.com/questions/16243955/numpy-first-occurence-of-value-greater-than-existing-value
                    #t_index = np.searchsorted(self.ts_buffer, ts[0] - 1.0)
                    t_index = np.searchsorted(self.ts_buffer, ts[0] - self.buffer_sec)
                    self.ts_buffer = self.ts_buffer[t_index:]
                    self.psd_buffer = self.psd_buffer[t_index:, :, :]  # numpy delete is slower
                # assert ts[0] - self.ts_buffer[0] <= self.buffer_sec
            '''

        if timestamp:
            return probs, t_prob
        else:
            return probs

    def get_prob_unread(self, timestamp=False):
        return self.get_prob(timestamp)

    def get_psd(self):
        """
        Returns
        -------
        The latest computed PSD
        """
        raise NotImplementedError('Sorry! PSD buffer is under testing.')
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
                 amp_name=None, fake_dirs=None, parallel=None, alpha_new=None, wait_init=True):
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
        wait_init: If True, wait (block) until the initial buffer of the decoder is full.

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
        self.wait_init = wait_init
        if alpha_new is None:
            alpha_new = 1
        if not 0 <= alpha_new <= 1:
            logger.error('alpha_new must be a real number between 0 and 1.')
            raise ValueError
        self.alpha_new = alpha_new
        self.alpha_old = 1 - alpha_new

        if fake == False or fake is None:
            self.model = qc.load_obj(self.classifier)
            if self.model == None:
                logger.error('Error loading %s' % self.model)
                raise IOError
            else:
                self.labels = self.model['cls'].classes_
                self.label_names = [self.model['classes'][k] for k in self.labels]
        else:
            # create a fake decoder with LEFT/RIGHT classes
            self.model = None
            tdef = trigger_def('triggerdef_16.ini')
            if type(fake_dirs) is not list:
                logger.error('Decoder(): wrong argument type of fake_dirs: %s.' % type(fake_dirs))
                raise RuntimeError
            self.labels = [tdef.by_name[t] for t in fake_dirs]
            self.label_names = [tdef.by_value[v] for v in self.labels]
            self.startmsg = '** WARNING: FAKE ' + self.startmsg
            self.stopmsg = 'FAKE ' + self.stopmsg

        self.psdlock = mp.Lock()
        self.reset()
        self.start()

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
            logger.error('Parallel decoding is under a rigorous test. Please do not use it for now.')
            raise NotImplementedError
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
        logger.debug('[DecodeWorker-%-6d] Decoder worker process started' % (pid))
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
                probs[:], t_prob = decoder.get_prob(True)
                probs_smooth_sum = 0
                for i in range(len(probs_smooth)):
                    probs_smooth[i] = probs_smooth[i] * self.alpha_old + probs[i] * self.alpha_new
                    probs_smooth_sum += probs_smooth[i]
                for i in range(len(probs_smooth)):
                    probs_smooth[i] /= probs_smooth_sum
                pread.value = 0
                t_problast.value = t_prob

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
                t_prob_wall = time.time()
                probs_local, t_prob_lsl = decoder.get_prob(True)

                # update the probs only if the current value is the latest
                ##################################################################
                # TODO: use timestamp to compare instead of time.time()
                ##################################################################
                if t_prob_wall > t_problast.value:
                    lock.acquire()
                    probs[:] = probs_local
                    for i in range(len(probs_smooth)):
                        probs_smooth[i] = probs_smooth[i] * self.alpha_old + probs[i] * self.alpha_new
                    pread.value = 0
                    t_problast.value = t_prob_wall
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
                    logger.warning('[DecodeWorker-%-6d] High decoding delay (%.1f ms): t_next = %.3f -> %.3f' %\
                          (pid, (time.time() - t_next + period) * 1000, t_next, t_next_new))
                    t_next = t_next_new

                # sleep until the next time slot
                t_sleep = t_next - time.time()
                if t_sleep > 0.001:
                    time.sleep(t_sleep)
                logger.debug('[DecodeWorker-%-6d] Woke up at %.3f' % (pid, time.time()))

    def start(self):
        """
        Start the daemon
        """
        if self.is_running() > 0:
            msg = 'Cannot start. Daemon already running. (PID' + ', '.join(['%d' % proc.pid for proc in self.procs]) + ')'
            logger.error(msg)
            return
        for proc in self.procs:
            proc.start()
        if self.wait_init:
            for running in self.running:
                while running.value == 0:
                    time.sleep(0.001)
        logger.info(self.startmsg)

    def stop(self):
        """
        Stop the daemon
        """
        if self.is_running() == 0:
            logger.warning('Decoder already stopped.')
            return
        for running in self.running:
            running.value = 0
        for proc in self.procs:
            proc.join(10)
            if proc.is_alive():
                logger.warning('Process %s did not die properly.' % proc.pid())
        self.reset()
        logger.info(self.stopmsg)

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

    def get_prob(self, timestamp=False):
        """
        Returns
        -------
        The last computed probability.
        """
        self.pread.value = 1
        if timestamp:
            return list(self.probs[:]), self.t_problast.value
        else:
            return list(self.probs[:])

    def get_prob_unread(self, timestamp=False):
        """
        Returns
        -------
        Probability if it's not read previously.
        None otherwise.
        """
        if self.pread.value == 0:
            return self.get_prob(timestamp)
        elif timestamp:
            return None, None
        else:
            return None

    def get_prob_smooth(self, timestamp=False):
        """
        Returns
        -------
        The last computed probability.
        """
        self.pread.value = 1
        if timestamp:
            return list(self.probs_smooth[:]), self.t_problast.value
        else:
            return list(self.probs_smooth[:])

    def get_prob_smooth_unread(self, timestamp=False):
        """
        Returns
        -------
        Probability if it's not read previously.
        None otherwise.
        """
        if self.pread.value == 0:
            return self.get_prob_smooth(timestamp)
        elif timestamp:
            return None, None
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



def log_decoding_helper(state, event_queue, amp_name=None, amp_serial=None, autostop=False):
    """
    Helper function to run StreamReceiver object in background
    """
    logger.info('Event acquisition subprocess started.')

    # wait for the start signal
    while state.value == 0:
        time.sleep(0.01)
    
    # acquire event values and returns event times and event values
    sr = StreamReceiver(buffer_size=0, amp_name=amp_name, amp_serial=amp_serial)
    tm = qc.Timer(autoreset=True)
    started = False
    while state.value == 1:
        chunk, ts_list = sr.acquire()
        if autostop:
            if started is True:
                if len(ts_list) == 0:
                    state.value = 0
                    break
            elif len(ts_list) > 0:
                started = True
        tm.sleep_atleast(0.001)
    logger.info('Event acquisition subprocess finishing up ...')

    buffers, times = sr.get_buffer()
    events = buffers[:, 0] # first channel is the trigger channel
    event_index = np.where(events != 0)[0]
    event_times = times[event_index].reshape(-1).tolist()
    event_values = events[event_index].tolist()
    assert len(event_times) == len(event_values)
    event_queue.put((event_times, event_values))

def log_decoding(decoder, logfile, amp_name=None, amp_serial=None, pklfile=True, matfile=False, autostop=False, prob_smooth=False):
    """
    Decode online and write results with event timestamps

    input
    -----
    decoder: Decoder or DecoderDaemon class object.
    logfile: File name to contain the result in Python pickle format.
    amp_name: LSL server name (if known).
    amp_serial: LSL server serial number (if known).
    pklfile: Export results to Python pickle format.
    matfile: Export results to Matlab .mat file if True.
    autostop: Automatically finish when no more data is received.
    prob_smooth: Use smoothed probability values according to decoder's smoothing parameter.
    """

    import cv2
    import scipy

    # run event acquisition process in the background
    state = mp.Value('i', 1)
    event_queue = mp.Queue()
    proc = mp.Process(target=log_decoding_helper, args=[state, event_queue, amp_name, amp_serial, autostop])
    proc.start()
    logger.info_green('Spawned event acquisition process.')

    # init variables and choose decoding function
    labels = decoder.get_label_names()
    probs = []
    prob_times = []
    if prob_smooth:
        decode_fn = decoder.get_prob_smooth_unread
    else:
        decode_fn = decoder.get_prob_unread
        
    # simple controller UI
    cv2.namedWindow("Decoding", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Decoding", 1400, 50)
    img = np.zeros([100, 400, 3], np.uint8)
    cv2.putText(img, 'Press any key to start', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Decoding", img)
    cv2.waitKeyEx()
    img *= 0
    cv2.putText(img, 'Press ESC to stop', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Decoding", img)

    key = 0
    started = False
    tm_watchdog = qc.Timer(autoreset=True)
    tm_cls = qc.Timer()
    while key != 27:
        prob, prob_time = decode_fn(True)
        t_lsl = pylsl.local_clock()
        key = cv2.waitKeyEx(1)
        if prob is None:
            # watch dog
            if tm_cls.sec() > 5:
                if autostop and started:
                    logger.info('No more streaming data. Finishing.')
                    break
                tm_cls.reset()
            tm_watchdog.sleep_atleast(0.001)
            continue
        probs.append(prob)
        prob_times.append(prob_time)
        txt = '[%.3f] ' % prob_time
        txt += ', '.join(['%s: %.2f' % (l, p) for l, p in zip(labels, prob)])
        txt += ' (%d ms, LSL Diff = %.3f)' % (tm_cls.msec(), (t_lsl-prob_time))
        logger.info(txt)
        if not started:
            started = True
        tm_cls.reset()

    # finish up processes
    cv2.destroyAllWindows()
    logger.info('Cleaning up event acquisition process.')
    state.value = 0
    decoder.stop()
    event_times, event_values = event_queue.get()
    proc.join()

    # save values
    if len(prob_times) == 0:
        logger.error('No decoding result. Please debug.')
        import pdb
        pdb.set_trace()
    t_start = prob_times[0]
    probs = np.vstack(probs)
    event_times = np.array(event_times)
    event_times = event_times[np.where(event_times >= t_start)[0]] - t_start
    prob_times = np.array(prob_times) - t_start
    event_values = np.array(event_values)
    data = dict(probs=probs, prob_times=prob_times, event_times=event_times, event_values=event_values, labels=labels)
    if pklfile:
        qc.save_obj(logfile, data)
        logger.info('Saved to %s' % logfile)
    if matfile:
        pp = qc.parse_path(logfile)
        matout = '%s/%s.mat' % (pp.dir, pp.name)
        scipy.io.savemat(matout, data)
        logger.info('Saved to %s' % matout)

def check_speed(decoder, max_count=float('inf')):
    """
    Test decoding speed
    """
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


def sample_decoding(decoder):
    """
    Decoding example
    """
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
                logger.warning('No classification was done in the last 5 seconds. Are you receiving data streams?')
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
    model_file = r'D:\data\STIMO_EEG\DM002\offline\all\classifier_L_vs_Feedback\classifier-64bit.pkl'

    if len(sys.argv) == 2:
        amp_name = sys.argv[1]
        amp_serial = None
    elif len(sys.argv) == 3:
        amp_name, amp_serial = sys.argv[1:3]
    else:
        amp_name, amp_serial = pu.search_lsl(ignore_markers=True)
    if amp_name == 'None':
        amp_name = None
    logger.info('Connecting to a server %s (Serial %s).' % (amp_name, amp_serial))

    # run on background
    parallel = None # no process interleaving
    #parallel = dict(period=0.06, num_strides=3)
    decoder = BCIDecoderDaemon(model_file, buffer_size=1.0, fake=False, amp_name=amp_name,\
        amp_serial=amp_serial, parallel=parallel, alpha_new=0.1)

    # run on foreground
    #decoder = BCIDecoder(model_file, buffer_size=1.0, amp_name=amp_name, amp_serial=amp_serial)

    # run a fake classifier on background
    #decoder= BCIDecoderDaemon(fake=True, fake_dirs=['L','R'])

    check_speed(decoder, 5000)

    #sample_decoding(decoder)

    decoder.stop()
