"""
Online decoder using frequency features.

Interleaved parallel decoding is supported to achieve high frequency decoding.
For example, if a single decoder takes 40ms to compute the likelihoods of a
single window, 100 Hz decoding rate can be achieved reliably using 4 cpu cores.

TODO: Allow self.ref_new to be a list.
TODO: Use Pathos to overcome non-picklable class object limitations.

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
import pickle
import random
import psutil
import numpy as np
from pathlbi import Path
from numpy import ctypeslib
import multiprocessing as mp
import multiprocessing.sharedctypes as sharedctypes

import neurodecode.utils.io as io

from neurodecode import logger
from neurodecode.utils.timer import Timer
from neurodecode.triggers import trigger_def
from neurodecode.utils.lsl import search_lsl
from neurodecode.utils.preprocess import preprocess
from neurodecode.stream_receiver import StreamReceiver

mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

#----------------------------------------------------------------------
class BCIDecoder(object):
    """
    Decoder class

    The labels order of self.labels and self.label_names match likelihood orders computed by get_prob().

    Parameters
    ----------
    amp_name : str
        Connect to a specific stream.
    classifier : str
        The (absolute) path to the classifier file (.pkl)
    buffer_size : float
        The signal buffer's length in seconds
    fake : bool
        If True, load a fake decoder
    label : mp.Variable('i')
        Used for adaptive classifier, share the actual trial label
    """
    #----------------------------------------------------------------------
    def __init__(self, amp_name, classifier=None, buffer_size=1.0, fake=False, label=None):

        self.classifier = classifier
        self.buffer_sec = buffer_size
        self.fake = fake
        self.amp_name = amp_name

        if self.fake == False:
            with open(self.classifier, 'rb') as f:
                model = pickle.load(f)
            if model is None:
                logger.error('Classifier model is None.')
                raise ValueError
            self.cls = model['cls']
            self.psde = model['psde']
            self._labels = list(self.cls.classes_)
            self._label_names = [model['classes'][k] for k in self.labels]
            self._spatial = model['spatial']
            self._spectral = model['spectral']
            self._notch = model['notch']
            self._w_seconds = model['w_seconds']
            self._w_frames = model['w_frames']
            self._wstep = model['wstep']
            self._sfreq = model['sfreq']
            if 'decim' not in model:
                model['decim'] = 1
            self._decim = model['decim']
            if not int(round(self._sfreq * self._w_seconds)) == self._w_frames:
                logger.error('sfreq * w_sec %d != w_frames %d' % (int(round(self._sfreq * self._w_seconds)), self._w_frames))
                raise RuntimeError

            if 'multiplier' in model:
                self._multiplier = model['multiplier']
            else:
                self._multiplier = 1

            # Stream Receiver
            self._sr = StreamReceiver(bufsize=self._w_seconds, winsize=self._w_seconds,
                                      stream_name=self.amp_name)
            if self._sfreq != self._sr.streams[self.amp_name].sample_rate:
                logger.error('Amplifier sampling rate (%.3f) != model sampling rate (%.3f). Stop.' % (self._sr.streams[self.amp_name].sample_rate, self._sfreq))
                raise RuntimeError

            self._spatial_ch = model['spatial_ch']
            self._spectral_ch = model['spectral_ch']
            self._notch_ch = model['notch_ch']
            self._ref_ch = model['ref_ch']

            self._ch_names = self._sr.streams[self.amp_name].ch_list
            mc = model['ch_names']
            self._picks = [self._ch_names.index(mc[p]) for p in model['picks']]

            # Map channel indices based on channel names of the streaming server
            if len(self._spatial_ch) > 0:
                self._spatial_ch = [self._ch_names.index(p) for p in model['spatial_ch']]
            if len(self._spectral_ch) > 0:
                self._spectral_ch = [self._ch_names.index(p) for p in model['spectral_ch']]
            if len(self._notch_ch) > 0:
                self._notch_ch = [self._ch_names.index(p) for p in model['notch_ch']]
            if self._ref_ch is not None:
                self._ref_ch['New'] = [self._ch_names.index(p) for p in model['_ref_ch']['New']]
                self._ref_ch['Old'] = [self._ch_names.index(p) for p in model['_ref_ch']['Old']]

            self.label = label

            if "SAVED_FEAT" in model:
                self.xdata = model["SAVED_FEAT"]["X"].tolist()
                self.ydata = model["SAVED_FEAT"]["Y"].tolist()

            # PSD buffer
            #psd_temp = self.psde.transform(np.zeros((1, len(self._picks), self._w_frames // self._decim)))
            #self.psd_shape = psd_temp.shape
            #self.psd_size = psd_temp.size
            #self.psd_buffer = np.zeros((0, self.psd_shape[1], self.psd_shape[2]))
            #self.psd_buffer = None

            self.ts_buffer = []

            logger.info('Loaded classifier %s (sfreq=%.3f, decim=%d)' % (' vs '.join(self.label_names), self._sfreq, self._decim))
        else:
            # Fake left-right decoder
            # TODO: parameterize directions using fake_dirs
            self._labels = [11, 9]
            self._label_names = ['LEFT_GO', 'RIGHT_GO']

    #----------------------------------------------------------------------
    @property
    def labels(self):
        """
        Class labels numbers in the same order as the likelihoods returned by get_prob()
        """
        return self._labels

    #----------------------------------------------------------------------
    @labels.setter
    def labels(self, labels):
        logger.warning("The class labels numbers cannot be changed.")

    #----------------------------------------------------------------------
    @property
    def label_names(self):
        """
        Class labels names in the same order as get_prob().
        """
        return self._label_names
    #----------------------------------------------------------------------
    @label_names.setter
    def label_names(self):
        logger.warning("The class labels names cannot be changed.")

    #----------------------------------------------------------------------
    def start(self):
        """
        To fit the BCIDecoderDaemon class
        """
        pass

    #----------------------------------------------------------------------
    def stop(self):
        """
        To fit the BCIDecoderDaemon class
        """
        pass

    #----------------------------------------------------------------------
    def get_prob(self, timestamp=False):
        """
        Read the latest window, apply preprocessing, compute PSD and class probabilities.

        Parameters
        -----------
        timestamp : bool
            If True, returns LSL timestamp of the leading edge of the window used for decoding.

        Returns
        -------
        np.array
            The likelihood P(X|C), where X=window, C=model
        float
            LSL timestamp of the leading edge of the window used for decoding.
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
            self._sr.acquire()
            w, ts = self._sr.get_window()  # w = times x channels
            t_prob = ts[-1]
            w = w.T  # -> channels x times

            # apply filters. Important: maintain the original channel order at this point.
            # TODO: Not compatible with the new structure of preprocess.
            w = preprocess(w, sfreq=self._sfreq, spatial=self._spatial, spatial_ch=self._spatial_ch,
                          spectral=self._spectral, spectral_ch=self._spectral_ch, notch=self._notch,
                          notch_ch=self._notch_ch, multiplier=self._multiplier, ch_names=self._ch_names,
                          rereference=self._ref_ch, decim=self._decim)

            # select the same channels used for training
            w = w[self._picks]

            # psd = channels x freqs
            psd = self.psde.transform(w.reshape((1, w.shape[0], w.shape[1])))

            # make a feautre vector and classify
            feats = np.concatenate(psd[0])

            # For adaptive classifier
            if self.label:
                with self.label.get_lock():
                    if self.label.value in self.labels:
                        self.xdata.append(feats.tolist())
                        self.ydata.append(self.label.value)
                    elif self.label.value == 0:
                        pass
                    elif self.label.value == 1:
                        self.cls.fit(self.xdata, self.ydata)
                        logger.info("Classifier retrained")
                        self.label.value = 0

            # compute likelihoods
            feats = feats.reshape(1, -1)
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
            return probs, None

    #----------------------------------------------------------------------
    def get_prob_unread(self, timestamp=False):
        '''
        Simply call get_prob

        Used to fit BCIDecoderDaemon class.

        Parameters
        -----------
        timestamp : bool
            If True, returns LSL timestamp of the leading edge of the window used for decoding.

        Returns
        -------
        np.array
            The likelihood P(X|C), where X=window, C=model
        float
            LSL timestamp of the leading edge of the window used for decoding.
        '''
        return self.get_prob(timestamp)

    #----------------------------------------------------------------------
    def get_psd(self):
        """
        Get the latest computed PSD

        Returns
        -------
        np.array
            The latest computed PSD
        """
        raise NotImplementedError('Sorry! PSD buffer is under testing.')
        # return self.psd_buffer[-1].reshape((1, -1))

    #----------------------------------------------------------------------
    def is_ready(self):
        """
        Ready to decode?

        Returns
        -------
        bool
            True if the StreamReceiver is connected to a stream.
        """
        return self._sr.is_connected()


#----------------------------------------------------------------------
class BCIDecoderDaemon(object):
    """
    Decoder daemon class

    Instance a BCIDecoder object in a child process (BCIDecoder class object cannot be pickled).
    Set parallel parameter to achieve high-frequency decoding using multiple cores.

    Example: If the decoder runs 32ms per cycle, we can set period=0.04, stride=0.01, num_strides=4 to achieve 100 Hz decoding.
    """
    #----------------------------------------------------------------------
    def __init__(self, amp_name, classifier, buffer_size=1.0, fake=False, \
                 fake_dirs=None, parallel=None, alpha_new=None, wait_init=True, label=None):
        """
        Parameters
        ----------
        amp_name : str
            Connect to a specific stream
        classifier: str
            The absolute path to the classifier file
        buffer_size : float
            The buffer window size in seconds
        fake : bool
            If True, create a mock decoder (fake probabilities biased to 1.0)
        fake_dirs : list

        parallel: dict(period, stride, num_strides)
            period (float): Decoding period length for a single decoder in seconds
            stride (float): Time step between decoders in seconds.
            num_strides (int): Number of decoders to run in parallel
        alpha_new : float
            The exponential smoothing factor, [0, 1].
            p_new = p_new * alpha_new + p_old * (1 - alpha_new)
        wait_init : bool
            If True, wait (block) until the initial buffer of the decoder is full.
        label : mp.Variable('i')
            Used for adaptive classifier, share the actual trial label
        """

        self.classifier = classifier
        self.buffer_sec = buffer_size
        self.startmsg = 'Decoder daemon started.'
        self.stopmsg = 'Decoder daemon stopped.'
        self.fake = fake
        self.amp_name = amp_name
        self.parallel = parallel
        self.wait_init = wait_init

        # For adaptive classifier
        self.label = label

        if alpha_new is None:
            alpha_new = 1
        if not 0 <= alpha_new <= 1:
            logger.error('alpha_new must be a real number between 0 and 1.')
            raise ValueError
        self.alpha_new = alpha_new
        self.alpha_old = 1 - alpha_new

        if fake == False or fake is None:
            with open(self.classifier, 'rb') as f:
                self.model = pickle.load(f)
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

    #----------------------------------------------------------------------
    def reset(self):
        """
        Reset the classifier to its initial state.
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
                self.procs.append(mp.Process(target=self._daemon, args=\
                    [self.classifier, self.probs, self.probs_smooth, self.pread, self.t_problast,\
                     self.running[i], self.return_psd, psd_ctypes, self.psdlock,\
                     dict(t_start=(t_start+i*stride), period=period), self.label]))
        else:
            self.running = [mp.Value('i', 0)]
            self.procs = [mp.Process(target=self._daemon, args=\
                [self.classifier, self.probs, self.probs_smooth, self.pread, self.t_problast,\
                 self.running[0], self.return_psd, psd_ctypes, self.psdlock, None, self.label])]

    #----------------------------------------------------------------------
    def _daemon(self, classifier, probs, probs_smooth, pread, t_problast, running, return_psd, psd_ctypes, lock, interleave=None, label=None):
        """
        Runs Decoder class as a daemon.
        """

        pid = os.getpid()
        ps = psutil.Process(pid)

        if os.name == 'posix':
            # Unix
            ps.nice(0)      # A negative value increases priority but requires root privilages
        else:
            # Windows
            ps.nice(psutil.HIGH_PRIORITY_CLASS)

        logger.debug('[DecodeWorker-%-6d] Decoder worker process started' % (pid))
        decoder = BCIDecoder(self.amp_name, classifier, buffer_size=self.buffer_sec, fake=self.fake, label=label)
        if self.fake == False:
            psd = ctypeslib.as_array(psd_ctypes)
        else:
            psd = None

        if interleave is None:
            # single-core decoding
            with running.get_lock():
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

    #----------------------------------------------------------------------
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

    #----------------------------------------------------------------------
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

    #----------------------------------------------------------------------
    def get_labels(self):
        """
        Get the labels number in the same order as the likelihoods returned by get_prob()

        Returns
        -------
        list : The list of labels number
        """
        return self.labels

    #----------------------------------------------------------------------
    def get_label_names(self):
        """
        Get the labels names in the same order as get_labels()

        Returns
        -------
        list : The list of labels name
        """
        return self.label_names

    #----------------------------------------------------------------------
    def get_prob(self, timestamp=False):
        """
        Get the latest computed classes probability

        Parameters
        ----------
        timestamp : bool
            If True, provide the timestamp of the returned probabilities

        Returns
        -------
        list : The classes probability.
        float : The timestamp only if asked for
        """
        self.pread.value = 1
        if timestamp:
            return list(self.probs[:]), self.t_problast.value
        else:
            return list(self.probs[:])

    #----------------------------------------------------------------------
    def get_prob_unread(self, timestamp=False):
        """
        If not previously read, get the latest computed classes probability.

        Parameters
        ----------
        timestamp : bool
            If True, provide the timestamp of the returned probabilities

        Returns
        -------
        list : The classes probability.
        float : The timestamp only if asked for
        """
        if self.pread.value == 0:
            return self.get_prob(timestamp)
        elif timestamp:
            return None, None
        else:
            return None

    #----------------------------------------------------------------------
    def get_prob_smooth(self, timestamp=False):
        """
        Get the smoothed probabilities

        Parameters
        ----------
        timestamp : bool
            If True, provide the timestamp of the returned probabilities

        Returns
        -------
        list : The classes probability.
        float : The timestamp only if asked for
        """
        self.pread.value = 1
        if timestamp:
            return list(self.probs_smooth[:]), self.t_problast.value
        else:
            return list(self.probs_smooth[:])

    #----------------------------------------------------------------------
    def get_prob_smooth_unread(self, timestamp=False):
        """
        If not previously read, get the latest computed smoothed classes probability.

        Parameters
        ----------
        timestamp : bool
            If True, provide the timestamp of the returned probabilities

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

    #----------------------------------------------------------------------
    def get_psd(self):
        """
        Return the latest computed PSD

        Returns
        -------
        The latest computed PSD
        """
        self.return_psd.value = 1
        while self.return_psd.value == 1:
            time.sleep(0.001)
        return self.psd

    #----------------------------------------------------------------------
    def is_running(self):
        """
        Provide the number of daemon processes running

        Returns
        -------
        The number of daemons running
        """
        return sum([v.value for v in self.running])


#----------------------------------------------------------------------
def get_decoder_info(classifier):
    """
    Get the classifier information from a .pkl file.

    Parameters
    ----------
    classifier : str
        The (absolute) path to the classifier file (.pkl)

    Returns
    -------
    dict : Classifier info
    """

    with open(classifier, 'rb') as f:
        model = pickle.load(f)
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


#----------------------------------------------------------------------
def _log_decoding_helper(state, event_queue, amp_name=None, autostop=False):
    """
    Helper function to run StreamReceiver object in background

    Parameters
    ----------
    state : mp.Value
        The multiprocessing sharing variable
    event_queue : mp.Queue
        The queue used to share new events
    amp_name : str
        The stream name to connect to
    autostop : bool
        If True, automatically finish when no more data is received.
    """
    logger.info('Event acquisition subprocess started.')

    # wait for the start signal
    while state.value == 0:
        time.sleep(0.01)

    # acquire event values and returns event times and event values
    sr = StreamReceiver(bufsize=0, stream_name=amp_name)
    tm = Timer(autoreset=True)
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

#----------------------------------------------------------------------
def log_decoding(decoder, logfile, amp_name=None, pklfile=True, matfile=False, autostop=False, prob_smooth=False):
    """
    Decode online and write results with event timestamps

    Parameters
    ----------
    decoder : BCIDecoder or BCIDecoderDaemon class
        The decoder to use
    logfile : str
        The file path to contain the result in Python pickle format
    amp_name : str
        The stream name to connect to
    pklfile : bool
    If True, export the results to Python pickle format
    matfile : bool
    If True, export the results to .mat file
    autostop : bool
        If True, automatically finish when no more data is received.
    prob_smooth : bool
        If True, use smoothed probability values according to decoder's smoothing parameter.
    """

    import cv2
    import scipy

    # run event acquisition process in the background
    state = mp.Value('i', 1)
    event_queue = mp.Queue()
    proc = mp.Process(target=_log_decoding_helper, args=[state, event_queue, amp_name, autostop])
    proc.start()
    logger.info('Spawned event acquisition process.')

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
    tm_watchdog = Timer(autoreset=True)
    tm_cls = Timer()
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
        with open(logfile, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logger.info('Saved to %s' % logfile)
    if matfile:
        pp = io.parse_path(logfile)
        pp = Path(logfile)
        matout = '%s/%s.mat' % (pp.parent, pp.stem)
        scipy.io.savemat(matout, data)
        logger.info('Saved to %s' % matout)

#----------------------------------------------------------------------
def check_speed(decoder, max_count=float('inf')):
    """
    Test decoding speed accross several classifications.

    Parameters
    ----------
    decoder : BCIDecoder or BCIDecoderDaemon class
        The decoder to assess its performance
    max_count : int
        The number of classification for averaging
    """
    tm = Timer()
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

#----------------------------------------------------------------------
def sample_decoding(decoder):
    """
    Decoding example

    Parameters
    ----------
    decoder : The decoder to use
    """
    def get_index_max(seq):
        if type(seq) == list:
            return max(range(len(seq)), key=seq.__getitem__)
        elif type(seq) == dict:
            return max(seq, key=seq.__getitem__)
        else:
            logger.error('Unsupported input %s' % type(seq))
            return None

    # load trigger definitions for labeling
    labels = decoder.get_label_names()
    tm_watchdog = Timer(autoreset=True)
    tm_cls = Timer()
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
        maxi = get_index_max(psmooth)
        txt += '   %s' % labels[maxi]
        print(txt)
        tm_cls.reset()

#----------------------------------------------------------------------
if __name__ == '__main__':

    model_file = None
    from pathlib import Path

    if len(sys.argv) > 3:
        raise RuntimeError("Too many arguments provided, maximum is 2.")

    if len(sys.argv) > 2:
        model_file = sys.argv[2]

    if len(sys.argv) > 1:
        amp_name = sys.argv[1]
        if not model_file:
            model_file = str(Path(input(">> Provide the path to decoder file: \n")))

    if len(sys.argv) == 1:
        model_file = str(Path(input(">> Provide the path to decoder file: \n")))
        amp_name = search_lsl(ignore_markers=True)

    logger.info('Connecting to a server %s' % (amp_name))

    # run on background
    parallel = None # no process interleaving
    #parallel = dict(period=0.06, num_strides=3)
    # decoder = BCIDecoderDaemon(amp_name, model_file, buffer_size=1.0, fake=False, parallel=parallel, alpha_new=0.1)

    # run on foreground
    decoder = BCIDecoder(amp_name, model_file, buffer_size=1.0)

    # Assess classification speed
    check_speed(decoder, 5000)

    # Decoding example
    #sample_decoding(decoder)

    decoder.stop()
