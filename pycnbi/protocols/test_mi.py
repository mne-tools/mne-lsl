from __future__ import print_function, division

"""
Motor imagery testing.

After setting experimental parameters, it runs a trial with feedback
by calling the classify() method of a Feedback class object.
Trials are repeated until the set number of trials are achieved.


Kyuhwa Lee, 2015
Swiss Federal Institute of Technology (EPFL)


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

import os
import sys
import cv2
import imp
import time
import math
import scipy
import random
import pycnbi
import datetime
import numpy as np
import scipy.signal
import mne.io, mne.viz
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.glass.bgi_client as bgi_client
import pycnbi.triggers.pyLptControl as pyLptControl
from pycnbi.decoder.decoder import BCIDecoderDaemon, BCIDecoder
from pycnbi.triggers.trigger_def import trigger_def
from pycnbi.protocols.feedback import Feedback
from pycnbi import logger
from builtins import input

# visualization
keys = {'left':81, 'right':83, 'up':82, 'down':84, 'pgup':85, 'pgdn':86,
        'home':80, 'end':87, 'space':32, 'esc':27, ',':44, '.':46, 's':115,
        'c':99, '[':91, ']':93, '1':49, '!':33, '2':50, '@':64, '3':51, '#':35}
color = dict(G=(20, 140, 0), B=(210, 0, 0), R=(0, 50, 200), Y=(0, 215, 235),
             K=(0, 0, 0), W=(255, 255, 255), w=(200, 200, 200))


def load_config(cfg_file):
    cfg_file = qc.forward_slashify(cfg_file)
    if not (os.path.exists(cfg_file) and os.path.isfile(cfg_file)):
        raise IOError('%s cannot be loaded.' % os.path.realpath(cfg_file))
    return imp.load_source(cfg_file, cfg_file)

def check_config(cfg):
    critical_vars = [
        'CLS_MI',
        'TRIGGER_DEVICE',
        'TRIGGER_DEF',
        'DIRECTIONS',
        'TRIALS_EACH',
        'PROB_ALPHA_NEW'
    ]

    optional_vars = {
        'AMP_NAME':None,
        'AMP_SERIAL':None,
        'FAKE_CLS':None,
        'TRIALS_RANDOMIZE':True,
        'BAR_SLOW_START':None,
        'PARALLEL_DECODING':None,
        'SHOW_TRIALS':True,
        'FREE_STYLE':False,
        'REFRESH_RATE':30,
        'BAR_BIAS':None,
        'POSITIVE_FEEDBACK':False,
        'BAR_REACH_FINISH':False,
        'FEEDBACK_TYPE':'BAR',
        'BAR_STEP_LEFT':6,
        'BAR_STEP_RIGHT':6,
        'BAR_STEP_UP':6,
        'BAR_STEP_DOWN':6,
        'BAR_STEP_BOTH':6,
        'LOG_PROBS':False,
        'SHOW_CUE':True,
        'WITH_STIMO':False,
        'SCREEN_SIZE':(1920, 1080),
        'SCREEN_POS':(0, 0),
        'WITH_REX':False,
        'DEBUG_PROBS':False,
        'LOG_PROBS':False
    }

    if not (hasattr(cfg, 'BAR_STEP') or hasattr(cfg, 'BAR_STEP_LEFT') or\
            hasattr(cfg, 'BAR_STEP_RIGHT') or hasattr(cfg, 'BAR_STEP_UP') or\
            hasattr(cfg, 'BAR_STEP_DOWN') or hasattr(cfg, 'BAR_STEP_BOTH')):
            raise RuntimeError('BAR_STEP or at least two of BAR_STEP_* must be defined.')

    for key in critical_vars:
        if not hasattr(cfg, key):
            raise RuntimeError('%s not defined in config.' % key)

    for key in optional_vars:
        if not hasattr(cfg, key):
            setattr(cfg, key, optional_vars[key])
            logger.warning('load_cfg(): Setting undefined parameter %s=%s' % (key, getattr(cfg, key)))

    return cfg

# for batch script
def run(cfg):
    if cfg.FAKE_CLS is None:
        # chooose amp
        if cfg.AMP_NAME is None and cfg.AMP_SERIAL is None:
            amp_name, amp_serial = pu.search_lsl(ignore_markers=True)
        else:
            amp_name = cfg.AMP_NAME
            amp_serial = cfg.AMP_SERIAL
        fake_dirs = None
    else:
        amp_name = None
        amp_serial = None
        fake_dirs = [v for (k, v) in cfg.DIRECTIONS]

    # events and triggers
    tdef = trigger_def(cfg.TRIGGER_FILE)
    #if cfg.TRIGGER_DEVICE is None:
    #    input('\n** Warning: No trigger device set. Press Ctrl+C to stop or Enter to continue.')
    trigger = pyLptControl.Trigger(cfg.TRIGGER_DEVICE)
    if trigger.init(50) == False:
        logger.error('Cannot connect to USB2LPT device. Use a mock trigger instead?')
        input('Press Ctrl+C to stop or Enter to continue.')
        trigger = pyLptControl.MockTrigger()
        trigger.init(50)

    # init classification
    decoder = BCIDecoderDaemon(cfg.CLS_MI, buffer_size=1.0, fake=(cfg.FAKE_CLS is not None),
                               amp_name=amp_name, amp_serial=amp_serial, fake_dirs=fake_dirs,
                               parallel=cfg.PARALLEL_DECODING, alpha_new=cfg.PROB_ALPHA_NEW)

    # OLD: requires trigger values to be always defined
    #labels = [tdef.by_value[x] for x in decoder.get_labels()]
    # NEW: events can be mapped into integers:
    labels = []
    dirdata = set([d[1] for d in cfg.DIRECTIONS])
    for x in decoder.get_labels():
        if x not in dirdata:
            labels.append(tdef.by_value[x])
        else:
            labels.append(x)

    # map class labels to bar directions
    bar_def = {label:str(dir) for dir, label in cfg.DIRECTIONS}
    bar_dirs = [bar_def[l] for l in labels]
    dir_seq = []
    for x in range(cfg.TRIALS_EACH):
        dir_seq.extend(bar_dirs)
    if cfg.TRIALS_RANDOMIZE:
        random.shuffle(dir_seq)
    else:
        dir_seq = [d[0] for d in cfg.DIRECTIONS] * cfg.TRIALS_EACH
    num_trials = len(dir_seq)

    logger.info('Initializing decoder.')
    while decoder.is_running() is 0:
        time.sleep(0.01)

    # bar visual object
    if cfg.FEEDBACK_TYPE == 'BAR':
        from pycnbi.protocols.viz_bars import BarVisual
        visual = BarVisual(cfg.GLASS_USE, screen_pos=cfg.SCREEN_POS,
            screen_size=cfg.SCREEN_SIZE)
    elif cfg.FEEDBACK_TYPE == 'BODY':
        assert hasattr(cfg, 'FEEDBACK_IMAGE_PATH'), 'FEEDBACK_IMAGE_PATH is undefined in your config.'
        from pycnbi.protocols.viz_human import BodyVisual
        visual = BodyVisual(cfg.FEEDBACK_IMAGE_PATH, use_glass=cfg.GLASS_USE,
            screen_pos=cfg.SCREEN_POS, screen_size=cfg.SCREEN_SIZE)
    visual.put_text('Waiting to start')
    if cfg.LOG_PROBS:
        logdir = qc.parse_path_list(cfg.CLS_MI)[0]
        probs_logfile = time.strftime(logdir + "probs-%Y%m%d-%H%M%S.txt", time.localtime())
    else:
        probs_logfile = None
    feedback = Feedback(cfg, visual, tdef, trigger, probs_logfile)

    # start
    trial = 1
    dir_detected = []
    prob_history = {c:[] for c in bar_dirs}
    while trial <= num_trials:
        if cfg.SHOW_TRIALS:
            title_text = 'Trial %d / %d' % (trial, num_trials)
        else:
            title_text = 'Ready'
        true_label = dir_seq[trial - 1]

        # profiling feedback
        #import cProfile
        #pr = cProfile.Profile()
        #pr.enable()
        result = feedback.classify(decoder, true_label, title_text, bar_dirs, prob_history=prob_history)
        #pr.disable()
        #pr.print_stats(sort='time')

        if result is None:
            break
        else:
            pred_label = result
        dir_detected.append(pred_label)

        if cfg.WITH_REX is True and pred_label == true_label:
            # if cfg.WITH_REX is True:
            if pred_label == 'U':
                rex_dir = 'N'
            elif pred_label == 'L':
                rex_dir = 'W'
            elif pred_label == 'R':
                rex_dir = 'E'
            elif pred_label == 'D':
                rex_dir = 'S'
            else:
                logger.warning('Rex cannot execute undefined action %s' % pred_label)
                rex_dir = None
            if rex_dir is not None:
                visual.move(pred_label, 100, overlay=False, barcolor='B')
                visual.update()
                logger.info('Executing Rex action %s' % rex_dir)
                os.system('%s/Rex/RexControlSimple.exe %s %s' % (pycnbi.ROOT, cfg.REX_COMPORT, rex_dir))
                time.sleep(8)

        if true_label == pred_label:
            msg = 'Correct'
        else:
            msg = 'Wrong'
        if cfg.TRIALS_RETRY is False or true_label == pred_label:
            logger.info('Trial %d: %s (%s -> %s)' % (trial, msg, true_label, pred_label))
            trial += 1

    if len(dir_detected) > 0:
        # write performance and log results
        fdir, _, _ = qc.parse_path_list(cfg.CLS_MI)
        logfile = time.strftime(fdir + "/online-%Y%m%d-%H%M%S.txt", time.localtime())
        with open(logfile, 'w') as fout:
            fout.write('Ground-truth,Prediction\n')
            for gt, dt in zip(dir_seq, dir_detected):
                fout.write('%s,%s\n' % (gt, dt))
            cfmat, acc = qc.confusion_matrix(dir_seq, dir_detected)
            fout.write('\nAccuracy %.3f\nConfusion matrix\n' % acc)
            fout.write(cfmat)
            logger.info('Log exported to %s' % logfile)
        print('\nAccuracy %.3f\nConfusion matrix\n' % acc)
        print(cfmat)

    visual.finish()
    if decoder:
        decoder.stop()

    '''
    # automatic thresholding
    if prob_history and len(bar_dirs) == 2:
        total = sum(len(prob_history[c]) for c in prob_history)
        fout = open(probs_logfile, 'a')
        msg = 'Automatic threshold optimization.\n'
        max_acc = 0
        max_bias = 0
        for bias in np.arange(-0.99, 1.00, 0.01):
            corrects = 0
            for p in prob_history[bar_dirs[0]]:
                p_biased = (p + bias) / (bias + 1) # new sum = (p+bias) + (1-p) = bias+1
                if p_biased >= 0.5:
                    corrects += 1
            for p in prob_history[bar_dirs[1]]:
                p_biased = (p + bias) / (bias + 1) # new sum = (p+bias) + (1-p) = bias+1
                if p_biased < 0.5:
                    corrects += 1
            acc = corrects / total
            msg += '%s%.2f: %.3f\n' % (bar_dirs[0], bias, acc)
            if acc > max_acc:
                max_acc = acc
                max_bias = bias
        msg += 'Max acc = %.3f at bias %.2f\n' % (max_acc, max_bias)
        fout.write(msg)
        fout.close()
        print(msg)
    '''

    logger.info('Finished.')

# for batch script
def batch_run(cfg_file):
    cfg = load_config(cfg_file)
    cfg = check_config(cfg)
    run(cfg)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_file = input('Config file name? ')
    else:
        cfg_file = sys.argv[1]
    batch_run(cfg_file)
