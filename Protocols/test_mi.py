from __future__ import print_function, division

"""
Motor imagery testing

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

import pycnbi_config
import sys, os, math, random, time, datetime
import importlib, imp
import pyLptControl
import cv2
import cv2.cv as cv
import numpy as np
import scipy, scipy.signal
import mne.io, mne.viz
import q_common as qc
import bgi_client
from decoder import BCIDecoderDaemon, BCIDecoder
from IPython import embed

# visualization
keys = {'left':81, 'right':83, 'up':82, 'down':84, 'pgup':85, 'pgdn':86, 'home':80, 'end':87, 'space':32,
        'esc':27 \
    , ',':44, '.':46, 's':115, 'c':99, '[':91, ']':93, '1':49, '!':33, '2':50, '@':64, '3':51, '#':35}
color = dict(G=(20, 140, 0), B=(210, 0, 0), R=(0, 50, 200), Y=(0, 215, 235), K=(0, 0, 0), W=(255, 255, 255),
             w=(200, 200, 200))


def check_cfg(cfg):
    if not hasattr(cfg, 'POSITIVE_FEEDBACK'):
        qc.print_c('Warning: POSITIVE_FEEDBACK undefined. Setting it to False.', 'Y')
        cfg.POSITIVE_FEEDBACK = False
    if not hasattr(cfg, 'BAR_REACH_FINISH'):
        qc.print_c('Warning: BAR_REACH_FINISH undefined. Setting it to False.', 'Y')
        cfg.BAR_REACH_FINISH = False
    if not hasattr(cfg, 'FEEDBACK_TYPE'):
        qc.print_c('Warning: FEEDBACK_TYPE undefined. Setting it to BAR.', 'Y')
        cfg.FEEDBACK_TYPE = 'BAR'
    if (not hasattr(cfg, 'BAR_STEP_LEFT')) or (not hasattr(cfg, 'BAR_STEP_RIGHT')):
        assert hasattr(cfg, 'BAR_STEP')
        qc.print_c('Warning: BAR_STEP_LEFT and BAR_STEP_RIGHT undefined. Setting it to BAR_STEP(%d).' % cfg.BAR_STEP, 'Y')
        cfg.BAR_STEP_LEFT = cfg.BAR_STEP_RIGHT = cfg.BAR_STEP
    return cfg


if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_module = raw_input('Config file name? ')
    else:
        cfg_module = sys.argv[1]
    cfg = check_cfg(imp.load_source(cfg_module, cfg_module))
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
    tdefmod = importlib.import_module(cfg.TRIGGER_DEF)
    tdef = tdefmod.TriggerDef()
    if cfg.TRIGGER_DEVICE == None:
        raw_input('\n** Warning: No trigger device set. Press Ctrl+C to stop or Enter to continue.')
    trigger = pyLptControl.Trigger(cfg.TRIGGER_DEVICE)
    if trigger.init(50) == False:
        qc.print_c('\n** Error connecting to USB2LPT device. Use a mock trigger instead?', 'R')
        raw_input('Press Ctrl+C to stop or Enter to continue.')
        trigger = pyLptControl.MockTrigger()
        trigger.init(50)

    # init classification
    decoder = BCIDecoderDaemon(cfg.CLS_MI, buffer_size=10.0, fake=(cfg.FAKE_CLS is not None),
                               amp_name=amp_name, amp_serial=amp_serial, fake_dirs=fake_dirs)
    labels = [tdef.by_value[x] for x in decoder.get_labels()]
    bar_def = {label:dir for dir, label in cfg.DIRECTIONS}
    bar_dirs = [bar_def[l] for l in labels]
    dir_seq = []
    for x in range(cfg.TRIALS_EACH):
        dir_seq.extend(bar_dirs)
    random.shuffle(dir_seq)
    num_trials = len(dir_seq)

    qc.print_c('Initializing decoder.', 'W')
    while decoder.is_running() is 0:
        time.sleep(0.01)

    # bar visual object
    from feedback import Feedback

    if cfg.FEEDBACK_TYPE == 'BAR':
        from viz_bars import BarVisual
        visual = BarVisual(cfg.GLASS_USE, screen_pos=cfg.SCREEN_POS, screen_size=cfg.SCREEN_SIZE)
    elif cfg.FEEDBACK_TYPE == 'BODY':
        assert hasattr(cfg, 'IMAGE_PATH'), 'IMAGE_PATH is undefined in your config.'
        from viz_human import BodyVisual
        visual = BodyVisual(cfg.IMAGE_PATH, use_glass=cfg.GLASS_USE, screen_pos=cfg.SCREEN_POS,
                            screen_size=cfg.SCREEN_SIZE)
    visual.put_text('Waiting to start')
    feedback = Feedback(cfg, visual, tdef, trigger)

    # start
    trial = 1
    dir_detected = []
    while trial <= num_trials:
        title_text = 'Trial %d / %d' % (trial, num_trials)
        true_label = dir_seq[trial - 1]
        result = feedback.classify(decoder, true_label, title_text, bar_dirs)

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
                qc.print_c('Warning: Rex cannot execute undefined action %s' % pred_label, 'W')
                rex_dir = None
            if rex_dir is not None:
                visual.move(pred_label, 100, overlay=False, barcolor='B')
                visual.update()
                qc.print_c('Executing Rex action %s' % rex_dir, 'W')
                os.system('%s/Rex/RexControlSimple.exe %s %s' % (pycnbi_config.cnbiroot, cfg.REX_COMPORT, rex_dir))
                time.sleep(8)

        if true_label == pred_label:
            msg = 'Correct'
        else:
            msg = 'Wrong'
        print('Trial %d: %s (%s -> %s)' % (trial, msg, true_label, pred_label))
        trial += 1

    if len(dir_detected) > 0:
        # write performance
        fdir, _, _ = qc.parse_path(cfg.CLS_MI)
        logfile = time.strftime(fdir + "/online-%Y%m%d-%H%M%S.txt", time.localtime())
        with open(logfile, 'w') as fout:
            for dt, gt in zip(dir_detected, dir_seq):
                fout.write('%s,%s\n' % (gt, dt))
            cfmat, acc = qc.confusion_matrix(dir_seq, dir_detected)
            fout.write('\nAccuracy %.3f\nConfusion matrix\n' % acc)
            fout.write(cfmat)
            print('\nAccuracy %.3f\nConfusion matrix\n' % acc)
            print(cfmat)
        print('Log exported to %s' % logfile)

    visual.finish()
    if decoder:
        decoder.stop()

    print('Finished.')
