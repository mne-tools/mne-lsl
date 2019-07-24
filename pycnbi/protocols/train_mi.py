from __future__ import print_function, division

"""
Motor imagery training

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

import sys
import os
import time
import imp
import cv2
import random
import pycnbi.triggers.pyLptControl as pyLptControl
import pycnbi.utils.q_common as qc
from pycnbi.protocols.viz_bars import BarVisual
from pycnbi.triggers.trigger_def import trigger_def
from pycnbi import logger
from builtins import input

from pycnbi.gui.streams import redirect_stdout_to_queue

def load_config(cfg_file):
    cfg_file = qc.forward_slashify(cfg_file)
    if not (os.path.exists(cfg_file) and os.path.isfile(cfg_file)):
        raise IOError('%s cannot be loaded.' % os.path.realpath(cfg_file))
    return imp.load_source(cfg_file, cfg_file)

def check_config(cfg):
    critical_vars = {
        'COMMON': ['TRIGGER_DEVICE',
                   'TRIGGER_FILE', 
                   'SCREEN_SIZE',
                   'DIRECTIONS',
                   'DIR_RANDOM',
                   'TRIALS_EACH'], 
        'TIMINGS': ['INIT', 'GAP', 'CUE', 'READY', 'READY_RANDOMIZE', 'DIR', 'DIR_RANDOMIZE']
    }
    optional_vars = {'FEEDBACK_TYPE': 'BAR',
                     'FEEDBACK_IMAGE_PATH': None,
                     'SCREEN_POS': (0, 0),
                    'DIR_RANDOM':True,
                    'GLASS_USE':False,
                    'TRIAL_PAUSE': False,
                    'REFRESH_RATE':30
    }

    for key in critical_vars['COMMON']:
        if not hasattr(cfg, key):
            raise RuntimeError('%s is a required parameter' % key)
            
    if not hasattr(cfg, 'TIMINGS'):
        logger.error('"TIMINGS" not defined in config.')
        raise RuntimeError
    for v in critical_vars['TIMINGS']:
        if v not in cfg.TIMINGS:
            logger.error('%s not defined in config.' % v)
            raise RuntimeError            

    for key in optional_vars:
        if not hasattr(cfg, key):
            setattr(cfg, key, optional_vars[key])
            logger.warning('Setting undefined %s=%s' % (key, optional[key]))

    return cfg

# for batch script
def batch_run(cfg_file):
    cfg = load_config(cfg_file)
    cfg = check_config(cfg)
    run(cfg)

def run(cfg, queue=None):
    
    redirect_stdout_to_queue(queue)    
    refresh_delay = 1.0 / cfg.REFRESH_RATE
    
    cfg.tdef = trigger_def(cfg.TRIGGER_FILE)

    # visualizer
    keys = {'left':81, 'right':83, 'up':82, 'down':84, 'pgup':85, 'pgdn':86,
        'home':80, 'end':87, 'space':32, 'esc':27, ',':44, '.':46, 's':115, 'c':99, 
        '[':91, ']':93, '1':49, '!':33, '2':50, '@':64, '3':51, '#':35}
    color = dict(G=(20, 140, 0), B=(210, 0, 0), R=(0, 50, 200), Y=(0, 215, 235),
        K=(0, 0, 0), w=(200, 200, 200))

    dir_sequence = []
    for x in range(cfg.TRIALS_EACH):
        dir_sequence.extend(cfg.DIRECTIONS)
    random.shuffle(dir_sequence)
    num_trials = len(cfg.DIRECTIONS) * cfg.TRIALS_EACH

    event = 'start'
    trial = 1

    # Hardware trigger
    if cfg.TRIGGER_DEVICE is None:
        logger.warning('No trigger device set. Press Ctrl+C to stop or Enter to continue.')
        #input()
    trigger = pyLptControl.Trigger(cfg.TRIGGER_DEVICE)
    if trigger.init(50) == False:
        logger.error('\n** Error connecting to USB2LPT device. Use a mock trigger instead?')
        input('Press Ctrl+C to stop or Enter to continue.')
        trigger = pyLptControl.MockTrigger()
        trigger.init(50)

    # timers
    timer_trigger = qc.Timer()
    timer_dir = qc.Timer()
    timer_refresh = qc.Timer()
    t_dir = cfg.TIMINGS['DIR'] + random.uniform(-cfg.TIMINGS['DIR_RANDOMIZE'], cfg.TIMINGS['DIR_RANDOMIZE'])
    t_dir_ready = cfg.TIMINGS['READY'] + random.uniform(-cfg.TIMINGS['READY_RANDOMIZE'], cfg.TIMINGS['READY_RANDOMIZE'])

    bar = BarVisual(cfg.GLASS_USE, screen_pos=cfg.SCREEN_POS, screen_size=cfg.SCREEN_SIZE)
    bar.fill()
    bar.glass_draw_cue()

    # start
    while trial <= num_trials:
        timer_refresh.sleep_atleast(refresh_delay)
        timer_refresh.reset()

        # segment= { 'cue':(s,e), 'dir':(s,e), 'label':0-4 } (zero-based)
        if event == 'start' and timer_trigger.sec() > cfg.TIMINGS['INIT']:
            event = 'gap_s'
            bar.fill()
            timer_trigger.reset()
            trigger.signal(cfg.tdef.INIT)
        elif event == 'gap_s':
            if cfg.TRIAL_PAUSE:
                bar.put_text('Press any key')
                bar.update()
                key = cv2.waitKey()
                if key == keys['esc']:
                    break
                bar.fill()
            bar.put_text('Trial %d / %d' % (trial, num_trials))
            event = 'gap'
            timer_trigger.reset()
        elif event == 'gap' and timer_trigger.sec() > cfg.TIMINGS['GAP']:
            event = 'cue'
            bar.fill()
            bar.draw_cue()
            trigger.signal(cfg.tdef.CUE)
            timer_trigger.reset()
        elif event == 'cue' and timer_trigger.sec() > cfg.TIMINGS['CUE']:
            event = 'dir_r'
            dir = dir_sequence[trial - 1]
            if dir == 'L':  # left
                bar.move('L', 100, overlay=True)
                trigger.signal(cfg.tdef.LEFT_READY)
            elif dir == 'R':  # right
                bar.move('R', 100, overlay=True)
                trigger.signal(cfg.tdef.RIGHT_READY)
            elif dir == 'U':  # up
                bar.move('U', 100, overlay=True)
                trigger.signal(cfg.tdef.UP_READY)
            elif dir == 'D':  # down
                bar.move('D', 100, overlay=True)
                trigger.signal(cfg.tdef.DOWN_READY)
            elif dir == 'B':  # both hands
                bar.move('L', 100, overlay=True)
                bar.move('R', 100, overlay=True)
                trigger.signal(cfg.tdef.BOTH_READY)
            else:
                raise RuntimeError('Unknown direction %d' % dir)
            timer_trigger.reset()
        elif event == 'dir_r' and timer_trigger.sec() > t_dir_ready:
            bar.fill()
            bar.draw_cue()
            event = 'dir'
            timer_trigger.reset()
            timer_dir.reset()
            if dir == 'L':  # left
                trigger.signal(cfg.tdef.LEFT_GO)
            elif dir == 'R':  # right
                trigger.signal(cfg.tdef.RIGHT_GO)
            elif dir == 'U':  # up
                trigger.signal(cfg.tdef.UP_GO)
            elif dir == 'D':  # down
                trigger.signal(cfg.tdef.DOWN_GO)
            elif dir == 'B':  # both
                trigger.signal(cfg.tdef.BOTH_GO)
            else:
                raise RuntimeError('Unknown direction %d' % dir)
        elif event == 'dir' and timer_trigger.sec() > t_dir:
            event = 'gap_s'
            bar.fill()
            trial += 1
            logger.info('trial ' + str(trial - 1) + ' done')
            trigger.signal(cfg.tdef.BLANK)
            timer_trigger.reset()
            t_dir = cfg.TIMINGS['DIR'] + random.uniform(-cfg.TIMINGS['DIR_RANDOMIZE'], cfg.TIMINGS['DIR_RANDOMIZE'])
            t_dir_ready = cfg.TIMINGS['READY'] + random.uniform(-cfg.TIMINGS['READY_RANDOMIZE'], cfg.TIMINGS['READY_RANDOMIZE'])

        # protocol
        if event == 'dir':
            dx = min(100, int(100.0 * timer_dir.sec() / t_dir) + 1)
            if dir == 'L':  # L
                bar.move('L', dx, overlay=True)
            elif dir == 'R':  # R
                bar.move('R', dx, overlay=True)
            elif dir == 'U':  # U
                bar.move('U', dx, overlay=True)
            elif dir == 'D':  # D
                bar.move('D', dx, overlay=True)
            elif dir == 'B':  # Both
                bar.move('L', dx, overlay=True)
                bar.move('R', dx, overlay=True)

        # wait for start
        if event == 'start':
            bar.put_text('Waiting to start')

        bar.update()
        key = 0xFF & cv2.waitKey(1)
        if key == keys['esc']:
            break

    bar.finish()
   

if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_file = input('Config file name? ')
    else:
        cfg_file = sys.argv[1]
    batch_run(cfg_file)
