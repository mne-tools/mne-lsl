from __future__ import print_function
from __future__ import division

'''
Motor imagery training

Kyuhwa Lee, 2018
Chair in Non-invasive Brain-machine Interface Lab (CNBI)
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

'''

import sys
import random
import imp
import cv2
import pycnbi.triggers.pyLptControl as pyLptControl
import pycnbi.utils.q_common as qc
from pycnbi.protocols.viz_human import BodyVisual
from pycnbi.triggers.trigger_def import trigger_def
from pycnbi import logger, init_logger
from builtins import input

def load_config(cfg_file):
    cfg_file = qc.forward_slashify(cfg_file)
    if not (os.path.exists(cfg_file) and os.path.isfile(cfg_file)):
        raise IOError('%s cannot be loaded.' % os.path.realpath(cfg_file))
    return imp.load_source(cfg_file, cfg_file)

def check_config(cfg_module):
    mandatory = {'TRIGGER_DEVICE',
                 'TRIGGER_DEF',
                 'SCREEN_SIZE',
                 'SCREEN_POS',
                 'DIRECTIONS',
                 'DIR_RANDOMIZE',
                 'TRIALS_EACH',
                 'GAIT_STEPS',
                 'T_INIT',
                 'T_GAP',
                 'T_CUE',
                 'T_DIR_READY',
                 'T_DIR',
                 'T_RETURN',
                 'T_STOP',
                 }
    optional = {'FEEDBACK_TYPE':'BAR',
                'T_RETURN':2,
                'DIR_RANDOMIZE':True,
                'WITH_STIMO':False,
                'GLASS_USE':False,
                'REFRESH_RATE':30,
                'T_DIR_RANDOMIZE':0
                }

    for key in mandatory:
        if not hasattr(cfg, key):
            raise ValueError('%s is a required parameter' % key)
    for key in optional:
        if not hasattr(cfg, key):
            logger.warning('Setting undefined %s=%s' % (key, optional[key]))
    return cfg

# for batch script
def batch_run(cfg_file):
    cfg = load_config(cfg_file)
    cfg = check_config(cfg)
    run(cfg)

def run(cfg, *queue):
    
    # ----------------------------------------------------------------------------------
    try:   
        # Redirect stdout and stderr in case of GUI
        sys.stdout = WriteStream(queue[0])
        sys.stderr = WriteStream(queue[0])
        init_logger(sys.stdout)
    except:
        # In case of batch
        init_logger(sys.stdout)  
    # ----------------------------------------------------------------------------------

    tdef = trigger_def(cfg.TRIGGER_FILE)
    refresh_delay = 1.0 / cfg.REFRESH_RATE

    # visualizer
    keys = {'left': 81, 'right': 83, 'up': 82, 'down': 84, 'pgup': 85, 'pgdn': 86, 'home': 80, 'end': 87, 'space': 32,
            'esc': 27, ',': 44, '.': 46, 's': 115, 'c': 99, '[': 91, ']': 93, '1': 49, '!': 33, '2': 50, '@': 64, '3': 51, '#': 35}
    color = dict(G=(20, 140, 0), B=(210, 0, 0), R=(0, 50, 200), Y=(
        0, 215, 235), K=(0, 0, 0), W=(255, 255, 255), w=(200, 200, 200))

    dir_sequence = []
    for x in range(cfg.TRIALS_EACH):
        dir_sequence.extend(cfg.DIRECTIONS)
    random.shuffle(dir_sequence)
    num_trials = len(cfg.DIRECTIONS) * cfg.TRIALS_EACH
    
    state = 'start'
    trial = 1

    # STIMO protocol
    if cfg.WITH_STIMO is True:
        import serial
        logger.info('Opening STIMO serial port (%s / %d bps)' % (cfg.STIMO_COMPORT, cfg.STIMO_BAUDRATE))
        ser = serial.Serial(cfg.STIMO_COMPORT, cfg.STIMO_BAUDRATE)
        logger.info('STIMO serial port %s is_open = %s' % (cfg.STIMO_COMPORT, ser.is_open))

    # init trigger
    if cfg.TRIGGER_DEVICE is None:
        input('\n** Warning: No trigger device set. Press Ctrl+C to stop or Enter to continue.')
    trigger = pyLptControl.Trigger(cfg.TRIGGER_DEVICE)
    if trigger.init(50) == False:
        logger.error('Cannot connect to USB2LPT device. Use a mock trigger instead?')
        input('Press Ctrl+C to stop or Enter to continue.')
        trigger = pyLptControl.MockTrigger()
        trigger.init(50)

    # visual feedback
    if cfg.FEEDBACK_TYPE == 'BAR':
        from pycnbi.protocols.viz_bars import BarVisual
        visual = BarVisual(cfg.GLASS_USE, screen_pos=cfg.SCREEN_POS,
                           screen_size=cfg.SCREEN_SIZE)
    elif cfg.FEEDBACK_TYPE == 'BODY':
        if not hasattr(cfg, 'FEEDBACK_IMAGE_PATH'):
            raise ValueError('FEEDBACK_IMAGE_PATH is undefined in your config.')
        from pycnbi.protocols.viz_human import BodyVisual
        visual = BodyVisual(cfg.FEEDBACK_IMAGE_PATH, use_glass=cfg.GLASS_USE,
                            screen_pos=cfg.SCREEN_POS, screen_size=cfg.SCREEN_SIZE)
    visual.put_text('Waiting to start    ')

    timer_trigger = qc.Timer()
    timer_dir = qc.Timer()
    timer_refresh = qc.Timer()

    # start
    while trial <= num_trials:
        timer_refresh.sleep_atleast(refresh_delay)
        timer_refresh.reset()

        # segment= { 'cue':(s,e), 'dir':(s,e), 'label':0-4 } (zero-based)
        if state == 'start' and timer_trigger.sec() > cfg.T_INIT:
            state = 'gap_s'
            visual.fill()
            timer_trigger.reset()
            trigger.signal(tdef.INIT)
        elif state == 'gap_s':
            visual.put_text('Trial %d / %d' % (trial, num_trials))
            state = 'gap'
        elif state == 'gap' and timer_trigger.sec() > cfg.T_GAP:
            state = 'cue'
            visual.fill()
            visual.draw_cue()
            trigger.signal(tdef.CUE)
            timer_trigger.reset()
        elif state == 'cue' and timer_trigger.sec() > cfg.T_CUE:
            state = 'dir_r'
            dir = dir_sequence[trial-1]
            if dir == 'L':  # left
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move('L', 100)
                else:
                    visual.put_text('LEFT')
                trigger.signal(tdef.LEFT_READY)
            elif dir == 'R':  # right
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move('R', 100)
                else:
                    visual.put_text('RIGHT')
                trigger.signal(tdef.RIGHT_READY)
            elif dir == 'U':  # up
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move('U', 100)
                else:
                    visual.put_text('UP')
                trigger.signal(tdef.UP_READY)
            elif dir == 'D':  # down
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move('D', 100)
                else:
                    visual.put_text('DOWN')
                trigger.signal(tdef.DOWN_READY)
            elif dir == 'B':  # both hands
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move('L', 100)
                    visual.move('R', 100)
                else:
                    visual.put_text('BOTH')
                trigger.signal(tdef.BOTH_READY)
            else:
                raise RuntimeError('Unknown direction %d' % dir)
            gait_steps = 1
            timer_trigger.reset()
        elif state == 'dir_r' and timer_trigger.sec() > cfg.T_DIR_READY:
            visual.draw_cue()
            state = 'dir'
            timer_trigger.reset()
            timer_dir.reset()
            t_step = cfg.T_DIR + random.random() * cfg.T_DIR_RANDOMIZE
            if dir == 'L':  # left
                trigger.signal(tdef.LEFT_GO)
            elif dir == 'R':  # right
                trigger.signal(tdef.RIGHT_GO)
            elif dir == 'U':  # up
                trigger.signal(tdef.UP_GO)
            elif dir == 'D':  # down
                trigger.signal(tdef.DOWN_GO)
            elif dir == 'B':  # both
                trigger.signal(tdef.BOTH_GO)
            else:
                raise RuntimeError('Unknown direction %d' % dir)
        elif state == 'dir':
            if timer_trigger.sec() > t_step:
                if cfg.FEEDBACK_TYPE == 'BODY':
                    if cfg.WITH_STIMO is True:
                        if dir == 'L':  # left
                            ser.write(b'1')
                            logger.info('STIMO: Sent 1')
                            trigger.signal(tdef.LEFT_STIMO)
                        elif dir == 'R':  # right
                            ser.write(b'2')
                            logger.info('STIMO: Sent 2')
                            trigger.signal(tdef.RIGHT_STIMO)
                    else:
                        if dir == 'L':  # left
                            trigger.signal(tdef.LEFT_RETURN)
                        elif dir == 'R':  # right
                            trigger.signal(tdef.RIGHT_RETURN)
                else:
                    trigger.signal(tdef.FEEDBACK)
                state = 'return'
                timer_trigger.reset()
            else:
                dx = min(100, int(100.0 * timer_dir.sec() / t_step) + 1)
                if dir == 'L':  # L
                    visual.move('L', dx, overlay=True)
                elif dir == 'R':  # R
                    visual.move('R', dx, overlay=True)
                elif dir == 'U':  # U
                    visual.move('U', dx, overlay=True)
                elif dir == 'D':  # D
                    visual.move('D', dx, overlay=True)
                elif dir == 'B':  # Both
                    visual.move('L', dx, overlay=True)
                    visual.move('R', dx, overlay=True)
        elif state == 'return':
            if timer_trigger.sec() > cfg.T_RETURN:
                if gait_steps < cfg.GAIT_STEPS:
                    gait_steps += 1
                    state = 'dir'
                    visual.move('L', 0)
                    if dir == 'L':
                        dir = 'R'
                        trigger.signal(tdef.RIGHT_GO)
                    else:
                        dir = 'L'
                        trigger.signal(tdef.LEFT_GO)
                    timer_dir.reset()
                    t_step = cfg.T_DIR + random.random() * cfg.T_DIR_RANDOMIZE
                else:
                    state = 'gap_s'
                    visual.fill()
                    trial += 1
                    logger.info('Trial ' + str(trial-1) + ' done.')
                    trigger.signal(tdef.BLANK)
                timer_trigger.reset()
            else:
                dx = max(0, int(100.0 * (cfg.T_RETURN - timer_trigger.sec()) / cfg.T_RETURN))
                if dir == 'L':  # L
                    visual.move('L', dx, overlay=True)
                elif dir == 'R':  # R
                    visual.move('R', dx, overlay=True)
                elif dir == 'U':  # U
                    visual.move('U', dx, overlay=True)
                elif dir == 'D':  # D
                    visual.move('D', dx, overlay=True)
                elif dir == 'B':  # Both
                    visual.move('L', dx, overlay=True)
                    visual.move('R', dx, overlay=True)

        # wait for start
        if state == 'start':
            visual.put_text('Waiting to start    ')

        visual.update()
        key = 0xFF & cv2.waitKey(1)

        if key == keys['esc']:
            break

    # STIMO protocol
    if cfg.WITH_STIMO is True:
        ser.close()
        logger.info('Closed STIMO serial port %s' % cfg.STIMO_COMPORT)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        cfg_module = input('Config file name? ')
    else:
        cfg_module = sys.argv[1]
    config_run(cfg_module)
