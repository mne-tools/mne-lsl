from __future__ import print_function
from __future__ import division

'''
Motor imagery training

Kyuhwa Lee, 2014
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
from viz_human import BodyVisual


def check_cfg(cfg):
    if not hasattr(cfg, 'FEEDBACK_TYPE'):
        qc.print_c('Warning: FEEDBACK_TYPE undefined. Setting it to BAR.', 'Y')
        cfg.FEEDBACK_TYPE = 'BAR'
    if not hasattr(cfg, 'T_LEG_RETURN'):
        qc.print_c('Warning: T_LEG_RETURN undefined. Setting it to 2.', 'Y')
        cfg.T_LEG_RETURN = 2
    return cfg


if __name__=='__main__':
    if len(sys.argv) < 2:
        cfg_module= raw_input('Config file name? ')
    else:
        cfg_module= sys.argv[1]
    cfg= imp.load_source(cfg_module, cfg_module)
    tdefmod= importlib.import_module( cfg.TRIGGER_DEF )
    refresh_delay= 1.0 / cfg.REFRESH_RATE
    cfg = check_cfg(cfg)

    # visualizer
    keys= {'left':81,'right':83,'up':82,'down':84,'pgup':85,'pgdn':86,'home':80,'end':87,'space':32,'esc':27\
        ,',':44,'.':46,'s':115,'c':99,'[':91,']':93,'1':49,'!':33,'2':50,'@':64,'3':51,'#':35}
    color= dict(G=(20,140,0), B=(210,0,0), R=(0,50,200), Y=(0,215,235), K=(0,0,0), W=(255,255,255), w=(200,200,200))

    dir_sequence= []
    for x in range( cfg.TRIALS_EACH ):
        dir_sequence.extend( cfg.DIRECTIONS )
    random.shuffle( dir_sequence )

    num_trials= len(cfg.DIRECTIONS) * cfg.TRIALS_EACH

    event= 'start'
    trial= 1

    # connect trigger
    trigger= pyLptControl.Trigger(cfg.TRIGGER_DEVICE)
    if trigger.init(50)==False:
        print('\n# Error connecting to USB2LPT device. Use a mock trigger instead?')
        raw_input('Press Ctrl+C to stop or Enter to continue.')
        trigger= pyLptControl.MockTrigger()
        trigger.init(50)

    timer_trigger= qc.Timer()
    timer_dir= qc.Timer()
    timer_refresh= qc.Timer()
    tdef= tdefmod.TriggerDef()

    # visual feedback
    if cfg.FEEDBACK_TYPE == 'BAR':
        from viz_bars import BarVisual
        visual = BarVisual(cfg.GLASS_USE, screen_pos=cfg.SCREEN_POS,
            screen_size=cfg.SCREEN_SIZE)
    elif cfg.FEEDBACK_TYPE == 'BODY':
        assert hasattr(cfg, 'IMAGE_PATH'), 'IMAGE_PATH is undefined in your config.'
        from viz_human import BodyVisual
        visual = BodyVisual(cfg.IMAGE_PATH, use_glass=cfg.GLASS_USE,
            screen_pos=cfg.SCREEN_POS, screen_size=cfg.SCREEN_SIZE)
    visual.put_text('Waiting to start    ')

    # start
    while trial <= num_trials:
        timer_refresh.sleep_atleast(refresh_delay)
        timer_refresh.reset()

        # segment= { 'cue':(s,e), 'dir':(s,e), 'label':0-4 } (zero-based)
        if event=='start' and timer_trigger.sec() > cfg.T_INIT:
            event= 'gap_s'
            visual.fill()
            timer_trigger.reset()
            trigger.signal(tdef.INIT)
        elif event=='gap_s':
            visual.put_text('Trial %d / %d'%(trial,num_trials) )
            event= 'gap'
        elif event=='gap' and timer_trigger.sec() > cfg.T_GAP:
            event= 'cue'
            visual.fill()
            visual.draw_cue()
            trigger.signal(tdef.CUE)
            timer_trigger.reset()
        elif event=='cue' and timer_trigger.sec() > cfg.T_CUE:
            event= 'dir_r'
            dir= dir_sequence[trial-1]
            if dir == 'L':  # left
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move( 'L', 100 )
                else:
                    visual.put_text('LEFT')
                trigger.signal(tdef.LEFT_READY)
            elif dir == 'R':  # right
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move( 'R', 100 )
                else:
                    visual.put_text('RIGHT')
                trigger.signal(tdef.RIGHT_READY)
            elif dir == 'U':  # up
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move( 'U', 100 )
                else:
                    visual.put_text('UP')
                trigger.signal(tdef.UP_READY)
            elif dir == 'D':  # down
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move( 'D', 100 )
                else:
                    visual.put_text('DOWN')
                trigger.signal(tdef.DOWN_READY)
            elif dir == 'B':  # both hands
                if cfg.FEEDBACK_TYPE == 'BAR':
                    visual.move( 'L', 100 )
                    visual.move( 'R', 100 )
                else:
                    visual.put_text('BOTH')
                trigger.signal(tdef.BOTH_READY)
            else:
                raise RuntimeError('Unknown direction %d' % dir)
            gait_steps= 1
            timer_trigger.reset()
        elif event=='dir_r' and timer_trigger.sec() > cfg.T_DIR_READY:
            visual.draw_cue()
            event= 'dir'
            timer_trigger.reset()
            timer_dir.reset()
            t_step= cfg.T_DIR + random.random() * cfg.RANDOMIZE_LENGTH
            if dir=='L': # left
                trigger.signal(tdef.LEFT_GO)
            elif dir=='R': # right
                trigger.signal(tdef.RIGHT_GO)
            elif dir=='U': # up
                trigger.signal(tdef.UP_GO)
            elif dir=='D': # down
                trigger.signal(tdef.DOWN_GO)
            elif dir=='B': # both
                trigger.signal(tdef.BOTH_GO)
            else:
                raise RuntimeError, 'Unknown direction %d' % dir
        elif event=='dir' and timer_trigger.sec() > t_step:
            if cfg.FEEDBACK_TYPE == 'BODY':
                # return to standing position
                trigger.signal(tdef.FEEDBACK)
                timer_ret = qc.Timer()
                while timer_ret.sec() < cfg.T_LEG_RETURN: 
                    dx = int(100 - timer_ret.sec() * (100.0 / cfg.T_LEG_RETURN))
                    visual.move( dir, dx, overlay=True )
            if gait_steps < cfg.GAIT_STEPS:
                gait_steps += 1
                event= 'dir'
                visual.move( 'L', 0 )
                if dir=='L':
                    dir= 'R'
                    trigger.signal(tdef.RIGHT_GO)
                else:
                    dir= 'L'
                    trigger.signal(tdef.LEFT_GO)
                timer_dir.reset()
                timer_trigger.reset()
                t_step= cfg.T_DIR + random.random() * cfg.RANDOMIZE_LENGTH
            else:
                event= 'gap_s'
                visual.fill()
                trial += 1
                print('trial '+str(trial-1)+' done')
                trigger.signal(tdef.BLANK)
                timer_trigger.reset()

        # protocol
        if event=='dir':
            dx= min( 100, int( 100.0 * timer_dir.sec() / t_step ) + 1 )
            if dir=='L': # L
                visual.move( 'L', dx, overlay=True )
            elif dir=='R': # R
                visual.move( 'R', dx, overlay=True )
            elif dir=='U': # U
                visual.move( 'U', dx, overlay=True )
            elif dir=='D': # D
                visual.move( 'D', dx, overlay=True )
            elif dir=='B': # Both
                visual.move( 'L', dx, overlay=True )
                visual.move( 'R', dx, overlay=True )

        # wait for start
        if event=='start':
            visual.put_text('Waiting to start    ')
        
        visual.update()
        key= 0xFF & cv2.waitKey(1)

        if key==keys['esc']:
            break
