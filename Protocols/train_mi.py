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
from viz_bars import Bars


if __name__=='__main__':
	if len(sys.argv) < 2:
		cfg_module= raw_input('Config file name? ')
	else:
		cfg_module= sys.argv[1]
	cfg= imp.load_source(cfg_module, cfg_module)
	tdefmod= importlib.import_module( cfg.TRIGGER_DEF )
	refresh_delay= 1.0 / cfg.REFRESH_RATE

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

	# Hardware trigger
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

	bar= Bars(cfg.GLASS_USE, screen_pos=cfg.SCREEN_POS, screen_size=cfg.SCREEN_SIZE)
	bar.fill()
	bar.glass_draw_cue()

	# start
	while trial <= num_trials:
		timer_refresh.sleep_atleast(refresh_delay)
		timer_refresh.reset()

		# segment= { 'cue':(s,e), 'dir':(s,e), 'label':0-4 } (zero-based)
		if event=='start' and timer_trigger.sec() > cfg.T_INIT:
			event= 'gap_s'
			bar.fill()
			timer_trigger.reset()
			trigger.signal(tdef.INIT)
		elif event=='gap_s':
			bar.putText('Trial %d / %d'%(trial,num_trials) )
			event= 'gap'
		elif event=='gap' and timer_trigger.sec() > cfg.T_GAP:
			event= 'cue'
			bar.fill()
			bar.draw_cue()
			trigger.signal(tdef.CUE)
			timer_trigger.reset()
		elif event=='cue' and timer_trigger.sec() > cfg.T_CUE:
			event= 'dir_r'
			dir= dir_sequence[trial-1]
			if dir=='L': # left
				bar.move( 'L', 100, overlay=True )
				trigger.signal(tdef.LEFT_READY)
			elif dir=='R': # right
				bar.move( 'R', 100, overlay=True )
				trigger.signal(tdef.RIGHT_READY)
			elif dir=='U': # up
				bar.move( 'U', 100, overlay=True )
				trigger.signal(tdef.UP_READY)
			elif dir=='D': # down
				bar.move( 'D', 100, overlay=True )
				trigger.signal(tdef.DOWN_READY)
			elif dir=='B': # both hands
				bar.move( 'L', 100, overlay=True )
				bar.move( 'R', 100, overlay=True )
				trigger.signal(tdef.BOTH_READY)
			else:
				raise RuntimeError('Unknown direction %d' % dir)
			timer_trigger.reset()
		elif event=='dir_r' and timer_trigger.sec() > cfg.T_DIR_READY:
			bar.fill()
			bar.draw_cue()
			event= 'dir'
			timer_trigger.reset()
			timer_dir.reset()
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
				raise RuntimeError('Unknown direction %d' % dir)
		elif event=='dir' and timer_trigger.sec() > cfg.T_DIR:
			event= 'gap_s'
			bar.fill()
			trial += 1
			print('trial '+str(trial-1)+' done')
			trigger.signal(tdef.BLANK)
			timer_trigger.reset()

		# protocol
		if event=='dir':
			dx= min( 100, int( 100.0 * timer_dir.sec() / cfg.T_DIR ) + 1 )
			if dir=='L': # L
				bar.move( 'L', dx, overlay=True )
			elif dir=='R': # R
				bar.move( 'R', dx, overlay=True )
			elif dir=='U': # U
				bar.move( 'U', dx, overlay=True )
			elif dir=='D': # D
				bar.move( 'D', dx, overlay=True )
			elif dir=='B': # Both
				bar.move( 'L', dx, overlay=True )
				bar.move( 'R', dx, overlay=True )

		# wait for start
		if event=='start':
			bar.putText('Waiting to start')
		
		bar.update()
		key= 0xFF & cv2.waitKey(1)

		if key==keys['esc']:
			break
