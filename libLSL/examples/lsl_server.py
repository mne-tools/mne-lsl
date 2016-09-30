from __future__ import print_function, division

import pycnbi_config
import pylsl as lsl
import time, random
import cnbi_lsl

LSL_SERVER= 'RexController'
vals= [3, 5, 6, 7]

outlet= cnbi_lsl.start_server('RexController', channel_format='double64')
print(LSL_SERVER, 'server start.')

while True:
	data= random.choice( vals )
	raw_input('Press Enter to send data %d '% data)
	print('Sending data', data)
	outlet.push_sample([data])
