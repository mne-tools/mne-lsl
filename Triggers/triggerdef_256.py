from __future__ import print_function, division

'''
Trigger definition class

'keys' and 'values' member variables are automatically created when instantiated.

Usage: See the sample code


Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)

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

from triggerdef_template import TriggerDefTemplate

# Trigger values up to 255
class TriggerDef(TriggerDefTemplate):
	INIT= 255 # start of the trial
	CUE= 254 # cue shown
	BEEP= 253 # beep played
	BLANK= 252 # screen turned into blank
	HIT=251 # online correct
	MISS=250 # online wrong
	LEFT_READY= 249 # left bar shown
	LEFT_GO= 248 # left bar started moving
	RIGHT_READY= 247 # right bar shown
	RIGHT_GO= 246 # right started moving
	UP_READY= 245 # up bar shown
	UP_GO= 244 # up started moving
	DOWN_READY= 243 # down bar shown
	DOWN_GO= 242 # down started moving
	STAND_READY= 241 # stand message shown
	STAND_GO= 240 # start standing
	SIT_READY= 239 # sit message shown
	SIT_GO= 238 # start sitting
	WALK_READY= 237 # walk message shown
	WALK_GO= 236 # start walking
	WALK2STAND_READY= 235 # prepare to stop
	WALK2STAND_GO= 234 # stop now


# sample code
if __name__=='__main__':
	tdef= TriggerDef()

	# accessing a trigger value as a member variable
	print( 'INIT =', tdef.INIT )

	# check whether the trigger name is defined
	print( '\nINIT in tdef.by_key?')
	print( 'INIT' in tdef.by_key )

	# check whether the trigger value is defined
	print( '\n255 in tdef.by_value?' )
	print( 255 in tdef.by_value )

	# print all trigger names and associated values
	print( '\ntdef.by_key' )
	print( tdef.by_key )

	# print all trigger values and associated names
	print( '\ntdef.by_value' )
	print( tdef.by_value )
