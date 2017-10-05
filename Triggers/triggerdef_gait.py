from __future__ import print_function, division

'''
Gait events

'by_key' and 'values' member variables are automatically created when instantiated.

Usage: See the sample code


Kyuhwa Lee, 2015
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


class TriggerDef(TriggerDefTemplate):
    """
    Trigger definition class

    'by_key' and 'values' member variables are automatically created when instantiated.

    Usage: See the sample code
    """
    INIT = 1  # start of a trial
    END = 2  # end of a trial
    LO = 10  # left foot off the ground
    RO = 11  # right foot off the ground
    LT = 12  # left foot touch the ground
    RT = 13  # right foot touch the ground
    LOS = 14  # initiate gait cycle with left foot
    ROS = 15  # initiate gait cycle with right foot
    LTE = 16  # terminate gait cycle with left foot
    RTE = 17  # terminate gait cycle with right foot
    OS = 18  # initiate gait cycle with any foot
    TE = 19  # terminate gait cycle with any foot
    O = 20  # any foot off the ground
    T = 21  # any foot touch the ground
    VICON = 128  # all other classes
    OTHERS = 255  # all other classes


# sample code
if __name__ == '__main__':
    tdef = TriggerDef()

    # accessing a trigger value as a member variable
    print('INIT =', tdef.INIT)

    # check whether the trigger name is defined
    print('\nINIT in tdef.by_key?')
    print('INIT' in tdef.by_key)

    # check whether the trigger value is defined
    print('\n255 in tdef.by_value?')
    print(255 in tdef.by_value)

    # print all trigger names and associated values
    print('\ntdef.by_key')
    print(tdef.by_key)

    # print all trigger values and associated names
    print('\ntdef.by_value')
    print(tdef.by_value)
