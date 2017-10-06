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
    LS = 1  # left start
    LE = 2  # left end
    RS = 3  # right start
    RE = 4  # right end
    LSB = 11  # left start of a block
    LEB = 12  # left end of a block
    RSB = 13  # right start of a block
    REB = 14  # right end of a block
    SB = 21  # block start
    EB = 22  # block end
    S = 31  # left or right start
    E = 32  # left or right end
    INIT = 101  # start of a trial
    END = 102  # end of a trial
    VICON = 128  # Vicon sync event
    OTHERS = 255  # undefined


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
