from __future__ import print_function, division

'''
CNBI MI events

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


class TriggerDef(TriggerDefTemplate):
    INIT = 1  # start of the trial
    FIXATION = 786
    LEFT_READY = 769
    RIGHT_READY = 770
    FEET_READY = 771
    # REST_READY= 772
    GO = 781  # left bar started moving
    MISS = 898
    HIT = 897

    def __init__(self):
        for attr in dir(self):
            if hasattr(self, 'keys') == False:
                self.keys = {}
                self.values = {}
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                # print(attr, getattr(TriggerDef,attr))
                self.keys[attr] = getattr(TriggerDef, attr)
                self.values[getattr(TriggerDef, attr)] = attr


# sample code
if __name__ == '__main__':
    tdef = TriggerDef()

    # accessing a trigger value as a member variable
    print(tdef.INIT)

    # check whether the trigger name is defined
    print('INIT' in tdef.keys)

    # check whether the trigger value is defined
    print(786 in tdef.values)

    # print all trigger names and associated values
    print(tdef.keys)
    print()

    # print all trigger values and associated names
    print(tdef.values)
    print()
