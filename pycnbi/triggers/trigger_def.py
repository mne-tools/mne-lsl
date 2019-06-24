from __future__ import print_function, division

"""
Reads trigger info and creates a class object with the follwing attributes:
- self.event_id = event_value
- self.by_name() = {key:value, ...}
- self.by_value() = {value:key, ...}

"""

import sys
import os
import pycnbi.utils.q_common as qc
from configparser import ConfigParser
from pycnbi import logger

def trigger_def(ini_file, verbose=False):
    class TriggerDef(object):
        def __init__(self, items):
            self.by_name = {}
            self.by_value = {}
            for key, value in items:
                value = int(value)
                setattr(self, key, value)
                self.by_name[key] = value
                self.by_value[value] = key

        # show all possible trigger values
        def check_data(self):
            print('Attributes of the final class')
            for attr in dir(self):
                if not callable(getattr(self, attr)) and not attr.startswith("__"):
                    print(attr, getattr(self, attr))

    if not os.path.exists(ini_file):
        search_path = []
        path_ini = qc.parse_path(ini_file)
        path_self = qc.parse_path(__file__)
        search_path.append(ini_file + '.ini')
        search_path.append('%s/%s' % (path_self.dir, path_ini.name))
        search_path.append('%s/%s.ini' % (path_self.dir, path_ini.name))
        for ini_file in search_path:
            if os.path.exists(ini_file):
                if verbose:
                    logger.info('Found trigger definition file %s' % ini_file)
                break
        else:
            raise IOError('Trigger event definition file %s not found' % ini_file)
    config = ConfigParser(inline_comment_prefixes=('#', ';'))
    config.optionxform = str
    config.read(ini_file)
    return TriggerDef(config.items('events'))

# example
if __name__ == '__main__':
    ini_file = 'triggerdef_16.ini'
    tdef = trigger_def(ini_file)

    # accessing a trigger value as a member variable
    print('INIT =', tdef.INIT)

    # check whether the trigger name is defined
    print('\nINIT in tdef.by_name?')
    print('INIT' in tdef.by_name)

    # check whether the trigger value is defined
    print('\n255 in tdef.by_value?')
    print(255 in tdef.by_value)
    print('\n1 in tdef.by_value?')
    print(1 in tdef.by_value)

    # print all trigger names and associated values
    print('\ntdef.by_name')
    print(tdef.by_name)

    # print all trigger values and associated names
    print('\ntdef.by_value')
    print(tdef.by_value)
