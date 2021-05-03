from __future__ import print_function, division

import os
from configparser import ConfigParser
from neurodecode import logger


class TriggerDef(object):
    """
    Class for reading event's pairs (string-integer) from ini file.
    
    The class will also have as attributes self.event_str = event_int for all pairs.
    
    Parameters
    ----------
    ini_file : str
        The path of the ini file
    """
    
    #----------------------------------------------------------------------
    def __init__(self, ini_file):
        self._by_name = {}
        self._by_value = {}
        
        self._check_ini_path(ini_file)
        self._extract_from_ini(ini_file)

    #----------------------------------------------------------------------
    def _check_ini_path(self, ini_file):
        """
        Ensure that the provided file exists.
        
        Parameters
        ----------
        ini_file : str
            The absolute path of the ini file
        """
        if os.path.exists(ini_file):
            logger.info('Found trigger definition file %s' % ini_file)
        else:
            raise IOError('Trigger event definition file %s not found' % ini_file)        
    
    #----------------------------------------------------------------------    
    def _extract_from_ini(self, ini_file):
        """
        Extract the events' name and associated integer.
        """
        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.optionxform = str
        config.read(ini_file)
        self._create_attributes(config.items('events'))
        
    #----------------------------------------------------------------------
    def _create_attributes(self, items):
        """
        Fill the class attributes with the pairs string-integer
        """
        for key, value in items:
            setattr(self, key, int(value))
            self._by_name[key] = value
            self._by_value[value] = key

    #----------------------------------------------------------------------
    def check_data(self):
        """
        Display all attributes.
        """
        print('TriggerDef Attributes:')
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                print(attr, getattr(self, attr))

    #----------------------------------------------------------------------
    @property
    def by_name(self):
        """
        A dictionnary with string keys and integers value
        """
        return self._by_name
    
    #----------------------------------------------------------------------
    @by_name.setter
    def by_name(self, new):
        logger.warning("Cannot modify this attribute manually.")
        
    #----------------------------------------------------------------------
    @property
    def by_value(self):
        """
        A dictionnary with integers keys and string values
        """
        return self._by_value
    
    #----------------------------------------------------------------------
    @by_value.setter
    def by_value(self, new):
        logger.warning("Cannot modify this attribute manually.")
    
# example
if __name__ == '__main__':
    ini_file = './triggerdef_template.ini'
    tdef = TriggerDef(ini_file)
    tdef.check_data()
