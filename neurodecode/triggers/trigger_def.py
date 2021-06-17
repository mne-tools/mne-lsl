from pathlib import Path
from configparser import ConfigParser

from .. import logger


class TriggerDef:
    """
    Class for reading event's pairs (string-integer) from ini file.

    The class will also have as attributes self.event_str = event_int for all
    pairs.

    Parameters
    ----------
    ini_file : str
        The path of the ini file
    """

    def __init__(self, ini_file):
        self._ini_file = Path(ini_file)
        TriggerDef._check_ini_path(self._ini_file)

        self._by_name = dict()
        self._by_value = dict()
        self._extract_from_ini()

    def _extract_from_ini(self):
        """
        Load the .ini file.
        """
        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.optionxform = str
        config.read(str(self._ini_file))
        self._create_attributes(config.items('events'))

    def _create_attributes(self, items):
        """
        Fill the class attributes with the pairs string-integer.
        """
        for key, value in items:
            value = int(value)
            setattr(self, key, value)
            self._by_name[key] = value
            self._by_value[value] = key

    # --------------------------------------------------------------------
    @staticmethod
    def _check_ini_path(ini_file):
        """
        Checks that the provided file exists and ends with .ini.
        """
        if ini_file.exists():
            logger.info(f"Found trigger definition file '{ini_file}'")
        else:
            logger.error(
                f"Trigger event definition file '{ini_file}' not found.")
            raise IOError

        if ini_file.suffix != '.ini':
            logger.error(
                "Trigger event definition file format must be '.ini'.")
            raise IOError

    # --------------------------------------------------------------------
    @property
    def by_name(self):
        """
        A dictionnary with string keys and integers value.
        """
        return self._by_name

    @by_name.setter
    def by_name(self, new):
        logger.warning("Cannot modify this attribute manually.")

    @property
    def by_value(self):
        """
        A dictionnary with integers keys and string values.
        """
        return self._by_value

    @by_value.setter
    def by_value(self, new):
        logger.warning("Cannot modify this attribute manually.")
