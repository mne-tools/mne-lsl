from pathlib import Path
from configparser import ConfigParser

from .. import logger
from ..utils._docs import fill_doc


@fill_doc
class TriggerDef:
    """
    Class for reading event's pairs (string-integer) from ini file.

    The class will also have as attributes ``self.event_str = event_int`` for
    all pairs.

    Parameters
    ----------
    %(trigger_file)s
    """

    def __init__(self, trigger_file=None):
        self._trigger_file = TriggerDef._check_trigger_file(trigger_file)

        self._by_name = dict()
        self._by_value = dict()
        if self._trigger_file is not None:
            self._extract_from_ini()

    def _extract_from_ini(self):
        """
        Load the ``.ini`` file.
        """
        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.optionxform = str
        config.read(str(self._trigger_file))

        for key, value in config.items('events'):
            value = int(value)
            setattr(self, key, value)
            self._by_name[key] = value
            self._by_value[value] = key

    def add_event(self, name, value, overwrite=False):
        """
        Add an event to the trigger definition instance.

        Parameters
        ----------
        name : str
            Name of the event
        value : int
            Value of the event
        overwrite : bool
            If True, overwrite previous event with the same name.
        """
        value = int(value)
        if name in self._by_name and self._by_name[name] == value:
            logger.info(
                f'The event {name} is already set with the value {value}.')
            return
        elif name in self._by_name and self._by_name[name] != value:
            if not overwrite:
                logger.warning(
                    f'The event {name} is already set with the value {value}. '
                    'Skipping.')
                return
            else:
                logger.info(
                    f'The event {name} is already set with the value {value}. '
                    'Overwriting.')

        setattr(self, name, value)
        self._by_name[name] = value
        self._by_value[value] = name

    # --------------------------------------------------------------------
    @staticmethod
    def _check_trigger_file(trigger_file):
        """
        Checks that the provided file exists and ends with ``.ini``.
        """
        if trigger_file is None:
            return None
        trigger_file = Path(trigger_file)
        if trigger_file.exists():
            logger.info(f"Found trigger definition file '{trigger_file}'")
        else:
            logger.error(
                f"Trigger event definition file '{trigger_file}' not found.")
            raise IOError

        if trigger_file.suffix != '.ini':
            logger.error(
                "Trigger event definition file format must be '.ini'.")
            raise IOError

        return trigger_file

    # --------------------------------------------------------------------
    @property
    def by_name(self):
        """
        A dictionnary with string keys and integers value.
        """
        return self._by_name

    @property
    def by_value(self):
        """
        A dictionnary with integers keys and string values.
        """
        return self._by_value
