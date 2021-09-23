from pathlib import Path
from configparser import ConfigParser

from .. import logger
from ..utils._docs import fill_doc


@fill_doc
class TriggerDef:
    """
    Class used to store pairs (str: int) of events name and events value. Each
    name and each value is unique. The pairs can be read from a ``.ini`` file
    or edited manually with `~bsl.triggers.TriggerDef.add_event` and
    `~bsl.triggers.TriggerDef.remove_event`.

    The class will expose the name as attributes ``self.event_str = event_int``
    for all pairs.

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
        Load the .ini file.
        """
        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.optionxform = str
        config.read(str(self._trigger_file))

        for name, value in config.items('events'):
            value = int(value)
            if name in self._by_name:
                logger.info(f'Event name {name} already exists.')
                continue
            if value in self._by_value:
                logger.info(f'Event value {value} already exists.')
                continue
            setattr(self, name, value)
            self._by_name[name] = value
            self._by_value[value] = name

    def add_event(self, name, value, overwrite=False):
        """
        Add an event to the trigger definition instance.

        Parameters
        ----------
        name : `str`
            Name of the event
        value : `int`
            Value of the event
        overwrite : `bool`
            If ``True``, overwrite previous event with the same name or value.
        """
        value = int(value)
        if name in self._by_name and not overwrite:
            logger.info(f'Event name {name} already exists.')
            return
        if value in self._by_value and not overwrite:
            logger.info(f'Event value {value} already exists.')
            return

        if name in self._by_name:
            self.remove_event(name)
        if value in self._by_value:
            self.remove_event(value)

        setattr(self, name, value)
        self._by_name[name] = value
        self._by_value[value] = name

    def remove_event(self, event):
        """
        Remove an event from the trigger definition instance.
        The event can be given by name (str) or by value (int).

        Parameters
        ----------
        event : `str` | `int`
            If a `str` is provided, assumes event is the name.
            If a `int` is provided, assumes event is the value.
        """
        if isinstance(event, str):
            if event not in self._by_name:
                logger.info(f'Event name {event} not found.')
                return
            value = self._by_name[event]
            delattr(self, event)
            del self._by_name[event]
            del self._by_value[value]
        elif isinstance(event, (int, float)):
            event = int(event)
            if event not in self._by_value:
                logger.info(f'Event value {event} not found.')
                return
            name = self._by_value[event]
            delattr(self, name)
            del self._by_name[name]
            del self._by_value[event]
        else:
            raise TypeError(
                f'Supported event types are (str, int), not {type(event)}.')

    def __repr__(self):
        """Representation of the stored events."""
        if len(self._by_name) == 0:
            return 'TriggerDef: No event found.'
        repr_ = f'TriggerDef: {len(self._by_name)} events.\n'
        for name, value in self._by_name.items():
            repr_ += f'  {name}: {value}\n'
        return repr_

    # --------------------------------------------------------------------
    @staticmethod
    def _check_trigger_file(trigger_file):
        """
        Checks that the provided file exists and ends with .ini. Else returns
        None.
        """
        if trigger_file is None:
            return None
        trigger_file = Path(trigger_file)
        if trigger_file.exists():
            logger.info(f"Found trigger definition file '{trigger_file}'")
        else:
            logger.error(
                f"Trigger event definition file '{trigger_file}' not found.")
            return None

        if trigger_file.suffix != '.ini':
            logger.error(
                "Trigger event definition file format must be '.ini'.")
            return None

        return trigger_file

    # --------------------------------------------------------------------
    @property
    def by_name(self):
        """
        A dictionary with string keys and integers value.

        :type: `dict`
        """
        return self._by_name

    @property
    def by_value(self):
        """
        A dictionary with integers keys and string values.

        :type: `dict`
        """
        return self._by_value
