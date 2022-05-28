import os
from configparser import ConfigParser
from pathlib import Path

from ..utils._checks import _check_type
from ..utils._logs import logger


class TriggerDef:
    """Class used to store pairs {str: int} of events name and events value.

    Each name and each value is unique. The pairs can be read from a ``.ini``
    file or edited manually with :meth:`TriggerDef.add` and
    :meth:`TriggerDef.remove`.

    The class will expose the name as attributes ``self.event_str = event_int``
    for all pairs.

    Parameters
    ----------
    trigger_file : None | path-like
        Path to the ``.ini`` file containing the table converting event numbers
        into event strings.

        .. note:: The ``.ini`` file is read with `configparser` and has to be
                  structured as follows:

                  .. code-block:: python

                      [events]
                      event_str_1 = event_id_1   # comment
                      event_str_2 = event_id_2   # comment

                  Example:

                  .. code-block:: python

                      [events]
                      rest = 1
                      stim = 2
    """

    def __init__(self, trigger_file=None):
        self._by_name = dict()
        self._by_value = dict()
        self.read(trigger_file)

    def read(self, trigger_file):
        """Read events from a ``.ini`` trigger definition file.

        .. note:: The ``.ini`` file is read with `configparser` and has to be
                  structured as follows:

                  .. code-block:: python

                      [events]
                      event_str_1 = event_id_1   # comment
                      event_str_2 = event_id_2   # comment

                  Example:

                  .. code-block:: python

                      [events]
                      rest = 1
                      stim = 2

        Parameters
        ----------
        trigger_file : path-like
            Path to the ``.ini`` file containing the table converting event
            numbers into event strings.
        """
        self._trigger_file = TriggerDef._check_trigger_file(trigger_file)
        if self._trigger_file is None:
            return

        config = ConfigParser(inline_comment_prefixes=("#", ";"))
        config.optionxform = str
        config.read(str(self._trigger_file))

        for name, value in config.items("events"):
            value = int(value)
            if value in self._by_value:
                logger.info("Event value %s already exists. Skipping.", value)
                continue
            setattr(self, name, value)
            self._by_name[name] = value
            self._by_value[value] = name

    def write(self, trigger_file):
        """Write events to a ``.ini`` trigger definition file.

        .. note:: The ``.ini`` file is written with `configparser` and is
                  structured as follows:

                  .. code-block:: python

                      [events]
                      event_str_1 = event_id_1
                      event_str_2 = event_id_2
        """
        trigger_file = TriggerDef._check_write_to_trigger_file(trigger_file)

        config = ConfigParser()
        config["events"] = self._by_name
        with open(trigger_file, "w") as configfile:
            config.write(configfile)

    def add(self, name, value, overwrite=False):
        """Add an event to the trigger definition instance.

        Parameters
        ----------
        name : str
            Name of the event
        value : int
            Value of the event
        overwrite : bool
            If ``True``, overwrite previous event with the same name or value.
        """
        _check_type(name, (str,), item_name="name")
        _check_type(value, ("int",), item_name="value")
        _check_type(overwrite, (bool,), item_name="overwrite")
        if name in self._by_name and not overwrite:
            logger.info("Event name %s already exists. Skipping.", name)
            return
        if value in self._by_value and not overwrite:
            logger.info("Event value %s already exists. Skipping.", value)
            return

        if name in self._by_name:
            self.remove(name)
        if value in self._by_value:
            self.remove(value)

        setattr(self, name, value)
        self._by_name[name] = value
        self._by_value[value] = name

    def remove(self, event):
        """Remove an event from the trigger definition instance.

        The event can be given by name (str) or by value (int).

        Parameters
        ----------
        event : str | int
            If a str is provided, assumes event is the name.
            If a int is provided, assumes event is the value.
        """
        _check_type(event, (str, "numeric"), item_name="event")
        if isinstance(event, str):
            if event not in self._by_name:
                logger.info("Event name %s not found.", event)
                return
            value = self._by_name[event]
            delattr(self, event)
            del self._by_name[event]
            del self._by_value[value]
        elif isinstance(event, (int, float)):
            event = int(event)
            if event not in self._by_value:
                logger.info("Event value %s not found.", event)
                return
            name = self._by_value[event]
            delattr(self, name)
            del self._by_name[name]
            del self._by_value[event]

    def __repr__(self):
        """Representation of the stored events."""
        if len(self._by_name) == 0:
            return "TriggerDef: No event found."
        repr_ = f"TriggerDef: {len(self._by_name)} events.\n"
        for name, value in self._by_name.items():
            repr_ += f"  {name}: {value}\n"
        return repr_

    # --------------------------------------------------------------------
    @staticmethod
    def _check_trigger_file(trigger_file):
        """Check that the provided file exists and ends with .ini."""
        _check_type(
            trigger_file, (None, "path-like"), item_name="trigger_file"
        )

        if trigger_file is None:
            return None
        else:
            trigger_file = Path(trigger_file)

        if trigger_file.exists() and trigger_file.suffix == ".ini":
            logger.info(
                "Found trigger definition file '%s'", trigger_file.name
            )
            return trigger_file
        elif trigger_file.exists() and trigger_file.suffix != ".ini":
            logger.error(
                "Argument trigger_file must be a valid Path to a .ini file. "
                "Provided: %s",
                trigger_file.suffix,
            )
            return None
        else:
            logger.error(
                "Trigger event definition file '%s' not found.", trigger_file
            )
            return None

    @staticmethod
    def _check_write_to_trigger_file(trigger_file):  # noqa
        """Check that the directory exists and that the file name ends with
        .ini."""
        _check_type(trigger_file, ("path-like",), item_name="trigger_file")

        trigger_file = Path(trigger_file)
        if trigger_file.suffix != ".ini":
            raise ValueError(
                "Argument trigger_file must end with .ini. "
                "Provided: %s" % trigger_file.suffix
            )

        os.makedirs(trigger_file.parent, exist_ok=True)
        return trigger_file

    # --------------------------------------------------------------------
    @property
    def by_name(self):
        """A dictionary with string keys and integers value.

        :type: dict
        """
        return self._by_name

    @property
    def by_value(self):
        """A dictionary with integers keys and string values.

        :type: dict
        """
        return self._by_value
