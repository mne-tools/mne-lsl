import copy
import math
from abc import ABC, abstractmethod

from ...utils._docs import fill_doc
from ...utils.logs import logger


@fill_doc
class _Backend(ABC):
    """Class representing a base backend.

    Parameters
    ----------
    scope : Scope
        Scope connected to a StreamInlet acquiring the data and applying
        filtering. The scope has a buffer of _BUFFER_DURATION seconds
        (default: 30s
    geometry : tuple | list
        Window geometry as (pos_x, pos_y, size_x, size_y).
    xRange : int
        Range of the x-axis (plotting time duration) in seconds.
    yRange : float
        Range of the y-axis (amplitude) in uV.
    """

    @abstractmethod
    def __init__(self, scope, geometry, xRange, yRange):
        self._scope = scope

        # Variables
        self._xRange = xRange  # duration in seconds
        self._yRange = yRange  # amplitude range in uV

        self._show_LPT_trigger_events = False
        self._selected_channels = copy.deepcopy(self._scope.selected_channels)

    def _init_variables(self):  # noqa
        """Initialize variables depending on xRange, yRange and selected_channels."""
        logger.debug("Initialization of variables..")

        # xRange
        self._delta_with_buffer = self._scope.duration_buffer - self._xRange
        self._duration_plot_samples = math.ceil(self._xRange * self._scope.sample_rate)

        logger.debug("Initialization of variables from _Backend complete.")

    # ------------------------ Trigger Events ----------------------
    @abstractmethod
    def _update_LPT_trigger_events(self, trigger_arr):  # noqa
        """Check if new LPT events (on the trigger channel) have entered the buffer.

        New events are added to self._trigger_events and displayed if needed.
        """

    def _clean_up_trigger_events(self):
        """Remove events exiting the buffer."""
        for k in range(len(self._trigger_events) - 1, -1, -1):
            if self._trigger_events[k].position_buffer < 0:
                del self._trigger_events[k]

    # -------------------------- Main Loop -------------------------
    @abstractmethod
    def start_timer(self):
        """Start the update loop on a 20 ms timer."""

    @abstractmethod
    def _update_loop(self, *args, **kwargs):  # noqa
        """Update loop retrieving data from the buffer and updating the Canvas."""
        self._scope.update_loop()

    # --------------------------- Events ---------------------------
    @abstractmethod
    def close(self):
        """Stop the update loop and close the window."""

    # --------------------------------------------------------------------
    @property
    def scope(self):
        """Scope connected to an Inlet acquiring the data and applying filtering.

        The scope has a buffer of BUFFER_DURATION seconds (default: 30s).
        """
        return self._scope

    @property
    def xRange(self):
        """X-axis range/scale, i.e. the duration of the plotting window."""
        return self._xRange

    @xRange.setter
    @abstractmethod
    def xRange(self, xRange):
        """Called when the user changes the X-axis range/scale."""

    @property
    def yRange(self):
        """Y-axis range/scale, i.e. the signal amplitude."""
        return self._yRange

    @yRange.setter
    @abstractmethod
    def yRange(self, yRange):
        """Called when the user changes the signal range/scale."""

    @property
    def selected_channels(self):
        """Selected channels."""
        return self._selected_channels

    @selected_channels.setter
    @abstractmethod
    def selected_channels(self, selected_channels):
        """Called when the user changes the selection of channels."""

    @property
    def show_LPT_trigger_events(self):
        """Tick/Untick status of the show_LPT_trigger_events box."""
        return self._show_LPT_trigger_events

    @show_LPT_trigger_events.setter
    @abstractmethod
    def show_LPT_trigger_events(self, show_LPT_trigger_events):
        """Called when the user ticks or untick the show_LPT_trigger_events box."""


@fill_doc
class _Event(ABC):
    """Base class defining a trigger event.

    Parameters
    ----------
    event_type : str
        Type of event. Supported: 'LPT'.
    event_value : int
        Value of the event.
    position_buffer : float
        Time (seconds) at which the event is positioned in the buffer where:
            0 represents the older events exiting the buffer.
            _BUFFER_DURATION represents the newer events entering the
            buffer.
    position_plot : float
        Time (seconds) at which the event is positioned in the plotting window
        where:
            0 represents the older events exiting the window.
            xRange represents the newer events entering the window.
    """

    _supported = ["LPT"]

    @abstractmethod
    def __init__(self, event_type, event_value, position_buffer, position_plot):
        assert event_type in self._supported
        self._event_type = event_type
        self._event_value = event_value
        self._position_buffer = position_buffer  # In time (s)
        self._position_plot = position_plot  # In time (s)

    @property
    def event_type(self):
        """Event type."""
        return self._event_type

    @property
    def event_value(self):
        """Event value."""
        return self._event_value

    @property
    def position_buffer(self):
        """Position in the buffer."""
        return self._position_buffer

    @position_buffer.setter
    def position_buffer(self, position_buffer):
        """Update both position in the buffer and the plotting window."""
        delta = self._position_buffer - position_buffer
        self._position_buffer = position_buffer
        self._position_plot -= delta

    @property
    def position_plot(self):
        """Position in the plotting window."""
        return self._position_plot

    @position_plot.setter
    def position_plot(self, position_plot):
        """Update only the position in the plotting window."""
        self._position_plot = position_plot
