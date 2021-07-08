import copy
import numpy as np
from abc import ABC, abstractmethod


class _Backend(ABC):
    """
    Class representing a base backend.

    Parameters
    ----------
    scope : neurodecode.stream_viewer._scope._Scope
        Scope connected to a stream receiver acquiring the data and applying
        filtering. The scope has a buffer of _BUFFER_DURATION (default: 30s).
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
        self._available_colors = np.random.uniform(
            size=(self._scope.nb_channels, 3), low=128, high=230)
        self._show_LPT_trigger_events = False
        self._selected_channels = copy.deepcopy(self._scope.selected_channels)

    # -------------------------- Main Loop -------------------------
    @abstractmethod
    def start_timer(self):
        """
        Start the update loop on a 20ms timer.
        """
        pass

    @abstractmethod
    def _update_loop(self, *args, **kwargs):
        """
        Main update loop retrieving data from the scope's buffer and updating
        the Canvas.
        """
        self._scope.update_loop()

    # --------------------------- Events ---------------------------
    @abstractmethod
    def close(self):
        """
        Stops the update loop and close the window.
        """
        pass

    # --------------------------------------------------------------------
    @property
    def scope(self):
        """
        The scope connected to a stream receiver acquiring the data and
        applying filtering. The scope has a buffer of BUFFER_DURATION
        (default: 30s).
        """
        return self._scope

    @property
    def xRange(self):
        """
        The X-axis range/scale, i.e. the duration of the plotting window.
        """
        return self._xRange

    @xRange.setter
    @abstractmethod
    def xRange(self, xRange):
        """
        Called when the user changes the X-axis range/scale, i.e. the duration
        of the plotting window.
        """
        pass

    @property
    def yRange(self):
        """
        The signal range/scale.
        """
        return self._yRange

    @yRange.setter
    @abstractmethod
    def yRange(self, yRange):
        """
        Called when the user changes the signal range/scale.
        """
        pass

    @property
    def selected_channels(self):
        """
        The selected channels.
        """
        return self._selected_channels

    @selected_channels.setter
    @abstractmethod
    def selected_channels(self, selected_channels):
        """
        Called when the user changes the selection of channels.
        """
        pass

    @property
    def show_LPT_trigger_events(self):
        """
        Tick/Untick status of the show_LPT_trigger_events box.
        """
        return self._show_LPT_trigger_events

    @show_LPT_trigger_events.setter
    @abstractmethod
    def show_LPT_trigger_events(self, show_LPT_trigger_events):
        """
        Called when the user ticks or untick the show_LPT_trigger_events box.
        """
        pass
