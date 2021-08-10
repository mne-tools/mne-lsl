import copy
from abc import ABC, abstractmethod

from ...utils._docs import fill_doc


@fill_doc
class _Backend(ABC):
    """
    Class representing a base backend.

    Parameters
    ----------
    %(viewer_scope)s
    %(viewer_backend_geometry)s
    %(viewer_backend_xRange)s
    %(viewer_backend_yRange)s
    """

    @abstractmethod
    def __init__(self, scope, geometry, xRange, yRange):
        self._scope = scope

        # Variables
        self._xRange = xRange  # duration in seconds
        self._yRange = yRange  # amplitude range in uV

        self._show_LPT_trigger_events = False
        self._selected_channels = copy.deepcopy(self._scope.selected_channels)

    # -------------------------- Main Loop -------------------------
    @abstractmethod
    def start_timer(self):
        """
        Start the update loop on a 20 ms timer.
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
        Scope connected to a `StreamReceiver` acquiring the data and applying
        filtering. The scope has a buffer of ``BUFFER_DURATION`` seconds
        (default: 30s).
        """
        return self._scope

    @property
    def xRange(self):
        """
        X-axis range/scale, i.e. the duration of the plotting window.
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
        Y-axis range/scale, i.e. the signal amplitude.
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
        Selected channels.
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
        Tick/Untick status of the ``show_LPT_trigger_events`` box.
        """
        return self._show_LPT_trigger_events

    @show_LPT_trigger_events.setter
    @abstractmethod
    def show_LPT_trigger_events(self, show_LPT_trigger_events):
        """
        Called when the user ticks or untick the ``show_LPT_trigger_events``
        box.
        """
        pass
