from abc import ABC, abstractmethod


class _Backend(ABC):
    """
    Class representing a base backend.

    Parameters
    ----------
    scope : neurodecode.stream_viewer._scope._Scope
        The scope connected to a stream receiver acquiring the data and
        applying filtering. The scope has a buffer of _scope._BUFFER_DURATION
        (default: 30s).
    """

    @abstractmethod
    def __init__(self, scope):
        self._scope = scope
        self._show_LPT_events = False
        self._backend_initialized = False

    @abstractmethod
    def init_backend(self, geometry, x_scale, y_scale, channels_to_show_idx):
        """
        Initialize the backend.

        The backend requires the _ScopeControllerUI to be fully
        initialized and setup. Thus the backend is initialized in 2 steps:
            - Creation of the instance (with all the associated methods)
            - Initialization (with all the associated variables)

        Parameters
        ----------
        geometry : tuple | list
            The window geometry as (pos_x, pos_y, size_x, size_y).
        x_scale : int
            The plotting duration, i.e. the scale/range of the x-axis.
        y_scale : float
            The signal scale/range in uV.
        channels_to_show_idx : tuple | list
            The list of channels indices to display, ordered as retrieved from
            LSL.
        """
        assert len(channels_to_show_idx) <= self._scope.n_channels

    @abstractmethod
    def _init_variables(self, x_scale, y_scale, channels_to_show_idx):
        """
        Initialize variables.
        """
        self._x_scale = x_scale  # duration in seconds
        self._y_scale = y_scale  # amplitude scale in uV
        self._channels_to_show_idx = channels_to_show_idx

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
    def backend_initialized(self):
        return self._backend_initialized

    # ------------------------ Update program ----------------------
    @property
    def x_scale(self):
        """
        The X-axis range/scale, i.e. the duration of the plotting window.
        """
        return self._x_scale

    @x_scale.setter
    @abstractmethod
    def x_scale(self, x_scale):
        """
        Called when the user changes the X-axis range/scale, i.e. the duration
        of the plotting window.
        """
        pass

    @property
    def y_scale(self):
        """
        The signal range/scale.
        """
        return self._y_scale

    @y_scale.setter
    @abstractmethod
    def y_scale(self, y_scale):
        """
        Called when the user changes the signal range/scale.
        """
        pass

    @property
    def channels_to_show_idx(self):
        """
        The selected channels.
        """
        return self._channels_to_show_idx

    @channels_to_show_idx.setter
    @abstractmethod
    def channels_to_show_idx(self, channels_to_show_idx):
        """
        Called when the user changes the selection of channels.
        """
        pass

    @property
    def show_LPT_events(self):
        """
        Tick/Untick status of the show_LPT_events box.
        """
        return self._show_LPT_events

    @show_LPT_events.setter
    @abstractmethod
    def show_LPT_events(self, show_LPT_events):
        """
        Called when the user ticks or untick the show_LPT_events box.
        """
        pass
