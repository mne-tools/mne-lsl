"""
PyQt5 Canvas for Neurodecode's StreamViewer.
"""
import math
import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore

from ._backend import _Backend
from ... import logger

# pg.setConfigOptions(antialias=True)


class _BackendPyQt5(_Backend):
    """
    The PyQt5 backend for neurodecode's StreamViewer.

    Parameters
    ----------
    scope : neurodecode.stream_viewer._scope._Scope
        The scope connected to a stream receiver acquiring the data and
        applying filtering. The scope has a buffer of _scope._BUFFER_DURATION
        (default: 30s).
    """
    # ---------------------------- Init ---------------------------

    def __init__(self, scope):
        super().__init__(scope)
        self._trigger_events = list()

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
        super().init_backend(geometry, x_scale, y_scale, channels_to_show_idx)
        self._init_variables(x_scale, y_scale, channels_to_show_idx)

        self._win = pg.GraphicsLayoutWidget(
            size=geometry[2:],
            title=f'Stream Viewer: {self._scope.stream_name}')

        self._win.show()
        self._init_canvas()
        self._init_plot()

        self._backend_initialized = True
        self._timer = QtCore.QTimer(self._win)
        self._timer.timeout.connect(self._update_loop)

    def _init_variables(self, x_scale, y_scale, channels_to_show_idx):
        """
        Initialize variables.
        """
        super()._init_variables(x_scale, y_scale, channels_to_show_idx)
        self._available_colors = np.random.uniform(
            size=(self._scope.n_channels, 3), low=128, high=230)
        self._init_n_samples_plot()

    def _init_n_samples_plot(self):
        """
        Initialize the number of samples present in the plotting window, and
        the duration difference between the plotting window with the buffer.
        """
        self._delta_with_buffer = self._scope.duration_buffer - self._x_scale
        self._n_samples_plot = math.ceil(self._x_scale * self._scope.sample_rate)

    def _init_canvas(self):
        """
        Initialize the plot handler with the associated labels and settings.
        """
        self._plot_handler = self._win.addPlot()  # pyqtgraph.PlotItem
        subsampling_ratio = self._scope.sample_rate / 64
        self._plot_handler.setDownsampling(ds=subsampling_ratio,
                                           auto=None, mode='mean')

        self._plot_handler.setMouseEnabled(x=False, y=False)

        self._init_range()
        self._init_y_axis()
        self._init_x_axis()

    def _init_range(self):
        """
        Initialize the signal range and the plot window duration.
        """
        y_range = [
            1.5*self._y_scale,
            -self._y_scale*(len(self._channels_to_show_idx)+1)]
        self._plot_handler.setRange(
            xRange=[0, self._x_scale],
            yRange=y_range)
        self._plot_handler.disableAutoRange()
        self._plot_handler.showGrid(y=True)

    def _init_y_axis(self):
        """
        Initialize the Y-axis and its label and ticks.
        """
        yticks = [(-k*self._y_scale, self._scope.channels_labels[idx])
                  for k, idx in enumerate(self._channels_to_show_idx)]
        ticks = [yticks, []]  # [major, minor]
        self._plot_handler.getAxis('left').setTicks(ticks)
        self._plot_handler.setLabel(
            axis='left', text=f'Scale (uV): {self._y_scale}')

    def _init_x_axis(self):
        """
        Initialize the X-axis and its label.
        """
        self._x_arr = np.arange(self._n_samples_plot) / self._scope.sample_rate
        self._plot_handler.setLabel(axis='bottom', text='Time (s)')

    def _init_plot(self):
        """
        Initialize the plot by adding a PlotDataItem for every shown channel.
        """
        self._init_plotting_channel_offset()
        self._plot_handler.clear()
        self._plots = dict()
        for k, idx in enumerate(self._channels_to_show_idx):
            # Add PlotDataItem
            self._plots[idx] = self._plot_handler.plot(
                x=self._x_arr,
                y=self._scope.data_buffer[idx, -
                                         self._n_samples_plot:]+self._offset[k],
                pen=pg.mkColor(self._available_colors[idx, :]))

    def _init_plotting_channel_offset(self):
        """
        As all channels are plotted on the same plot handler / window, an
        offset is computed to vertically shift each channel.
        """
        self._offset = np.arange(
            0, -len(self._channels_to_show_idx)*self._y_scale, -self._y_scale)

    # ------------------------ Trigger Events ----------------------
    def _update_LPT_events(self, trigger_arr):
        """
        Check if new LPT events (on the trigger channel) have entered the
        buffer. New events are added to self._trigger_events and displayed if
        needed.
        """
        events_trigger_arr_idx = np.where(trigger_arr != 0)[0]
        events_values = trigger_arr[events_trigger_arr_idx]

        for k, ev_value in enumerate(events_values):
            position_buffer = self._scope.duration_buffer - \
                (trigger_arr.shape[0] - events_trigger_arr_idx[k]
                 )/self._scope.sample_rate
            position_plot = position_buffer - self._delta_with_buffer

            event = _TriggerEvent(
                event_type='LPT',
                event_value=ev_value,
                position_buffer=position_buffer,
                position_plot=position_plot,
                plot_handler=self._plot_handler,
                plot_y_scale=self._y_scale)

            if position_plot >= 0:
                if event.event_type == 'LPT' and self._show_LPT_events:
                    event.addEventPlot()

            self._trigger_events.append(event)

    def _clean_up_events(self):
        """
        Hide events exiting the plotting window and remove events exiting the
        buffer.
        """
        for event in self._trigger_events:
            if event.position_plot < 0:
                event.removeEventPlot()

        for k in range(len(self._trigger_events)-1, -1, -1):
            if self._trigger_events[k].position_buffer < 0:
                del self._trigger_events[k]

    # -------------------------- Main Loop -------------------------
    def start_timer(self):
        """
        Start the update loop on a 20ms timer.
        """
        self._timer.start(20)

    def _update_loop(self):
        """
        Main update loop retrieving data from the scope's buffer and updating
        the Canvas.
        """
        super()._update_loop()

        if len(self._scope.ts_list) > 0:
            for k, idx in enumerate(self._channels_to_show_idx):
                self._plots[idx].setData(
                    x=self._x_arr,
                    y=self._scope.data_buffer[idx, -self._n_samples_plot:] +
                    self._offset[k])

            # Update existing events position
            for event in self._trigger_events:
                event.update_position(
                    event.position_buffer -
                    len(self._scope.ts_list) / self._scope.sample_rate,
                    event.position_plot -
                    len(self._scope.ts_list) / self._scope.sample_rate)
            # Add new events entering the buffer
            self._update_LPT_events(
                self._scope.trigger_buffer[-len(self._scope.ts_list):])
            # Hide/Remove events exiting window and buffer
            self._clean_up_events()

    # --------------------------- Events ---------------------------
    def close(self):
        """
        Stops the update loop and close the window.
        """
        self._timer.stop()
        self._win.close()

    # ------------------------ Update program ----------------------
    @_Backend.x_scale.setter
    def x_scale(self, x_scale):
        if self._backend_initialized:
            self._x_scale = x_scale
            self._init_n_samples_plot()
            self._init_range()
            self._init_x_axis()

            for event in self._trigger_events:
                event.update_position(
                    event.position_buffer,
                    event.position_buffer - self._delta_with_buffer)
                if event.position_plot >= 0:
                    if event.event_type == 'LPT' and self._show_LPT_events:
                        event.addEventPlot()
                else:
                    event.removeEventPlot()

    @_Backend.y_scale.setter
    def y_scale(self, y_scale):
        if self._backend_initialized:
            self._y_scale = y_scale
            self._init_range()
            self._init_y_axis()
            self._init_plotting_channel_offset()

            for event in self._trigger_events:
                event.y_scale = self._y_scale

    @_Backend.channels_to_show_idx.setter
    def channels_to_show_idx(self, channels_to_show_idx):
        if self._backend_initialized:
            plots2remove = [idx for idx in self._channels_to_show_idx
                            if idx not in channels_to_show_idx]
            plots2add = [idx for idx in channels_to_show_idx
                         if idx not in self._channels_to_show_idx]

            self._channels_to_show_idx = channels_to_show_idx
            self._init_range()
            self._init_y_axis()
            self._init_plotting_channel_offset()

            for idx in plots2remove:
                self._plot_handler.removeItem(self._plots[idx])
                del self._plots[idx]
            for k, idx in enumerate(plots2add):
                self._plots[idx] = self._plot_handler.plot(
                    x=self._x_arr,
                    y=self._scope.data_buffer[
                        idx, -self._n_samples_plot:] + self._offset[k],
                    pen=pg.mkColor(self._available_colors[idx, :]))

    @_Backend.show_LPT_events.setter
    def show_LPT_events(self, show_LPT_events):
        self._show_LPT_events = show_LPT_events
        for event in self._trigger_events:
            if event.position_plot >= 0 and event.event_type == 'LPT':
                if self._show_LPT_events:
                    event.addEventPlot()
                else:
                    event.removeEventPlot()

class _TriggerEvent:
    """
    Class defining a trigger event.

    Parameters
    ----------
    event_type : str
        The type of event. Supported: 'LPT'.
    event_value : int | float
        The value of the event displayed in the TextItem.
    position_buffer : float
        The time at which the event is positionned in the buffer where:
            0 represents the older events exiting the buffer.
            _BUFFER_DURATION represents the newer events entering the buffer.
    position_plot : float
        The time at which the event is positionned in the plotting window.
    plot_handler : pyqtgraph.PlotItem
        The plot handler.
    plot_y_scale : int | float
        The currently set signal range/scale.
    """
    pens = {'LPT': pg.mkColor(0, 255, 0)}

    def __init__(self, event_type, event_value, position_buffer, position_plot,
                 plot_handler, plot_y_scale):
        assert event_type in self.pens.keys()
        self._event_type = event_type
        self._event_value = event_value
        self._position_buffer = position_buffer  # In time (s)
        self._position_plot = position_plot  # In time (s)

        self._plot_handler = plot_handler
        self._plot_y_scale = plot_y_scale

        self._lineItem = None
        self._textItem = None
        self._plotted = False

    def addEventPlot(self):
        """
        Plots the event on the handler.
        """
        if not self._plotted:
            self._lineItem = pg.InfiniteLine(
                pos=self._position_plot, pen=self.pens[self._event_type])
            self._plot_handler.addItem(self._lineItem)

            self._textItem = pg.TextItem(str(self._event_value),
                                        anchor=(0.5, 1),
                                        fill=(0, 0, 0),
                                        color=self.pens[self._event_type])
            self._textItem.setPos(self._position_plot, 1.5*self._plot_y_scale)
            self._plot_handler.addItem(self._textItem)
            self._plotted = True

    # TODO: Move as setter
    def update_position(self, position_buffer, position_plot):
        """
        Update the position on the plotting window and in the buffer.
        """
        self._position_buffer = position_buffer
        self._position_plot = position_plot
        self._update()

    def _update(self):
        """
        Updates the plot handler.
        """
        if self._lineItem is not None:
            self._lineItem.setValue(self._position_plot)
        if self._textItem is not None:
            self._textItem.setPos(self._position_plot, 1.5*self._plot_y_scale)

    def removeEventPlot(self):
        """
        Remove the event from the plot handler.
        """
        if self._plotted:
            self._plot_handler.removeItem(self._lineItem)
            self._plot_handler.removeItem(self._textItem)
            self._lineItem = None
            self._textItem = None
            self._plotted = False

    def __del__(self):
        try:
            self.removeEventPlot()
        except Exception:
            pass

    @property
    def event_type(self):
        """
        Event type.
        """
        return self._event_type

    @property
    def event_value(self):
        """
        Event value.
        """
        return self._event_value

    @property
    def plotted(self):
        """
        True if the event is displayed, else False.
        """
        return self._plotted

    @property
    def position_buffer(self):
        """
        Position in the buffer.
        """
        return self._position_buffer

    @property
    def position_plot(self):
        """
        Position in the plotting window.
        """
        return self._position_plot

    @property
    def plot_y_scale(self):
        """
        The signal range/scale used to position the TextItem.
        """
        return self.__plot_y_scale

    @plot_y_scale.setter
    def plot_y_scale(self, plot_y_scale):
        self._plot_y_scale = plot_y_scale
        self._update()
