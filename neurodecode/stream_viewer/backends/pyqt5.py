"""
PyQt5 Canvas for Neurodecode's StreamViewer.
"""
import math
import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore

# pg.setConfigOptions(antialias=True)

class _BackendPyQt5:
    """
    The PyQt5 backend for neurodecode's StreamViewer.

    Parameters
    ----------
    scope : neurodecode.stream_viewer._scope._Scope
        The scope connected to a stream receiver acquiring the data and applying
        filtering. The scope has a buffer of _scope._BUFFER_DURATION (default: 30s).
    """
    # ---------------------------- Init ---------------------------
    def __init__(self, scope):
        self.scope = scope
        self.backend_initialized = False
        self.trigger_events = list()
        self._show_LPT_events = False

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
        assert len(channels_to_show_idx) <= self.scope.n_channels
        self.init_variables(x_scale, y_scale, channels_to_show_idx)

        self._win = pg.GraphicsLayoutWidget(
            size=geometry[2:],
            title=f'Stream Viewer: {self.scope.stream_name}')

        self._win.show()
        self.init_canvas()
        self.init_plot()

        self.backend_initialized = True
        self._timer = QtCore.QTimer(self._win)
        self._timer.timeout.connect(self.update_loop)

    def init_variables(self, x_scale, y_scale, channels_to_show_idx):
        """
        Initialize variables.
        """
        self.x_scale = x_scale  # duration in seconds
        self.y_scale = y_scale  # amplitude scale in uV
        self.available_colors = np.random.uniform(
            size=(self.scope.n_channels, 3), low=128, high=230)
        self.channels_to_show_idx = channels_to_show_idx
        self.init_n_samples_plot()

    def init_n_samples_plot(self):
        """
        Initialize the number of samples present in the plotting window, and
        the duration difference between the plotting window with the buffer.
        """
        self.delta_with_buffer = self.scope.duration_buffer - self.x_scale
        self.n_samples_plot = math.ceil(self.x_scale * self.scope.sample_rate)

    def init_canvas(self):
        """
        Initialize the plot handler with all the associated labels and settings.
        """
        self._plot_handler = self._win.addPlot() # pyqtgraph.PlotItem
        subsampling_ratio = self.scope.sample_rate / 64
        self._plot_handler.setDownsampling(ds=subsampling_ratio,
                                            auto=None, mode='mean')

        self._plot_handler.setMouseEnabled(x=False, y=False)

        self.init_range()
        self.init_y_axis()
        self.init_x_axis()

    def init_range(self):
        """
        Initialize the signal range and the plot window duration.
        """
        self.y_range = [1.5*self.y_scale,
                        -self.y_scale*(len(self.channels_to_show_idx)+1)]
        self._plot_handler.setRange(
            xRange=[0, self.x_scale],
            yRange=self.y_range)
        self._plot_handler.disableAutoRange()
        self._plot_handler.showGrid(y=True)

    def init_y_axis(self):
        """
        Initialize the Y-axis and its label and ticks.
        """
        yticks = [(-k*self.y_scale, self.scope.channels_labels[idx]) \
                  for k, idx in enumerate(self.channels_to_show_idx)]
        ticks = [yticks, []] # [major, minor]
        self._plot_handler.getAxis('left').setTicks(ticks)
        self._plot_handler.setLabel(
            axis='left', text=f'Scale (uV): {self.y_scale}')

    def init_x_axis(self):
        """
        Initialize the X-axis and its label.
        """
        self.x_arr = np.arange(self.n_samples_plot) / self.scope.sample_rate
        self._plot_handler.setLabel(axis='bottom', text='Time (s)')

    # ------------------------- Init plot -------------------------
    def init_plot(self):
        """
        Initialize the plot by adding a PlotDataItem for every displayed channel.
        """
        self.init_plotting_channel_offset()
        self._plot_handler.clear()
        self.plots = dict()
        for k, idx in enumerate(self.channels_to_show_idx):
            # Add PlotDataItem
            self.plots[idx] = self._plot_handler.plot(
                x=self.x_arr,
                y=self.scope.data_buffer[idx, -self.n_samples_plot:]+self.offset[k],
                pen=pg.mkColor(self.available_colors[idx, :]))

    def init_plotting_channel_offset(self):
        """
        As all channels are plotted on the same plot handler / window, an offset
        is computed to vertically shift each channel.
        """
        self.offset =  np.arange(
            0, -len(self.channels_to_show_idx)*self.y_scale, -self.y_scale)

    # ------------------------ Trigger Events ----------------------
    def update_LPT_events(self, trigger_arr):
        """
        Check if new LPT events (on the trigger channel) have entered the buffer.
        New events are added to self.trigger_events and displayed if needed.
        """
        events_trigger_arr_idx = np.where(trigger_arr != 0)[0]
        events_values = trigger_arr[events_trigger_arr_idx]

        for k, ev_value in enumerate(events_values):
            position_buffer = self.scope.duration_buffer - \
                (trigger_arr.shape[0] - events_trigger_arr_idx[k])/self.scope.sample_rate
            position_plot = position_buffer - self.delta_with_buffer

            event =  _TriggerEvent(
                event_type='LPT',
                event_value=ev_value,
                position_buffer=position_buffer,
                position_plot=position_plot,
                plot_handler=self._plot_handler,
                plot_y_scale=self.y_scale)

            if position_plot >= 0:
                if event.event_type == 'LPT' and self._show_LPT_events:
                        event.addEventPlot()

            self.trigger_events.append(event)

    def clean_up_events(self):
        """
        Hide events exiting the plotting window and remove events exiting the
        buffer.
        """
        for event in self.trigger_events:
            if event.position_plot < 0:
                event.removeEventPlot()

        for k in range(len(self.trigger_events)-1, -1, -1):
            if self.trigger_events[k].position_buffer < 0:
                del self.trigger_events[k]

    # -------------------------- Main Loop -------------------------
    def start_timer(self):
        """
        Start the update loop on a 20ms timer.
        """
        self._timer.start(20)

    def update_loop(self):
        """
        Main update loop retrieving data from the scope's buffer and updating
        the Canvas.
        """
        self.scope.update_loop()
        if len(self.scope._ts_list) > 0:
            for k, idx in enumerate(self.channels_to_show_idx):
                self.plots[idx].setData(
                    x=self.x_arr,
                    y=self.scope.data_buffer[idx, -self.n_samples_plot:]+self.offset[k])

            # Update existing events position
            for event in self.trigger_events:
                event.update_position(
                    event.position_buffer - len(self.scope._ts_list)/self.scope.sample_rate,
                    event.position_plot - len(self.scope._ts_list)/self.scope.sample_rate)
            # Add new events entering the buffer
            self.update_LPT_events(self.scope.trigger_buffer[-len(self.scope._ts_list):])
            # Hide/Remove events exiting window and buffer
            self.clean_up_events()

    # ------------------------ Update program ----------------------
    def update_x_scale(self, new_x_scale):
        """
        Called when the user changes the X-axis range/scale, i.e. the duration
        of the plotting window.
        """
        if self.backend_initialized:
            self.x_scale = new_x_scale
            self.init_n_samples_plot()
            self.init_range()
            self.init_x_axis()

            for event in self.trigger_events:
                event.update_position(
                    event.position_buffer,
                    event.position_buffer - self.delta_with_buffer)
                if event.position_plot >= 0:
                    if event.event_type == 'LPT' and self._show_LPT_events:
                        event.addEventPlot()
                else:
                    event.removeEventPlot()

    def update_y_scale(self, new_y_scale):
        """
        Called when the user changes the signal range/scale.
        """
        if self.backend_initialized:
            self.y_scale = new_y_scale
            self.init_range()
            self.init_y_axis()
            self.init_plotting_channel_offset()

            for event in self.trigger_events:
                event.update_scales(self.y_scale)

    def update_channels_to_show_idx(self, new_channels_to_show_idx):
        """
        Called when the user changes the selection of channels.
        """
        if self.backend_initialized:
            plots2remove = [idx for idx in self.channels_to_show_idx \
                            if idx not in new_channels_to_show_idx]
            plots2add = [idx for idx in new_channels_to_show_idx \
                         if idx not in self.channels_to_show_idx]

            self.channels_to_show_idx = new_channels_to_show_idx
            self.init_range()
            self.init_y_axis()
            self.init_plotting_channel_offset()

            for idx in plots2remove:
                self._plot_handler.removeItem(self.plots[idx])
                del self.plots[idx]
            for k, idx in enumerate(plots2add):
                self.plots[idx] = self._plot_handler.plot(
                    x=self.x_arr,
                    y=self.scope.data_buffer[idx, -self.n_samples_plot:]+self.offset[k],
                    pen=pg.mkColor(self.available_colors[idx, :]))

    def update_show_LPT_events(self):
        """
        Called when the user ticks or untick the show_LPT_events box.
        """
        for event in self.trigger_events:
            if event.position_plot >= 0 and event.event_type == 'LPT':
                if self._show_LPT_events:
                    event.addEventPlot()
                else:
                    event.removeEventPlot()

    # --------------------------- Events ---------------------------
    def close(self):
        """
        Stops the update loop and close the window.
        """
        self._timer.stop()
        self._win.close()

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
        self.event_type = event_type
        self.event_value = event_value
        self.position_buffer = position_buffer # In time (s)
        self.position_plot = position_plot # In time (s)

        self.plot_handler = plot_handler
        self.plot_y_scale = plot_y_scale

        self.LineItem = None
        self.TextItem = None
        self.plotted = False

    def addEventPlot(self):
        """
        Plots the event on the handler.
        """
        if not self.plotted:
            self.LineItem = pg.InfiniteLine(pos=self.position_plot, pen=self.pens[self.event_type])
            self.plot_handler.addItem(self.LineItem)

            self.TextItem = pg.TextItem(str(self.event_value), anchor=(0.5, 1),
                               fill=(0, 0, 0), color=self.pens[self.event_type])
            self.TextItem.setPos(self.position_plot, 1.5*self.plot_y_scale)
            self.plot_handler.addItem(self.TextItem)
            self.plotted = True

    def update_scales(self, plot_y_scale):
        """
        Update the signal range/scale used to position the TextItem.
        """
        self.plot_y_scale = plot_y_scale
        self._update()

    def update_position(self, position_buffer, position_plot):
        """
        Update the position on the plotting window and in the buffer.
        """
        self.position_buffer = position_buffer
        self.position_plot = position_plot
        self._update()

    def _update(self):
        """
        Updates the plot handler.
        """
        if self.LineItem is not None:
            self.LineItem.setValue(self.position_plot)
        if self.TextItem is not None:
            self.TextItem.setPos(self.position_plot, 1.5*self.plot_y_scale)

    def removeEventPlot(self):
        """
        Remove the event from the plot handler.
        """
        if self.plotted:
            self.plot_handler.removeItem(self.LineItem)
            self.plot_handler.removeItem(self.TextItem)
            self.LineItem = None
            self.TextItem = None
            self.plotted = False

    def __del__(self):
        try:
            self.removeEventPlot()
        except:
            pass
