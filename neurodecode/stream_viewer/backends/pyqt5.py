"""
PyQt5 Canvas for Neurodecode's StreamViewer.
"""
import math
import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore

from ._backend import _Backend

# pg.setConfigOptions(antialias=True)


class _BackendPyQt5(_Backend):
    """
    The PyQt5 backend for neurodecode's StreamViewer.

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

    # ---------------------------- Init ---------------------------
    def __init__(self, scope, geometry, xRange, yRange):
        super().__init__(scope, geometry, xRange, yRange)
        self._trigger_events = list()

        # Variables
        self._available_colors = np.random.uniform(
            size=(self._scope.nb_channels, 3), low=128, high=230)
        self._init_variables()

        # Canvas
        self._win = pg.GraphicsLayoutWidget(
            size=geometry[2:],
            title=f'Stream Viewer: {self._scope.stream_name}')
        self._win.show()
        self._plot_handler = self._win.addPlot()  # pyqtgraph.PlotItem
        subsampling_ratio = self._scope.sample_rate / 64
        self._plot_handler.setDownsampling(ds=subsampling_ratio,
                                           auto=None, mode='mean')
        self._plot_handler.setMouseEnabled(x=False, y=False)
        self._init_canvas()

        # Plots
        self._plot_handler.clear()
        self._plots = dict()
        # Add PlotDataItem
        for k, idx in enumerate(self._scope.selected_channels):
            self._plots[idx] = self._plot_handler.plot(
                x=self._x_arr,
                y=self._scope.data_buffer[
                    idx, -self._duration_plot_samples:]+self._offset[k],
                pen=pg.mkColor(self._available_colors[idx, :]))

        # Timer
        self._timer = QtCore.QTimer(self._win)
        self._timer.timeout.connect(self._update_loop)

    def _init_variables(self):
        """
        Initialize variables depending on xRange, yRange and selected_channels.
        """
        # xRange
        self._delta_with_buffer = self._scope.duration_buffer - self._xRange
        self._duration_plot_samples = math.ceil(
            self._xRange*self._scope.sample_rate)

        # yRange
        self._offset = np.arange(
            0, -len(self._scope.selected_channels)*self._yRange,
            -self._yRange)

    def _init_canvas(self):
        """
        Initialize the drawing canvas.
        """
        # Ranges
        yRange = [
            1.5*self._yRange,
            -self._yRange*(len(self._scope.selected_channels)+1)]
        self._plot_handler.setRange(
            xRange=[0, self._xRange],
            yRange=yRange)
        self._plot_handler.disableAutoRange()
        self._plot_handler.showGrid(y=True)

        # Y-axis
        yticks = [(-k*self._yRange, self._scope.channels_labels[idx])
                  for k, idx in enumerate(self._scope.selected_channels)]
        ticks = [yticks, []]  # [major, minor]
        self._plot_handler.getAxis('left').setTicks(ticks)
        self._plot_handler.setLabel(
            axis='left', text=f'Scale (uV): {self._yRange}')

        # X-axis
        self._x_arr = np.arange(self._duration_plot_samples) \
            / self._scope.sample_rate
        self._plot_handler.setLabel(axis='bottom', text='Time (s)')

    # ------------------------ Trigger Events ----------------------
    def _update_LPT_trigger_events(self, trigger_arr):
        """
        Check if new LPT events (on the trigger channel) have entered the
        buffer. New events are added to self._trigger_events and displayed if
        needed.
        """
        events_trigger_arr_idx = np.where(trigger_arr != 0)[0]
        events_values = trigger_arr[events_trigger_arr_idx]

        for k, ev_value in enumerate(events_values):
            position_buffer = self._scope.duration_buffer - \
                (trigger_arr.shape[0] - events_trigger_arr_idx[k]) \
                / self._scope.sample_rate
            position_plot = position_buffer - self._delta_with_buffer

            event = _TriggerEvent(
                event_type='LPT',
                event_value=ev_value,
                position_buffer=position_buffer,
                position_plot=position_plot,
                plot_handler=self._plot_handler,
                plot_yRange=self._yRange)

            if position_plot >= 0:
                if event.event_type == 'LPT' and self._show_LPT_trigger_events:
                    event.addEventPlot()

            self._trigger_events.append(event)

    def _clean_up_trigger_events(self):
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
            for k, idx in enumerate(self._scope.selected_channels):
                self._plots[idx].setData(
                    x=self._x_arr,
                    y=self._scope.data_buffer[
                        idx, -self._duration_plot_samples:] + self._offset[k])

            # Update existing events position
            for event in self._trigger_events:
                event.update_position(
                    event.position_buffer -
                    len(self._scope.ts_list) / self._scope.sample_rate,
                    event.position_plot -
                    len(self._scope.ts_list) / self._scope.sample_rate)
            # Add new events entering the buffer
            self._update_LPT_trigger_events(
                self._scope.trigger_buffer[-len(self._scope.ts_list):])
            # Hide/Remove events exiting window and buffer
            self._clean_up_trigger_events()

    # --------------------------- Events ---------------------------
    def close(self):
        """
        Stops the update loop and close the window.
        """
        self._timer.stop()
        self._win.close()

    # ------------------------ Update program ----------------------
    @_Backend.xRange.setter
    def xRange(self, xRange):
        """
        Called when the user changes the X-axis range/scale, i.e. the duration
        of the plotting window.
        """
        self._xRange = xRange
        self._init_variables()
        self._init_canvas()

        for event in self._trigger_events:
            event.update_position(
                event.position_buffer,
                event.position_buffer - self._delta_with_buffer)
            if event.position_plot >= 0:
                if event.event_type == 'LPT' and self._show_LPT_trigger_events:
                    event.addEventPlot()
            else:
                event.removeEventPlot()

    @_Backend.yRange.setter
    def yRange(self, yRange):
        """
        Called when the user changes the signal range/scale.
        """
        self._yRange = yRange
        self._init_variables()
        self._init_canvas()

        for event in self._trigger_events:
            event.yRange = self._yRange

    @_Backend.selected_channels.setter
    def selected_channels(self, selected_channels):
        """
        Called when the user changes the selection of channels.
        """
        plots2remove = [idx for idx in self._selected_channels
                        if idx not in selected_channels]
        plots2add = [idx for idx in selected_channels
                     if idx not in self._selected_channels]
        self._selected_channels = selected_channels
        self._init_variables()
        self._init_canvas()

        for idx in plots2remove:
            self._plot_handler.removeItem(self._plots[idx])
            del self._plots[idx]
        for k, idx in enumerate(plots2add):
            self._plots[idx] = self._plot_handler.plot(
                x=self._x_arr,
                y=self._scope.data_buffer[
                    idx, -self._duration_plot_samples:] + self._offset[k],
                pen=pg.mkColor(self._available_colors[idx, :]))

    @_Backend.show_LPT_trigger_events.setter
    def show_LPT_trigger_events(self, show_LPT_trigger_events):
        """
        Called when the user ticks or untick the show_LPT_trigger_events box.
        """
        self._show_LPT_trigger_events = show_LPT_trigger_events
        for event in self._trigger_events:
            if event.position_plot >= 0 and event.event_type == 'LPT':
                if self._show_LPT_trigger_events:
                    event.addEventPlot()
                else:
                    event.removeEventPlot()


class _TriggerEvent:
    """
    Class defining a trigger event.

    Parameters
    ----------
    event_type : str
        Type of event. Supported: 'LPT'.
    event_value : int | float
        Value of the event displayed in the TextItem.
    position_buffer : float
        Time at which the event is positionned in the buffer where:
            0 represents the older events exiting the buffer.
            _BUFFER_DURATION represents the newer events entering the buffer.
    position_plot : float
        Time at which the event is positionned in the plotting window.
    plot_handler : pyqtgraph.PlotItem
        Plot handler.
    plot_yRange : int | float
        Currently set signal range/scale.
    """
    pens = {'LPT': pg.mkColor(0, 255, 0)}

    def __init__(self, event_type, event_value, position_buffer, position_plot,
                 plot_handler, plot_yRange):
        assert event_type in self.pens.keys()
        self._event_type = event_type
        self._event_value = event_value
        self._position_buffer = position_buffer  # In time (s)
        self._position_plot = position_plot  # In time (s)

        self._plot_handler = plot_handler
        self._plot_yRange = plot_yRange

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
            self._textItem.setPos(self._position_plot, 1.5*self.plot_yRange)
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
            self._textItem.setPos(self._position_plot, 1.5*self.plot_yRange)

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
    def plot_yRange(self):
        """
        Signal range/scale used to position the TextItem.
        """
        return self._plot_yRange

    @plot_yRange.setter
    def plot_yRange(self, plot_yRange):
        self._plot_yRange = plot_yRange
        self._update()
