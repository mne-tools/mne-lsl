"""
PyQt5 Canvas for Neurodecode's StreamViewer.
"""
import math
import numpy as np

import pyqtgraph as pg
from PyQt5 import QtCore

class _BackendPyQt5:
    # ---------------------------- Init ---------------------------
    def __init__(self, scope):
        self.scope = scope
        self.backend_initialized = False
        self._show_LPT_events = False
        self._show_Key_events = False

    def init_backend(self, geometry, x_scale, y_scale, channels_to_show_idx):
        assert len(channels_to_show_idx) <= self.scope.n_channels
        self.init_variables(x_scale, y_scale, channels_to_show_idx)

        self._win = pg.GraphicsLayoutWidget(
            size=geometry[2:],
            title=f'Stream Viewer: {self.scope.stream_name}')

        self._win.show()
        self.init_canvas()
        self.init_plot()

        self.events = list()

        self.backend_initialized = True
        self._timer = QtCore.QTimer(self._win)
        self._timer.timeout.connect(self.update_loop)

    def init_variables(self, x_scale, y_scale, channels_to_show_idx):
        self.x_scale = x_scale  # duration in seconds
        self.y_scale = y_scale  # amplitude scale in uV
        self.available_colors = np.random.uniform(
            size=(self.scope.n_channels, 3), low=128, high=230)
        self.channels_to_show_idx = channels_to_show_idx
        self.init_n_samples_plot()

    def init_n_samples_plot(self):
        self.delta_with_buffer = self.scope.duration_buffer-self.x_scale
        self.n_samples_plot = math.ceil(self.x_scale * self.scope.sample_rate)

    def init_canvas(self):
        self._plot_handler = self._win.addPlot() # pyqtgraph.PlotItem
        # We want a lightweight scope, so we downsample the plotting to 64 Hz
        subsampling_ratio = self.scope.sample_rate / 64
        self._plot_handler.setDownsampling(ds=subsampling_ratio,
                                            auto=None, mode='mean')
        # TODO: Not sure the downsampling is required, feels laggy either way.

        self.init_range()
        self.init_y_axis()
        self.init_x_axis()

    def init_range(self):
        self.y_range = [1.5*self.y_scale,
                        -self.y_scale*(len(self.channels_to_show_idx)+1)]
        self._plot_handler.setRange(
            xRange=[0, self.x_scale],
            yRange=self.y_range)
        self._plot_handler.disableAutoRange()
        self._plot_handler.showGrid(y=True)

    def init_y_axis(self):
        yticks = [(-k*self.y_scale, self.scope.channels_labels[idx]) \
                  for k, idx in enumerate(self.channels_to_show_idx)]
        ticks = [yticks, []] # [major, minor]
        self._plot_handler.getAxis('left').setTicks(ticks)
        self._plot_handler.setLabel(
            axis='left', text=f'Scale (uV): {self.y_scale}')

    def init_x_axis(self):
        self.x_arr = np.arange(self.n_samples_plot) / self.scope.sample_rate
        self._plot_handler.setLabel(axis='bottom', text='Time (s)')

    # ------------------------- Init plot -------------------------
    def init_plot(self):
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
        self.offset =  np.arange(
            0, -len(self.channels_to_show_idx)*self.y_scale, -self.y_scale)

    # ------------------------ Trigger Events ----------------------
    def update_LPT_events(self, trigger_arr):
        events_trigger_arr_idx = np.where(trigger_arr != 0)[0]
        events_values = trigger_arr[events_trigger_arr_idx]

        for k, ev_value in enumerate(events_values):
            position_buffer = self.scope.duration_buffer - \
                (trigger_arr.shape[0] - events_trigger_arr_idx[k])/self.scope.sample_rate
            position_plot = position_buffer - self.delta_with_buffer

            event =  _Event(
                event_type='LPT',
                event_value=ev_value,
                position_buffer=position_buffer,
                position_plot=position_plot,
                plot_handler=self._plot_handler,
                plot_y_scale=self.y_scale)

            if position_plot >= 0:
                event.addEventPlot()

            self.events.append(event)

    def clean_up_events(self):
        for ev in self.events:
            if ev.position_plot < 0:
                ev.removeEventPlot()

        for k in range(len(self.events)-1, -1, -1):
            if self.events[k].position_buffer < 0:
                del self.events[k]

    # -------------------------- Main Loop -------------------------
    def start_timer(self):
        self._timer.start(20)

    def update_loop(self):
        self.scope.update_loop()
        if len(self.scope._ts_list) > 0:
            for k, idx in enumerate(self.channels_to_show_idx):
                self.plots[idx].setData(
                    x=self.x_arr,
                    y=self.scope.data_buffer[idx, -self.n_samples_plot:]+self.offset[k])

            # Update existing events position
            for ev in self.events:
                ev.update_position(
                    ev.position_buffer - len(self.scope._ts_list)/self.scope.sample_rate,
                    ev.position_plot - len(self.scope._ts_list)/self.scope.sample_rate)
            # Add new events entering the buffer
            self.update_LPT_events(self.scope.trigger_buffer[-len(self.scope._ts_list):])
            # Hide/Remove events exiting window and buffer
            self.clean_up_events()

    # ------------------------ Update program ----------------------
    def update_x_scale(self, new_x_scale):
        if self.backend_initialized:
            self.x_scale = new_x_scale
            self.init_n_samples_plot()
            self.init_range()
            self.init_x_axis()

            for ev in self.events:
                ev.update_position(
                    ev.position_buffer,
                    ev.position_buffer - self.delta_with_buffer)
                if ev.position_plot >= 0:
                    ev.addEventPlot()
                else:
                    ev.removeEventPlot()

    def update_y_scale(self, new_y_scale):
        if self.backend_initialized:
            self.y_scale = new_y_scale
            self.init_range()
            self.init_y_axis()
            self.init_plotting_channel_offset()

            for ev in self.events:
                ev.update_scales(self.y_scale)

    def update_channels_to_show_idx(self, new_channels_to_show_idx):
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

    # --------------------------- Events ---------------------------
    def close(self):
        self._timer.stop()
        self._win.close()

class _Event:
    pens = {'KEY': pg.mkColor(255, 0, 0),
            'LPT': pg.mkColor(0, 255, 0)}

    def __init__(self, event_type, event_value, position_buffer, position_plot,
                 plot_handler, plot_y_scale):
        assert event_type in self.pens.keys()
        self.event_type = event_type
        self.event_value = event_value
        self.position_buffer = position_buffer # In time (s)
        self.position_plot = position_plot # In time (s)

        self.plot_handler = plot_handler
        self.plot_y_scale = plot_y_scale

        self.plotted = False

    def addEventPlot(self):
        if not self.plotted:
            self.LineItem = pg.InfiniteLine(pos=self.position_plot, pen=self.pens[self.event_type])
            self.plot_handler.addItem(self.LineItem)

            self.TextItem = pg.TextItem(str(self.event_value), anchor=(0.5, 1),
                               fill=(0, 0, 0), color=self.pens[self.event_type])
            self.TextItem.setPos(self.position_plot, 1.5*self.plot_y_scale)
            self.plot_handler.addItem(self.TextItem)
            self.plotted = True

    def update_scales(self, plot_y_scale):
        self.plot_y_scale = plot_y_scale
        self._update()

    def update_position(self, position_buffer, position_plot):
        self.position_buffer = position_buffer
        self.position_plot = position_plot
        self._update()

    def _update(self):
        self.LineItem.setValue(self.position_plot)
        self.TextItem.setPos(self.position_plot, 1.5*self.plot_y_scale)

    def removeEventPlot(self):
        if self.plotted:
            self.plot_handler.removeItem(self.LineItem)
            self.plot_handler.removeItem(self.TextItem)
            self.plotted = False

    def __del__(self):
        try:
            self.removeEventPlot()
        except:
            pass
