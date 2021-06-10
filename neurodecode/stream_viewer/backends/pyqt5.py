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
        self._plot_handler.setRange(
            xRange=[0, self.x_scale],
            yRange=[self.y_scale, -self.y_scale*len(self.channels_to_show_idx)])
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

    def init_plotting_channel_offset(self):
        self.offset =  np.arange(
            0, -len(self.channels_to_show_idx)*self.y_scale, -self.y_scale)

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

    # ------------------------ Update program ----------------------
    def update_x_scale(self, new_x_scale):
        if self.backend_initialized:
            self.x_scale = new_x_scale
            self.init_n_samples_plot()
            self.init_range()
            self.init_x_axis()

    def update_y_scale(self, new_y_scale):
        if self.backend_initialized:
            self.y_scale = new_y_scale
            self.init_range()
            self.init_y_axis()
            self.init_plotting_channel_offset()

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
