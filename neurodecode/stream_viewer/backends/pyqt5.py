"""
PyQt5 Canvas for Neurodecode's StreamViewer.
"""
import math
import numpy as np

import pyqtgraph as pg


class _BackendPyQt5:
    # ---------------------------- Init ---------------------------
    def __init__(self, scope):
        self.scope = scope
        self.backend_initialized = False

    def init_backend(self, geometry, x_scale, y_scale, channels_to_show_idx):
        assert len(channels_to_show_idx) <= self.scope.n_channels
        self.init_variables(x_scale, y_scale, channels_to_show_idx)

        self._win = pg.GraphicsLayoutWidget(
            size=geometry[2:],
            title=f'Stream Viewer: {self.scope.stream_name}')

        self._win.show()
        self.init_canvas()
        self.init_data_plot()
        self.init_plot()

    def init_variables(self, x_scale, y_scale, channels_to_show_idx):
        self.x_scale = x_scale  # duration in seconds
        self.y_scale = y_scale  # amplitude scale in uV
        self.available_colors = np.random.uniform(
            size=(self.scope.n_channels, 3), low=.5, high=.9)
        self.channels_to_show_idx = channels_to_show_idx
        self.init_n_samples_plot()

    def init_n_samples_plot(self):
        self.n_samples_plot = math.ceil(self.x_scale * self.scope.sample_rate)

    def init_canvas(self):
        self._plot_handler = self._win.addPlot()
        # We want a lightweight scope, so we downsample the plotting to 64 Hz
        subsampling_ratio = self.scope.sample_rate / 64
        self._plot_handler.setDownsampling(ds=subsampling_ratio,
                                           auto=None, mode='mean')

        # Range
        self._plot_handler.setRange(
            xRange=[0, self.x_scale],
            yRange=[+1.5*self.y_scale,
                    -0.5*self.y_scale - self.y_scale*self.scope.n_channels])
        self._plot_handler.disableAutoRange()
        self._plot_handler.showGrid(y=True)

        # # Y-axis
        yticks = [(-k*self.y_scale, self.scope.channels_labels[idx]) \
                  for k, idx in enumerate(self.channels_to_show_idx)]
        ticks = [yticks, []] # [major, minor]
        self._plot_handler.getAxis('left').setTicks(ticks)
        self._plot_handler.setLabel(
            axis='left', text=f'Scale (uV): {self.y_scale}')

        # # X-axis
        self.x_arr = np.arange(self.n_samples_plot) / self.scope.sample_rate
        self._plot_handler.setLabel(axis='bottom', text='Time (s)')

    # ------------------------ Init program -----------------------
    def init_data_plot(self):
        self.data_plot = np.zeros((len(self.channels_to_show_idx),
                                   self.n_samples_plot),
                                  dtype=np.float32)

    def init_plot(self):
        self.plots = list()
        for k, idx in enumerate(self.channels_to_show_idx):
            self.plots.append(self._plot_handler.plot(
                x=self.x_arr, y=self.data_plot[idx, :],
                pen=pg.mkColor(self.available_colors[idx, :])))

    # ------------------------ Update program ----------------------
    def update_x_scale(self, new_x_scale):
        pass

    def update_y_scale(self, new_y_scale):
        pass

    def update_channels_to_show_idx(self, new_channels_to_show_idx):
        pass

    # --------------------------- Events ---------------------------
    def close(self):
        self._win.close()
