"""
Vispy Canvas for Neurodecode's StreamViewer.
Code adapted from Vispy official example:
https://github.com/vispy/vispy/blob/main/examples/demo/gloo/realtime_signals.py
"""
import math

import vispy
import numpy as np
from vispy import app
from vispy import gloo
vispy.use("pyqt5")

from ._backend import _Backend


VERT_SHADER = """
#version 120
// y coordinate of the position.
attribute float a_position;
// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
// Number of samples per signal.
uniform float u_n;
// Color.
attribute vec3 a_color;
varying vec4 v_color;
void main() {
    float nrows = u_size.x;
    float ncols = u_size.y;
    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -1 + 2*(a_index.y+.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_color = vec4(a_color, 1.);
    v_index = a_index;
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
varying vec3 v_index;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;
}
"""


class _BackendVispy(_Backend, app.Canvas):
    """
    The Vispy backend for neurodecode's StreamViewer.

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
        self._init_program_variables()
        self._init_gloo(geometry)
        self.show()

    def _init_variables(self, x_scale, y_scale, channels_to_show_idx):
        """
        Initialize variables.
        """
        super()._init_variables(x_scale, y_scale, channels_to_show_idx)
        self._available_colors = np.random.uniform(
            size=(self._scope.n_channels, 3), low=.5, high=.9)
        self._init_nrows_ncols()
        self._init_n_samples_plot()

    def _init_nrows_ncols(self):
        """
        Initialize the number of rows and columns on which the channels are
        displayed.
        """
        self.nrows = len(self._channels_to_show_idx)
        self.ncols = 1

    def _init_n_samples_plot(self):
        """
        Initialize the number of samples present in the plotting window, and
        the duration difference between the plotting window with the buffer.
        """
        self._delta_with_buffer = self._scope.duration_buffer - self._x_scale
        self._n_samples_plot = math.ceil(self._x_scale * self._scope.sample_rate)

    # ------------------------ Init program -----------------------
    def _init_program_variables(self):
        """
        Initialize the variables of the Vispy program. The variables are:
            a_color : the color of every vertex
            a_index : the position of every vertex
            u_scale : the (x, y) scaling
            u_size : the number of rows and columns as (row, col).
            u_n : the number of samples per

        Initialize the timer calling the main update loop every 20 ms.
        """
        self._init_a_color()
        self._init_a_index()
        self._init_u_scale()
        self._init_u_size()
        self._init_u_n()

        self._backend_initialized = True
        self._timer = app.Timer(0.020, connect=self._update_loop, start=False)

    def _init_a_color(self):
        """
        Initialize the vertex the colors.
        """
        self._a_color = np.repeat(
            self._available_colors[self._channels_to_show_idx, :],
            self._n_samples_plot, axis=0).astype(np.float32)

    def _init_a_index(self):
        """
        Initialize the vertex positions on the Canvas.
        """
        self._a_index = np.c_[
            np.repeat(
                np.repeat(np.arange(self.ncols),
                          self.nrows),
                self._n_samples_plot),
            np.repeat(
                np.tile(np.arange(self.nrows),
                        self.ncols),
                self._n_samples_plot),
            np.tile(
                np.arange(self._n_samples_plot),
                self.nrows*self.ncols)].astype(np.float32)

    def _init_u_scale(self):
        """
        Initialize the X/Y scale/range.
        """
        self._u_scale = (1., 1/20)

    def _init_u_size(self):
        """
        Initialize the number of rows and columns.
        """
        self._u_size = (self.nrows, self.ncols)

    def _init_u_n(self):
        """
        Initilaize the number of sample per signal.
        """
        self._u_n = self._n_samples_plot

    def _init_gloo(self, geometry):
        """
        Initialize the Canvas and the Vispy gloo.
        """
        app.Canvas.__init__(
            self, title=f'Stream Viewer: {self._scope.stream_name}',
            size=geometry[2:], position=geometry[:2],
            keys='interactive')
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._program['a_position'] = self._scope.data_buffer[
            self._channels_to_show_idx,
            -self._n_samples_plot:].ravel().astype(np.float32)
        self._program['a_color'] = self._a_color
        self._program['a_index'] = self._a_index
        self._program['u_scale'] = self._u_scale
        self._program['u_size'] = self._u_size
        self._program['u_n'] = self._u_n
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    # -------------------------- Main Loop -------------------------
    def start_timer(self):
        """
        Start the update loop on a 20ms timer.
        """
        self._timer.start()

    def _update_loop(self, event):
        """
        Main update loop retrieving data from the scope's buffer and updating
        the Canvas.
        """
        super()._update_loop()

        if len(self._scope.ts_list) > 0:
            self._program['a_position'].set_data(
                self._scope.data_buffer[
                    self._channels_to_show_idx,
                    -self._n_samples_plot:].ravel().astype(np.float32))
            self.update()

    # ------------------------ Update program ----------------------
    # TODO: Moved as setters
    def update_x_scale(self, new_x_scale):
        """
        Called when the user changes the X-axis range/scale, i.e. the duration
        of the plotting window.
        """
        if self._backend_initialized:
            self._x_scale = new_x_scale
            self._init_n_samples_plot()
            self._init_a_color()
            self._init_a_index()
            self._init_u_n()

            self._program['a_color'].set_data(self._a_color)
            self._program['a_index'].set_data(self._a_index)
            self._program['u_n'] = self._u_n
            self.update()

    def update_y_scale(self, new_y_scale):
        """
        Called when the user changes the signal range/scale.
        """
        if self._backend_initialized:
            self._y_scale = new_y_scale
            self._init_u_scale()

            self._program['u_scale'] = self._u_scale
            self.update()

    def update_channels_to_show_idx(self, new_channels_to_show_idx):
        """
        Called when the user changes the selection of channels.
        """
        if self._backend_initialized:
            self._channels_to_show_idx = new_channels_to_show_idx
            self._init_nrows_ncols()
            self._init_a_color()
            self._init_a_index()
            self._init_u_size()

            self._program['a_color'].set_data(self._a_color)
            self._program['a_index'].set_data(self._a_index)
            self._program['u_size'] = self._u_size
            self.update()

    def update_show_LPT_events(self):
        """
        Called when the user ticks or untick the show_LPT_events box.
        """
        pass

    # --------------------------- Events ---------------------------
    def on_resize(self, event):
        """
        Called when the window is resized.
        """
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        gloo.clear()
        self._program.draw('line_strip')

    def close(self):
        """
        Stops the update loop and close the window.
        """
        self._timer.stop()
        super().close()
