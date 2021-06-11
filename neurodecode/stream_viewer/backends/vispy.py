"""
Vispy Canvas for Neurodecode's StreamViewer.
"""

from vispy import app
from vispy import gloo
import numpy as np
import math
import vispy
vispy.use("pyqt5")


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


class _BackendVispy(app.Canvas):
    """
    The Vispy backend for neurodecode's StreamViewer.

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
        self._show_LPT_events = False
        self._show_Key_events = False

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
        self.init_program_variables()
        self.init_gloo(geometry)
        self.show()

    def init_variables(self, x_scale, y_scale, channels_to_show_idx):
        """
        Initialize variables.
        """
        self.x_scale = x_scale  # duration in seconds
        self.y_scale = y_scale  # amplitude scale in uV
        self.available_colors = np.random.uniform(
            size=(self.scope.n_channels, 3), low=.5, high=.9)
        self.channels_to_show_idx = channels_to_show_idx
        self.init_nrows_ncols()
        self.init_n_samples_plot()

    def init_nrows_ncols(self):
        """
        Initialize the number of rows and columns on which the channels are displayed.
        """
        self.nrows = len(self.channels_to_show_idx)
        self.ncols = 1

    def init_n_samples_plot(self):
        """
        Initialize the number of samples present in the plotting window, and
        the duration difference between the plotting window with the buffer.
        """
        self.delta_with_buffer = self.scope.duration_buffer - self.x_scale
        self.n_samples_plot = math.ceil(self.x_scale * self.scope.sample_rate)

    # ------------------------ Init program -----------------------
    def init_program_variables(self):
        """
        Initialize the variables of the Vispy program. The variables are:
            a_color : the color of every vertex
            a_index : the position of every vertex
            u_scale : the (x, y) scaling
            u_size : the number of rows and columns as (row, col).
            u_n : the number of samples per

        Initialize the timer calling the main update loop every 20 ms.
        """
        self.init_a_color()
        self.init_a_index()
        self.init_u_scale()
        self.init_u_size()
        self.init_u_n()

        self.backend_initialized = True
        self._timer = app.Timer(0.020, connect=self.update_loop, start=False)

    def init_a_color(self):
        """
        Initialize the vertex the colors.
        """
        self.a_color = np.repeat(
            self.available_colors[self.channels_to_show_idx, :],
            self.n_samples_plot, axis=0).astype(np.float32)

    def init_a_index(self):
        """
        Initialize the vertex positions on the Canvas.
        """
        self.a_index = np.c_[
            np.repeat(
                np.repeat(np.arange(self.ncols),
                          self.nrows),
                self.n_samples_plot),
            np.repeat(
                np.tile(np.arange(self.nrows),
                        self.ncols),
                self.n_samples_plot),
            np.tile(
                np.arange(self.n_samples_plot),
                self.nrows*self.ncols)].astype(np.float32)

    def init_u_scale(self):
        """
        Initialize the X/Y scale/range.
        """
        self.u_scale = (1., 1/20)

    def init_u_size(self):
        """
        Initialize the number of rows and columns.
        """
        self.u_size = (self.nrows, self.ncols)

    def init_u_n(self):
        """
        Initilaize the number of sample per signal.
        """
        self.u_n = self.n_samples_plot

    def init_gloo(self, geometry):
        """
        Initialize the Canvas and the Vispy gloo.
        """
        app.Canvas.__init__(
            self, title=f'Stream Viewer: {self.scope.stream_name}',
            size=geometry[2:], position=geometry[:2],
            keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = self.scope.data_buffer[
            self.channels_to_show_idx, -self.n_samples_plot:].ravel().astype(np.float32)
        self.program['a_color'] = self.a_color
        self.program['a_index'] = self.a_index
        self.program['u_scale'] = self.u_scale
        self.program['u_size'] = self.u_size
        self.program['u_n'] = self.u_n
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    # -------------------------- Main Loop -------------------------
    def start_timer(self):
        """
        Start the update loop on a 20ms timer.
        """
        self._timer.start()

    def update_loop(self, event):
        """
        Main update loop retrieving data from the scope's buffer and updating
        the Canvas.
        """
        self.scope.update_loop()

        if len(self.scope._ts_list) > 0:
            self.program['a_position'].set_data(
                self.scope.data_buffer[
                    self.channels_to_show_idx,
                    -self.n_samples_plot:].ravel().astype(np.float32))
            self.update()

    # ------------------------ Update program ----------------------
    def update_x_scale(self, new_x_scale):
        """
        Called when the user changes the X-axis range/scale, i.e. the duration
        of the plotting window.
        """
        if self.backend_initialized:
            self.x_scale = new_x_scale
            self.init_n_samples_plot()
            self.init_a_color()
            self.init_a_index()
            self.init_u_n()

            self.program['a_color'].set_data(self.a_color)
            self.program['a_index'].set_data(self.a_index)
            self.program['u_n'] = self.u_n
            self.update()

    def update_y_scale(self, new_y_scale):
        """
        Called when the user changes the signal range/scale.
        """
        if self.backend_initialized:
            self.y_scale = new_y_scale
            self.init_u_scale()

            self.program['u_scale'] = self.u_scale
            self.update()

    def update_channels_to_show_idx(self, new_channels_to_show_idx):
        """
        Called when the user changes the selection of channels.
        """
        if self.backend_initialized:
            self.channels_to_show_idx = new_channels_to_show_idx
            self.init_nrows_ncols()
            self.init_a_color()
            self.init_a_index()
            self.init_u_size()

            self.program['a_color'].set_data(self.a_color)
            self.program['a_index'].set_data(self.a_index)
            self.program['u_size'] = self.u_size
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
        self.program.draw('line_strip')

    def close(self):
        """
        Stops the update loop and close the window.
        """
        self._timer.stop()
        super().close()
