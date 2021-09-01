"""
Vispy Canvas for BSL's StreamViewer.
Code adapted from Vispy official example:
https://github.com/vispy/vispy/blob/main/examples/demo/gloo/realtime_signals.py
"""
import numpy as np

from ._backend import _Backend, _Event
from ...utils._docs import fill_doc, copy_doc
from ...utils._imports import import_optional_dependency

vispy = import_optional_dependency(
    "vispy", extra="Install Vispy for backend support.")
vispy.use("pyqt5")


# --------------------------------- Data -------------------------------------
VERT_SHADER_data = """
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
    float x = -0.9 + 1.9*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./ncols, 1./nrows);
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
                  -0.9 + 1.8*(a_index.y+0.5) / nrows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_color = vec4(a_color, 1.);
    v_index = a_index;
}
"""

FRAG_SHADER_data = """
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


@fill_doc
class _BackendVispy(_Backend, vispy.app.Canvas):
    """
    The Vispy backend for BSL's StreamViewer.

    Parameters
    ----------
    %(viewer_scope)s
    %(viewer_backend_geometry)s
    %(viewer_backend_xRange)s
    %(viewer_backend_yRange)s
    """

    # ---------------------------- Init ---------------------------
    def __init__(self, scope, geometry, xRange, yRange):
        super().__init__(scope, geometry, xRange, yRange)
        self._trigger_events = list()

        # Variables
        self._available_colors = np.random.uniform(
            size=(self._scope.nb_channels, 3), low=.5, high=.9)
        self._init_variables()
        self._init_program_variables()

        # Canvas
        self._init_gloo(geometry)
        self.show()

    @copy_doc(_Backend._init_variables)
    def _init_variables(self):
        super()._init_variables()

        # Number of channels
        self._nrows = len(self._scope.selected_channels)
        self._ncols = 1

    # ------------------------ Init program -----------------------
    def _init_program_variables(self):
        """
        Initialize the variables of the Vispy program. The variables are:
            - a_color : the color of every vertex
            - a_index : the position of every vertex
            - u_scale : the (x, y) scaling
            - u_size : the number of rows and columns as (row, col).
            - u_n : the number of samples

        Initialize the timer calling the main update loop every 20 ms.
        """
        self._init_a_color()
        self._init_a_index()
        self._init_u_scale()
        self._init_u_size()
        self._init_u_n()

        self._timer = vispy.app.Timer(
            0.020, connect=self._update_loop, start=False)

    def _init_a_color(self):
        """
        Initialize the vertex colors.
        """
        self._a_color = np.repeat(
            self._available_colors[self._scope.selected_channels, :],
            self._duration_plot_samples, axis=0).astype(np.float32, copy=False)

    def _init_a_index(self):
        """
        Initialize the vertex positions on the Canvas.
        """
        self._a_index = np.c_[
            np.repeat(
                np.repeat(np.arange(self._ncols),
                          self._nrows),
                self._duration_plot_samples),
            np.repeat(
                np.tile(np.arange(self._nrows),
                        self._ncols),
                self._duration_plot_samples),
            np.tile(
                np.arange(self._duration_plot_samples),
                self._nrows*self._ncols)].astype(np.float32, copy=False)

    def _init_u_scale(self):
        """
        Initialize the X/Y scale/range.
        """
        self._u_scale = (1., 1/self._yRange)

    def _init_u_size(self):
        """
        Initialize the number of rows and columns.
        """
        self._u_size = (self._nrows, self._ncols)

    def _init_u_n(self):
        """
        Initilaize the number of samples.
        """
        self._u_n = self._duration_plot_samples

    # --------------------------- Canvas ---------------------------
    def _init_gloo(self, geometry):
        """
        Initialize the Canvas and the Vispy gloo.
        """
        vispy.app.Canvas.__init__(
            self, title=f'Stream Viewer: {self._scope.stream_name}',
            size=geometry[2:], position=geometry[:2],
            keys='interactive')

        # Data program
        self._program_data = vispy.gloo.Program(
            VERT_SHADER_data, FRAG_SHADER_data)
        self._program_data['a_position'] = self._scope.data_buffer[
            self._scope.selected_channels[::-1],
            -self._duration_plot_samples:].ravel().astype(
                np.float32, copy=False)
        self._program_data['a_color'] = self._a_color
        self._program_data['a_index'] = self._a_index
        self._program_data['u_scale'] = self._u_scale
        self._program_data['u_size'] = self._u_size
        self._program_data['u_n'] = self._u_n

        vispy.gloo.set_viewport(0, 0, *self.physical_size)
        vispy.gloo.set_state(
            clear_color='black', blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha'))

    # ------------------------ Trigger Events ----------------------
    @copy_doc(_Backend._update_LPT_trigger_events)
    def _update_LPT_trigger_events(self, trigger_arr):
        events_trigger_arr_idx = np.where(trigger_arr != 0)[0]
        events_values = trigger_arr[events_trigger_arr_idx]

        for k, event_value in enumerate(events_values):
            position_buffer = self._scope.duration_buffer - \
                (trigger_arr.shape[0] - events_trigger_arr_idx[k]) \
                / self._scope.sample_rate
            position_plot = position_buffer - self._delta_with_buffer

            event = _TriggerEvent(
                event_type='LPT',
                event_value=event_value,
                position_buffer=position_buffer,
                position_plot=position_plot)

            self._trigger_events.append(event)

    # -------------------------- Main Loop -------------------------
    @copy_doc(_Backend.start_timer)
    def start_timer(self):
        self._timer.start()

    @copy_doc(_Backend._update_loop)
    def _update_loop(self, event):
        super()._update_loop()

        if len(self._scope.ts_list) > 0:
            self._program_data['a_position'].set_data(
                self._scope.data_buffer[
                    self._scope.selected_channels[::-1],
                    -self._duration_plot_samples:].ravel().astype(
                        np.float32, copy=False))

            # Update existing events position
            for event in self._trigger_events:
                event.position_buffer = event.position_buffer \
                    - len(self._scope.ts_list) / self._scope.sample_rate
            # Add new events entering the buffer
            self._update_LPT_trigger_events(
                self._scope.trigger_buffer[-len(self._scope.ts_list):])
            # Remove events exiting window and buffer
            self._clean_up_trigger_events()

            self.update()

    # --------------------------- Events ---------------------------
    def on_resize(self, event):
        """
        Called when the window is resized.
        """
        vispy.gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        vispy.gloo.clear()
        self._program_data.draw('line_strip')

    @copy_doc(_Backend.close)
    def close(self):
        self._timer.stop()
        vispy.app.Canvas.close(self)

    # ------------------------ Update program ----------------------
    @_Backend.xRange.setter
    @copy_doc(_Backend.xRange.setter)
    def xRange(self, xRange):
        self._xRange = xRange
        self._init_variables()
        self._init_a_color()
        self._init_a_index()
        self._init_u_n()

        self._program_data['a_color'].set_data(self._a_color)
        self._program_data['a_index'].set_data(self._a_index)
        self._program_data['u_n'] = self._u_n

        for event in self._trigger_events:
            event.position_plot = event.position_buffer-self._delta_with_buffer

        self.update()

    @_Backend.yRange.setter
    @copy_doc(_Backend.yRange.setter)
    def yRange(self, yRange):
        self._yRange = yRange
        self._init_u_scale()

        self._program_data['u_scale'] = self._u_scale
        self.update()

    @_Backend.selected_channels.setter
    @copy_doc(_Backend.selected_channels.setter)
    def selected_channels(self, selected_channels):
        self._selected_channels = selected_channels
        self._init_variables()
        self._init_a_color()
        self._init_a_index()
        self._init_u_size()

        self._program_data['a_color'].set_data(self._a_color)
        self._program_data['a_index'].set_data(self._a_index)
        self._program_data['u_size'] = self._u_size
        self.update()

    @_Backend.show_LPT_trigger_events.setter
    @copy_doc(_Backend.show_LPT_trigger_events.setter)
    def show_LPT_trigger_events(self, show_LPT_trigger_events):
        self._show_LPT_trigger_events = show_LPT_trigger_events


@fill_doc
class _TriggerEvent(_Event):
    """
    Class defining a trigger event for the vispy backend.

    Parameters
    ----------
    %(viewer_event_type)s
    %(viewer_event_value)s
    %(viewer_position_buffer)s
    %(viewer_position_plot)s
    """
    colors = {'LPT': np.array([0., 1.0, 0.], dtype=np.float32)}

    def __init__(self, event_type, event_value,
                 position_buffer, position_plot):
        super().__init__(event_type, event_value,
                         position_buffer, position_plot)
