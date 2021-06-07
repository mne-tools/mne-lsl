#!/usr/bin/env python
"""
Vispy Canvas for Neurodecode's StreamViewer.
"""

import vispy
vispy.use("pyqt5")

from neurodecode.stream_receiver import StreamReceiver

from scipy.signal import butter, sosfilt, sosfilt_zi
from vispy import gloo
from vispy import app
import numpy as np
import time
import math

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


class _CanvasScope(app.Canvas):
    # ---------------------------- INIT ----------------------------
    def __init__(self, stream_name):
        # Init loop
        self.init_loop(stream_name)
        self.init_arr(duration_plot=10)
        self.init_gloo(nrows=self.n_channels, ncols=1)
        self.init_filter(low=1., high=40.)
        self._timer = app.Timer(0.020, connect=self.update_loop, start=True)
        self.show()

    def init_loop(self, stream_name, bufsize=0.2, winsize=0.2):
        """
        Instance a StreamReceiver and extract info from the stream.
        """
        self.stream_name = stream_name
        self.sr = StreamReceiver(bufsize=bufsize, winsize=winsize,
                                 stream_name=self.stream_name)
        self.sr.streams[self.stream_name].blocking = False
        time.sleep(bufsize) # Delay to fill the LSL buffer.

        self.n_channels = len(
            self.sr.streams[self.stream_name].ch_list[1:])
        self.sample_rate = int(
            self.sr.streams[self.stream_name].sample_rate)

    def init_arr(self, duration_plot):
        self.n_samples_plot = duration_plot * self.sample_rate
        self.trigger = np.zeros(self.n_samples_plot)
        self.data = np.zeros((self.n_channels, self.n_samples_plot),
                             dtype=np.float32)
        self._ts_list = list()

    def init_gloo(self, nrows, ncols):
        # Colors for each subplot
        self.color = np.repeat(
            np.random.uniform(size=(nrows*ncols, 3), low=.5, high=.9),
            self.n_samples_plot, axis=0).astype(np.float32)
        # Index/Position of each point (vertex) on the Canvas
        self.index = np.c_[
            np.repeat(np.repeat(np.arange(ncols), nrows), self.n_samples_plot),
            np.repeat(np.tile(np.arange(nrows), ncols), self.n_samples_plot),
            np.tile(np.arange(self.n_samples_plot), nrows*ncols)].astype(np.float32)

        app.Canvas.__init__(self, title=f'Stream Viewer: {self.stream_name}',
                            keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = self.data.reshape(-1, 1)
        self.program['a_color'] = self.color
        self.program['a_index'] = self.index
        self.program['u_scale'] = (1., 1/20)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = self.n_samples_plot
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def init_filter(self, low, high):
        self.bp_low = low / (0.5 * self.sample_rate)
        self.bp_high = high / (0.5 * self.sample_rate)
        self.sos = butter(2, [self.bp_low, self.bp_high],
                     btype='band', output='sos')
        self.zi_coeff = sosfilt_zi(self.sos).reshape((self.sos.shape[0], 2, 1))
        self.zi = None

    # -------------------------- Main Loop --------------------------
    def update_loop(self, event):
        self.read_lsl_stream()
        if len(self._ts_list) > 0:
            self.filter_signal()
            self.data = np.roll(self.data, -len(self._ts_list), axis=1)
            self.data[:, -len(self._ts_list):] = np.transpose(self.window)

            self.program['a_position'].set_data(
                self.data.ravel().astype(np.float32))
            self.update()

    def read_lsl_stream(self):
        self.sr.acquire()
        data, self._ts_list = self.sr.get_buffer()
        self.sr.reset_all_buffers()

        if len(self._ts_list) == 0:
            return

        self.trigger = data[:, 0].reshape((-1, 1))               # (samples, )
        self.window = data[:, 1:].reshape((-1, self.n_channels)) # (samples, channels)

    def filter_signal(self):
        if self.zi is None:
            self.zi = self.zi_coeff * np.mean(self.window, axis=0) # Multiply by DC
        self.window, self.zi = sosfilt(self.sos, self.window, 0, self.zi)

    # --------------------------- Events ---------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_mouse_wheel(self, event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = self.program['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
                                    scale_y * math.exp(0.0*dx))
        self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')

if __name__ == '__main__':
    stream_name = 'StreamPlayer'
    c = _CanvasScope(stream_name)
    app.run()
