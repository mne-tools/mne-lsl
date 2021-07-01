"""
Module for signal visualization.

Visualize signals in real time with spectral filtering, common average
filtering options and real-time FFT.

TODO:
    ScopeController:
        - Add Key Press Events (Close, Pause, ...)
        - Add Key triggers
        - Add a trigger selection to allow Key trigger to be saved in the
        recorded file.
        - Reorganize / Improve GUI.

    Vispy backend:
        - Add channels names + axis labels
        - Add trigger events (Common _TriggerEvent class for every backends?)
        - Fix position when creating the window
        - Define second value of u_scale based on the y_scale.
        - Fix the non-fatal errors raised.

    PyQt5 backend:
        - Subsampling doesn't look needed for 64 channels@512 Hz. Set a dynamic
        application of subsampling based on number of channels and sample rate.
        - Fix position when creating the graphic window

    PyQt6 backend:
        - Is there an upgrade to be done from PyQt5?

    - Set the nrows, ncols parameter for every backend based on the number of
    channels. e.g. 128 channels system could be displayed on 2 columns of 64
    channels.
"""

from .stream_viewer import StreamViewer
