"""Fill docstrings to avoid redundant docstrings in multiple files.

Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""

import sys
from typing import Callable, Dict, List

# ------------------------- Documentation dictionary -------------------------
docdict: Dict[str, str] = dict()

# -----------------------------------------------
docdict[
    "stream_name"
] = """
stream_name : list | str | None
    Servers' name or list of servers' name to connect to.
    If ``None``, connects to all the available streams."""
docdict[
    "verbose"
] = """
verbose : int | str | bool | None
    Sets the verbosity level. The verbosity increases gradually between
    ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"`` and ``"DEBUG"``.
    If None is provided, the verbosity is set to ``"WARNING"``.
    If a bool is provided, the verbosity is set to ``"WARNING"`` for False and
    to ``"INFO"`` for True."""

# -----------------------------------------------
# Stream Receiver
docdict[
    "receiver_get_stream_name"
] = """
stream_name : str | None
    Name of the stream from which data is retrieved.
    Can be set to ``None`` (default) if the StreamReceiver is connected to a
    single stream."""
docdict[
    "receiver_get_return_raw"
] = """
return_raw : bool
    By default (``False``), data is returned as a `~numpy.array` of shape
    ``(samples, channels)``. If set to ``True``, the StreamReceiver will
    attempt to return data as a MNE Raw instances."""
docdict[
    "receiver_data"
] = """
data : `~numpy.array`
    Data ``(samples, channels)``."""
docdict[
    "receiver_timestamps"
] = """
timestamps : `~numpy.array`
    Data's timestamps ``(samples, )``."""
docdict[
    "receiver_get_unit"
] = """
Returns a raw data array in the unit streamed by the LSL outlet. For conversion
the corresponding scaling factor must be set for each stream, with e.g. for a
stream in uV to convert to V:

.. code-block:: python

    sr.streams['stream_to_convert'].scaling_factor = 1e-6"""
docdict[
    "receiver_bufsize"
] = """
bufsize : int | float
    Buffer's size ``[secs]``. ``MAX_BUF_SIZE`` (def: 1-day) is the maximum
    size. Large buffer may lead to a delay if not pulled frequently."""
docdict[
    "receiver_winsize"
] = """
winsize : int | float
    Window's size ``[secs]``. Must be smaller than the buffer's size."""

# Not read by sphinx autodoc
docdict[
    "receiver_streamInfo"
] = """
streamInfo : LSL StreamInfo.
    Contain all the info from the LSL stream to connect to."""
docdict[
    "receiver_tslist"
] = """
tslist : list
    Data's timestamps (samples, )."""

# -----------------------------------------------
# Stream Recorder
docdict[
    "recorder_record_dir"
] = """
record_dir : None | path-like
    Path to the directory where data will be saved. If the directory does not
    exist, it is created. If ``None``, the current working directory is used.
"""
docdict[
    "recorder_fname"
] = """
fname : None | str
    File name stem used to create the files. The StreamRecorder creates 2 files
    plus an optional third if a software trigger was used, respecting the
    following naming::

      PCL: '{fname}-[stream_name]-raw.pcl'
      FIF: '{fname}-[stream_name]-raw.fif'
      (optional) SOFTWARE trigger events: '{fname}-eve.txt'
"""
docdict[
    "recorder_fif_subdir"
] = """
fif_subdir : bool
    If ``True``, the ``.pcl`` files are converted to ``.fif`` in a
    subdirectory ``'fif': record_dir/fif/...`` instead of ``record_dir``."""
docdict[
    "recorder_verbose"
] = """
verbose : bool
    If ``True``, a timer showing since when the recorder started is displayed
    every seconds."""

# -----------------------------------------------
# Stream Viewer

# Not read by sphinx autodoc
docdict[
    "viewer_scope"
] = """
scope : Scope
    Scope connected to a StreamReceiver acquiring the data and applying
    filtering. The scope has a buffer of _BUFFER_DURATION seconds
    (default: 30s)."""
docdict[
    "viewer_backend_geometry"
] = """
geometry : tuple | list
    Window geometry as (pos_x, pos_y, size_x, size_y)."""
docdict[
    "viewer_backend_xRange"
] = """
xRange : int
    Range of the x-axis (plotting time duration) in seconds."""
docdict[
    "viewer_backend_yRange"
] = """
yRange : float
    Range of the y-axis (amplitude) in uV."""
docdict[
    "viewer_scope_stream_receiver"
] = """
stream_receiver : StreamReceiver
    Connected StreamReceiver."""
docdict[
    "viewer_scope_stream_name"
] = """
stream_name : str
    Stream to connect to."""
docdict[
    "viewer_event_type"
] = """
event_type : str
    Type of event. Supported: 'LPT'."""
docdict[
    "viewer_event_value"
] = """
event_value : int
    Value of the event."""
docdict[
    "viewer_position_buffer"
] = """
position_buffer : float
    Time (seconds) at which the event is positioned in the buffer where:
        0 represents the older events exiting the buffer.
        _BUFFER_DURATION represents the newer events entering the
        buffer."""
docdict[
    "viewer_position_plot"
] = """
position_plot : float
    Time (seconds) at which the event is positioned in the plotting window
    where:
        0 represents the older events exiting the window.
        xRange represents the newer events entering the window."""

# -----------------------------------------------
# Triggers
docdict[
    "trigger_verbose"
] = """
verbose : bool
    If ``True``, display a ``logger.info`` message when a trigger is sent."""

# ------------------------- Documentation functions --------------------------
docdict_indented: Dict[int, Dict[str, str]] = dict()


def fill_doc(f: Callable) -> Callable:
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of (modified in place).

    Returns
    -------
    f : callable
        The function, potentially with an updated __doc__.
    """
    docstring = f.__doc__
    if not docstring:
        return f

    lines = docstring.splitlines()
    indent_count = _indentcount_lines(lines)

    try:
        indented = docdict_indented[indent_count]
    except KeyError:
        indent = " " * indent_count
        docdict_indented[indent_count] = indented = dict()

        for name, docstr in docdict.items():
            lines = [
                indent + line if k != 0 else line
                for k, line in enumerate(docstr.strip().splitlines())
            ]
            indented[name] = "\n".join(lines)

    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")

    return f


def _indentcount_lines(lines: List[str]) -> int:
    """Minimum indent for all lines in line list.

    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    """
    indent = sys.maxsize
    for k, line in enumerate(lines):
        if k == 0:
            continue
        line_stripped = line.lstrip()
        if line_stripped:
            indent = min(indent, len(line) - len(line_stripped))
    return indent


def copy_doc(source: Callable) -> Callable:
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : callable
        The function to copy the docstring from.

    Returns
    -------
    wrapper : callable
        The decorated function.

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B(A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """

    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise RuntimeError(
                f"The docstring from {source.__name__} could not be copied "
                "because it was empty."
            )
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func

    return wrapper
