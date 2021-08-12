"""
Fill function docstrings to avoid redundant docstrings in multiple files.
Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""
import sys


# ------------------------- Documentation dictionary -------------------------
docdict = dict()

# -----------------------------------------------
docdict['stream_name'] = """
stream_name : list | str | None
    Servers' name or list of servers' name to connect to.
    ``None``: no constraint."""

# -----------------------------------------------
# Receiver
docdict['receiver_get_stream_name'] = """
stream_name : str | None
    Name of the stream to extract from.
    Can be set to ``None`` if the `StreamReceiver` is connected to a single
    stream."""
docdict['receiver_data'] = """
data : np.array
     Data ``[samples x channels]``."""
docdict['receiver_timestamps'] = """
timestamps : np.array
     Data's timestamps ``[samples]``."""
docdict['receiver_streamInfo'] = """
streamInfo : LSL StreamInfo.
    Contain all the info from the LSL stream to connect to."""
docdict['receiver_bufsize'] = """
bufsize : int | float
    Buffer's size [secs]. ``MAX_BUF_SIZE`` (def: 1-day) is the maximum size.
    Large buffer may lead to a delay if not pulled frequently."""
docdict['receiver_winsize'] = """
winsize : int | float
    Window's size [secs]. Must be smaller than the buffer's size."""
docdict['receiver_tslist'] = """
tslist : list
    Data's timestamps [samples]."""

# -----------------------------------------------
# Stream Recorder
docdict['recorder_record_dir'] = """
record_dir : str | Path
    Directory where the data will be saved."""
docdict['recorder_fname'] = """
fname : str | None
    File name stem used to create the files:
        PCL: ``'{fname}-[stream]-raw.pcl'``
        FIF: ``'{fname}-[stream]-raw.fif'``
        (optional) SOFTWARE trigger events: ``'{fname}-eve.txt'``"""
docdict['recorder_fif_subdir'] = """
fif_subdir : bool
    If ``True``, the ``.pcl`` files are converted to ``.fif`` in a
    subdirectory ``'fif': record_dir/fif/...`` instead of ``record_dir``."""
docdict['recorder_verbose'] = """
verbose : bool
    If ``True``, a timer showing since when the recorder started is displayed
    every seconds."""

# -----------------------------------------------
# Stream Player
docdict['player_stream_name'] = """
stream_name : str
    Stream's server name, displayed on LSL network."""
docdict['player_fif_file'] = """
fif_file : str | Path
    Path to the ``.fif`` file to play."""
docdict['player_chunk_size'] = """
chunk_size : int
    Number of samples to send at once (usually ``16-32`` is good enough)."""
docdict['player_repeat'] = """
repeat : int
    Number of times to replay the data (``default=inf``)."""
docdict['player_high_resolution'] = """
high_resolution : bool
    If ``True``, it uses ``perf_counter()`` instead of ``sleep()`` for higher
    time resolution. However, it uses more CPU."""

# -----------------------------------------------
# Stream Viewer
docdict['viewer_scope'] = """
scope : Scope
    Scope connected to a `StreamReceiver` acquiring the data and applying
    filtering. The scope has a buffer of ``_BUFFER_DURATION`` seconds
    (default: 30s)."""
docdict['viewer_backend_geometry'] = """
geometry : tuple | list
    Window geometry as ``(pos_x, pos_y, size_x, size_y)``."""
docdict['viewer_backend_xRange'] = """
xRange : int
    Range of the x-axis (plotting time duration) in seconds."""
docdict['viewer_backend_yRange'] = """yRange : float
    Range of the y-axis (amplitude) in uV."""
docdict['viewer_backend'] = """
backend : str
    One of the supported backend's name. Supported ``'vispy'``, ``'pyqt5'``."""
docdict['viewer_scope_stream_receiver'] = """
stream_receiver : StreamReceiver
    Connected `StreamReceiver`."""
docdict['viewer_scope_stream_name'] = """
stream_name : str
    Stream to connect to."""

# -----------------------------------------------
# Triggers
docdict['trigger_verbose'] = """
verbose : bool
    If ``True``, display a ``logger.info`` message when a trigger is sent."""
docdict['trigger_lpt_delay'] = """
delay : int
    Delay in milliseconds until which a new trigger cannot be sent."""
docdict['trigger_file'] = """
trigger_file : str | Path | None
    Path to the ``.ini`` file containing the table converting event numbers
    into event strings."""


# ------------------------- Documentation functions --------------------------
docdict_indented = dict()


def fill_doc(f):
    """
    Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f

    lines = docstring.splitlines()
    indent_count = _indentcount_lines(lines)

    try:
        indented = docdict_indented[indent_count]
    except KeyError:
        indent = ' ' * indent_count
        docdict_indented[indent_count] = indented = dict()

        for name, docstr in docdict.items():
            lines = [indent+line if k != 0 else line
                     for k, line in enumerate(docstr.strip().splitlines())]
            indented[name] = '\n'.join(lines)

    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))

    return f


def _indentcount_lines(lines):
    """
    Minimum indent for all lines in line list.

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
    for line in lines:
        line_stripped = line.lstrip()
        if line_stripped:
            indent = min(indent, len(line) - len(line_stripped))
    if indent == sys.maxsize:
        return 0
    return indent


def copy_doc(source):
    """
    Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated function

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
            raise ValueError('Cannot copy docstring: docstring was empty.')
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper
