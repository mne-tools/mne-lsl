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
    None: no constraint."""

# -----------------------------------------------
# Triggers
docdict['trigger_verbose'] = """
verbose : bool
    If True, display a logger.info message when a trigger is sent."""
docdict['trigger_lpt_delay'] = """
delay : int
    Delay in milliseconds until which a new trigger cannot be sent."""
docdict['trigger_file'] = """
trigger_file : str | Path | None
    Path to the ini file containing the table converting event numbers into
    event strings."""

# -----------------------------------------------
# interfaces.audio
docdict['audio_volume'] = """
volume : list | int | float
    If an int or a float is provided, the sound will use only one channel
    (mono). If a 2-length sequence is provided, the sound will use 2 channels
    (stereo). Volume of each channel, given between 0 and 100. For stereo, the
    volume is given as [L, R]."""
docdict['audio_sample_rate'] = """
sample_rate : int, optional
    Sampling frequency of the sound. The default is 44100 kHz."""
docdict['audio_duration'] = """
duration : float, optional
    Duration of the sound. The default is 1.0 second."""

# -----------------------------------------------
# interfaces.visual
docdict['visual_window_name'] = """
window_name : str
    Name of the window in which the visual is displayed."""
docdict['visual_window_size'] = """
window_size : list | None
    Either None to automatically select a window size based on the available
    monitors, or a 2-length of positive integer sequence, as (width, height).
"""

# interfaces.visual color
color_var = 'color'
color_types = 'str | tuple'
color_format = 'as a matplotlib string or a (B, G, R) tuple of int8 set ' + \
               'between 0 and 255'
docdict['visual_color_background'] = f"""
{color_var} : {color_types}
    Color used to draw the background {color_format}."""
docdict['visual_color_text'] = f"""
{color_var} : {color_types}
    Color used to write the text {color_format}."""
docdict['visual_color_cross'] = f"""
{color_var} : {color_types}
    Color used to fill the cross {color_format}."""
docdict['visual_color_moving_bar'] = f"""
{color_var} : {color_types}
    Color used to fill the bar {color_format}."""
docdict['visual_color_filling_bar'] = f"""
{color_var} : {color_types}
    Color used to draw the bar background {color_format}."""
docdict['visual_fill_color_filling_bar'] = f"""
fill_color : {color_types}
    Color used to fill the bar {color_format}."""

# interfaces.visual dimension
var_types = 'int'

var = 'thickness'
base = f'Number of pixels used to draw the {var} of the'
docdict[f'visual_{var}_cross'] = f"""
{var} : {var_types}
    {base} cross."""

var = 'length'
base = f'Number of pixels used to draw the {var} of the'
docdict[f'visual_{var}_cross'] = f"""
{var} : {var_types}
    {base} cross."""
docdict[f'visual_{var}_bar'] = f"""
{var} : {var_types}
    {base} bar."""

var = 'width'
base = f'Number of pixels used to draw the {var} of the'
docdict[f'visual_{var}_bar'] = f"""
{var} : {var_types}
    {base} bar."""

# interfaces.visual position
inst = 'cross'
anchor = 'Position of the center of the cross'
docdict['visual_position_cross'] = f"""
position : str | list
    {anchor}.
    Either the string 'center' or 'centered' to position the {inst} in the
    center of the window; or a 2-length sequence of positive integer defining
    the {anchor.lower()} in the window. The position is defined in cv2
    coordinates, with (0, 0) being the top left corner of the window.
"""
inst = 'text'
anchor = 'Position of the bottom left corner of the text'
docdict['visual_position_text'] = f"""
position : str | list
    {anchor}.
    Either the string 'center' or 'centered' to position the {inst} in the
    center of the window; or a 2-length sequence of positive integer defining
    the {anchor.lower()} in the window. The position is defined in cv2
    coordinates, with (0, 0) being the top left corner of the window.
"""


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
