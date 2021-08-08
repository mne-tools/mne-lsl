"""
Fill function docstrings to avoid redundant docstrings in multiple files.
Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""
import sys


# ------------------------- Documentation dictionary -------------------------
docdict = dict()

docdict['stream_name'] = """
stream_name : list | str | None
    Servers' name or list of servers' name to connect to.
    None: no constraint."""

# Triggers
docdict['trigger_verbose'] = """
verbose : bool
    If True, display a logger.info message when a trigger is sent."""
docdict['trigger_lpt_delay'] = """
delay : int
    Delay in milliseconds until which a new trigger cannot be sent."""

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

# interfaces.visual

# ------------------------- Documentation functions --------------------------
docdict_indented = {}


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
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
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
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno
