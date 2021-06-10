import itertools

from neurodecode import logger

#----------------------------------------------------------------------
def list2string(vec, fmt, sep=' '):
    """
    Convert a list to string with formatting, separated by sep (default is space).
    
    Example: fmt= '%.32e', '%.6f', etc.
    
    Parameters
    ----------
    vec : list
        The list to convert
    fmt : str
        The formating (e.g. '%.32e', '%.6f')
    sep : str
        The separator
    
    Returns:
    --------
    str : The list formated to string.
    """
    
    return sep.join(fmt % x for x in vec)

#----------------------------------------------------------------------
def flatten_list(l):
    """
    Flatten a n dimensions list to one dimension list
    
    Parameters
    ----------
    l : list
        The list to flatten
    
    Returns:
    --------
    list : The flattened list
    """
    return list(itertools.chain.from_iterable(l))

#----------------------------------------------------------------------
def get_index_min(seq):
    """
    Get the index of the minimum item in a list or dict
    
    Parameters
    ----------
    seq : list | dict
        The list or dict on which to find its min
    
    Returns:
    --------
    float | int :
        The minimum value of the list or dict
    """
    if type(seq) == list:
        return min(range(len(seq)), key=seq.__getitem__)
    
    elif type(seq) == dict:
        return min(seq, key=seq.__getitem__)
    
    else:
        logger.error('Unsupported input %s' % type(seq))
        return None

#----------------------------------------------------------------------
def get_index_max(seq):
    """
    Get the index of the maximum item in a list or dict
    
    Parameters
    ----------
    seq : list | dict
        The list or dict on which to find its max
    
    Returns:
    --------
    float | int :
        The maximum value of the list or dict
    """
    if type(seq) == list:
        return max(range(len(seq)), key=seq.__getitem__)
    
    elif type(seq) == dict:
        return max(seq, key=seq.__getitem__)
    
    else:
        logger.error('Unsupported input %s' % type(seq))
        return None

#----------------------------------------------------------------------
def sort_by_value(s, reverse=False):
    """
    Sort dictionary or list by value and return a sorted list of keys and values.
    
    Values must be hashable and unique.
    
    Parameters
    ----------
    s : list | dict
        The list or dict to sort
    reverse : bool
        If True, sorting in descending order
    
    Returns:
    --------
    keys : list
        The sorted list of keys (dict) or indices (list)
    values : list
        The sorted list of values
    """
    assert type(s) == dict or type(s) == list, 'Input must be a dictionary or list.'
    
    if type(s) == list:
        s = dict(enumerate(s))
    
    s_rev = dict((v, k) for k, v in s.items())
    
    if not len(s_rev) == len(s):
        logger.warning('sort_by_value(): %d identical values' % (len(s.values()) - len(set(s.values())) + 1))
    
    values = sorted(s_rev, reverse=reverse)
    keys = [s_rev[x] for x in values]
    
    return keys, values

#----------------------------------------------------------------------
def detect_delim(filename, allowSingleCol=True):
    """
    Automatically find the right delimiter of a file from the first line.

    Returns '' if the input file is single column or unknown format.
    
    Parameters
    ----------
    filename : str
        The absolute file path
    allowSingleCol : bool
        If False, it will raise an error in the above case.
        
    Returns:
    --------
    delim : str
        The found delimiter
    """
    
    with open(filename, 'r') as f:
        temp = f.readline().strip()
    delim = ''
    
    for d in [',', ' ', '\t']:
        if len(temp.split(d)) > 1:
            delim = d
            break
    else:
        if not allowSingleCol:
            raise Exception('Cannot detect the right delimiter')

    return delim
