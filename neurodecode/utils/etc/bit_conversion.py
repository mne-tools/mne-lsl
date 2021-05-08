
#----------------------------------------------------------------------
def int2bits(num, nbits=8):
    """
    Convert an integer into bits representation.
    
    Parameters
    ----------
    num : int
        The integer to convert
    nbits : int
        The bits number (default=8 bits (0-255))
        
    Returns
    -------
    list : The bits lit
    """
    return [int(num) >> x & 1 for x in range(nbits - 1, -1, -1)]

#----------------------------------------------------------------------
def bits2int(bitlist):
    """
    Convert a list of bits into an integer
    
    Parameters
    ----------
    bitlist : list
        The bits list to convert
        
    Returns
    -------
    int : The converted integer
    """
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out
