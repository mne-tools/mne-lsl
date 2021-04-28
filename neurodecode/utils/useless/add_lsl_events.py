from __future__ import print_function, division

from neurodecode import logger
from neurodecode.utils.io import get_dir_list, get_file_list

#----------------------------------------------------------------------
def add_lsl_events(event_dir, recursive=False, interactive=True):
    """
    Add LSL events to raw data. Data must be pcl file to be converted to fif format.
    
    Useful for software trigger. The data in pickle and the event in txt must be in the same folder.

    Parameters
    ----------
    event_dir : str
        Path to the folder containing both *-eve.txt and *-raw.pcl
    recursive : bool
        Search sub-directories recursively
    interactive : bool
        If true, press Enter will be asked to convert the found files
    """
    import neurodecode.utils.q_common as qc
    from neurodecode.utils.io import pcl2fif
    from builtins import input

    to_process = []
    logger.info('Files to be processed')
    if recursive:
        for d in get_dir_list(event_dir):
            for f in get_file_list(d, True):
                if f[-8:] == '-eve.txt':
                    to_process.append(f)
                    logger.info(f)
    else:
        for f in get_file_list(event_dir, True):
            if f[-8:] == '-eve.txt':
                to_process.append(f)
                logger.info(f)

    if interactive:
        input('\nPress Enter to start')
    for f in to_process:
        pclfile = f.replace('-eve.txt', '-raw.pcl')
        pcl2fif(pclfile, external_event=f)

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    from pathlib import Path
    
    event_dir = str(Path(input(">> Provide the path to the folder containing the events files: \n")))
    
    recursive = input(">> Do you search recursively (subfolders)? (y / n) \n")
    recursive = True if recursive in ['y', 'yes', 'Y', 'YES', 'Yes'] else False
    
    add_lsl_events(event_dir, recursive=recursive, interactive=True)
