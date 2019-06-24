from __future__ import print_function, division

from pycnbi import logger


def add_lsl_events(event_dir, offset=0, recursive=False, interactive=True):
    """
    Add events recorded with LSL timestamps to raw data files.
    Useful for software triggering.

    @params
    -------
    event_dir:
    Path to *-eve.txt files.

    offset:
    Timestamp offset (in seconds) in case the LSL server timestamps are shifted.
    Some OpenVibe acquisition servers send timestamps of their own running time (always
    starting from 0) instead of LSL timestamps. In this case, the only way to deal with
    this problem is to add an offset, a difference between LSL timestamp and OpenVibe
    server time stamp.

    recursive:
    Search sub-directories recursively.


    Kyuhwa Lee
    Swiss Federal Institute of Technology Lausanne (EPFL)
    2017
    """
    import pycnbi.utils.q_common as qc
    from pycnbi.utils.convert2fif import pcl2fif
    from builtins import input

    offset = float(offset)
    if offset != 0:
        logger.info_yellow('Time offset = %.3f' % offset)
    to_process = []
    logger.info('Files to be processed')
    if recursive:
        for d in qc.get_dir_list(event_dir):
            for f in qc.get_file_list(d, True):
                if f[-8:] == '-eve.txt':
                    to_process.append(f)
                    logger.info(f)
    else:
        for f in qc.get_file_list(event_dir, True):
            if f[-8:] == '-eve.txt':
                to_process.append(f)
                logger.info(f)

    if interactive:
        input('\nPress Enter to start')
    for f in to_process:
        pclfile = f.replace('-eve.txt', '-raw.pcl')
        pcl2fif(pclfile, external_event=f, offset=offset, overwrite=True)

# sample code
if __name__ == '__main__':
    event_dir = r'D:\data\Records'
    offset = -0.093936
    add_lsl_events(event_dir, offset=offset, recursive=False)
