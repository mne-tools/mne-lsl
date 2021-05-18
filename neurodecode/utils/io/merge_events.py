from __future__ import print_function, division

import numpy as np

from neurodecode import logger
from neurodecode.triggers import trigger_def
import neurodecode.utils.io as io

#----------------------------------------------------------------------
def merge_events(trigger_file, events, rawfile_in, rawfile_out):
    """
    Merge different events. Can be also used to simply change event values.
    
    Parameters
    ----------
    trigger_file : str
        The absolute path to the trigger file
    events : dict
        dict(LABEL_MERGED:[LABEL1, LABEL2, ...])
    rawfile_in : str
        The absolute path to the .fif file to modify
    rawfile_out : str
        The absolute path to save the new .fif file
    """
    tdef = trigger_def(trigger_file)
    raw, eve = io.load_fif_raw(rawfile_in)

    logger.info('=== Before merging ===')
    notfounds = []
    for key in np.unique(eve[:, 2]):
        if key in tdef.by_value:
            logger.info('%s: %d events' % (tdef.by_value[key], len(np.where(eve[:, 2] == key)[0])))
        else:
            logger.info('%d: %d events' % (key, len(np.where(eve[:, 2] == key)[0])))
            notfounds.append(key)
    if notfounds:
        for key in notfounds:
            logger.warning('Key %d was not found in the definition file.' % key)

    for key in events:
        ev_src = events[key]
        ev_out = tdef.by_name[key]
        x = []
        for e in ev_src:
            x.append(np.where(eve[:, 2] == tdef.by_name[e])[0])
        eve[np.concatenate(x), 2] = ev_out

    # sanity check
    dups = np.where(0 == np.diff(eve[:, 0]))[0]
    assert len(dups) == 0

    # reset trigger channel
    raw._data[0] *= 0
    raw.add_events(eve, 'TRIGGER')
    raw.save(rawfile_out, overwrite=True)

    logger.info('=== After merging ===')
    for key in np.unique(eve[:, 2]):
        if key in tdef.by_value:
            logger.info('%s: %d events' % (tdef.by_value[key], len(np.where(eve[:, 2] == key)[0])))
        else:
            logger.info('%s: %d events' % (key, len(np.where(eve[:, 2] == key)[0])))

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) > 5:
        raise IOError("Too many arguments provided. Max is 4: fif_dir; out_dir; trigger_file; events")
    
    if len(sys.argv) == 5:
        fif_dir = sys.argv[1]
        out_dir =  sys.argv[2]
        trigger_file = sys.argv[3]
        events =  sys.argv[4]
    
    if len(sys.argv) == 4:
        fif_dir = sys.argv[1]
        out_dir =  sys.argv[2]
        trigger_file = sys.argv[3]
        events =  dict(input("Provide the events to merge as follow: {LABEL_MERGED:[LABEL1, LABEL2, ...]} \n>> "))
    
    if len(sys.argv) == 3:
        fif_dir = sys.argv[1]
        out_dir =  sys.argv[2]
        trigger_file = input("Trigger definitions file: \n>> ")
        events =  dict(input("Provide the events to merge as follow: {LABEL_MERGED:[LABEL1, LABEL2, ...]} \n>> "))    
    
    if len(sys.argv) == 2:
        fif_dir = sys.argv[1]
        out_dir =  input("Ouput directory: \n>> ")
        trigger_file = input("Trigger definitions file: \n>> ")
        events =  dict(input("Provide the events to merge as follow: {LABEL_MERGED:[LABEL1, LABEL2, ...]} \n>> "))    

    if len(sys.argv) == 1:
        fif_dir = input("Directory with the fif files: \n>> ")
        out_dir =  input("Ouput directory: \n>> ")
        trigger_file = input("Trigger definitions file: \n>> ")
        events =  dict(input("Provide the events to merge as follow: {LABEL_MERGED:[LABEL1, LABEL2, ...]} \n>> "))

    io.make_dirs(out_dir)
    
    for rawfile_in in io.get_file_list(fif_dir):
        p = io.parse_path(rawfile_in)
        
        if p.ext != 'fif':
            continue
        
        rawfile_out = '%s/%s.%s' % (out_dir, p.name, p.ext)
        merge_events(trigger_file, events, rawfile_in, rawfile_out)
