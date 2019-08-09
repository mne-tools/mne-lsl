from __future__ import print_function, division

"""
Merge different events. Can be also used to simply change event values.

EVENTS = dict(LABEL_MERGED:[LABEL1, LABEL2, ...])

Kyuhwa Lee, 2015

"""

import numpy as np
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from pycnbi.triggers.trigger_def import trigger_def
from pycnbi import logger

def merge_events(trigger_file, events, rawfile_in, rawfile_out):
    tdef = trigger_def(trigger_file)
    raw, eve = pu.load_raw(rawfile_in)

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

# sample code
if __name__ == '__main__':
    fif_dir = r'D:\data\STIMO\GO004\offline\all'
    trigger_file = 'triggerdef_gait_chuv.ini'
    events = {'BOTH_GO':['LEFT_GO', 'RIGHT_GO']}

    out_dir = fif_dir + '/merged'
    qc.make_dirs(out_dir)
    for rawfile_in in qc.get_file_list(fif_dir):
        p = qc.parse_path(rawfile_in)
        if p.ext != 'fif':
            continue
        rawfile_out = '%s/%s.%s' % (out_dir, p.name, p.ext)
        merge_events(trigger_file, events, rawfile_in, rawfile_out)
