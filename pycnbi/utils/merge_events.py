from __future__ import print_function, division

"""
Merge different events

EVENTS = dict(LABEL_MERGED:[LABEL1, LABEL2, ...])

Kyuhwa Lee, 2015

"""

import numpy as np
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
from pycnbi.triggers.trigger_def import trigger_def
from pycnbi import logger

def merge_events(trigger_file, events, eeg_in, eeg_out):
    tdef = trigger_def(trigger_file)
    raw, eve = pu.load_raw(eeg_in)

    logger.info('\nEvents before merging')
    for key in np.unique(eve[:, 2]):
        logger.info('%s: %d' % (tdef.by_value[key], len(np.where(eve[:, 2] == key)[0])))

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
    assert max(eve[:, 2]) <= max(tdef.by_value.keys())

    # reset trigger channel
    raw._data[0] *= 0
    raw.add_events(eve, 'TRIGGER')
    raw.save(eeg_out, overwrite=True)

    logger.info('\nResulting events')
    for key in np.unique(eve[:, 2]):
        logger.info('%s: %d' % (tdef.by_value[key], len(np.where(eve[:, 2] == key)[0])))

if __name__ == '__main__':
    fif_dir = r'D:\data\STIMO\GO004\offline\all'
    trigger_file = 'triggerdef_gait_chuv.ini'
    events = {'BOTH_GO':['LEFT_GO', 'RIGHT_GO']}

    fiflist = []
    out_dir = fif_dir + '/merged'
    qc.make_dirs(out_dir)
    for f in qc.get_file_list(fif_dir):
        p = qc.parse_path(f)
        if p.ext != 'fif':
            continue
        eeg_in = f
        eeg_out = '%s/%s.%s' % (out_dir, p.name, p.ext)
        merge_events(trigger_file, events, eeg_in, eeg_out)
