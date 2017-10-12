from __future__ import print_function, division

"""
Merge different events

EVENTS = dict(LABEL_MERGED:[LABEL1, LABEL2, ...])

Kyuhwa Lee, 2015
"""

EEG_IN = r'D:\data\MI\rx1\train\20161031-110045-raw.fif'
EVENTS = {'DOWN_GO':['LEFT_GO', 'RIGHT_GO']}
EEG_OUT = r'D:\data\MI\rx1\train\20161031-110045-D-raw.fif'
TRIGGER_DEF = 'triggerdef_gait.ini'


import pycnbi
import math, mne, csv
import numpy as np
import pycnbi.utils.q_common as qc
from pycnbi.triggers.trigger_def import trigger_def

tdef = trigger_def(TRIGGER_DEF)
raw, eve = pu.load_raw(EEG_IN)

print('\nEvents before merging')
for key in np.unique(eve[:, 2]):
    print('%s: %d' % (tdef.by_value[key], len(np.where(eve[:, 2] == key)[0])))

for key in EVENTS:
    ev_src = EVENTS[key]
    ev_out = tdef.by_key[key]
    x = []
    for e in ev_src:
        x.append(np.where(eve[:, 2] == tdef.by_key[e])[0])
    eve[np.concatenate(x), 2] = ev_out

# sanity check
dups = np.where(0 == np.diff(eve[:, 0]))[0]
assert len(dups) == 0
assert max(eve[:, 2]) <= max(tdef.by_value.keys())

# reset trigger channel
raw._data[0] *= 0
raw.add_events(eve, 'TRIGGER')
raw.save(EEG_OUT, overwrite=True)

print('\nResulting events')
for key in np.unique(eve[:, 2]):
    print('%s: %d' % (tdef.by_value[key], len(np.where(eve[:, 2] == key)[0])))
