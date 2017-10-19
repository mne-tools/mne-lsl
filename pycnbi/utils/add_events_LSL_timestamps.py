from __future__ import print_function, division

"""
add_lsl_events.py

Add events recorded with LSL timestamps to raw data files.
Useful for software triggering.

Note:
Some OpenVibe acquisition servers send timestamps of their own running time (always
starting from 0) instead of LSL timestamps. In this case, the only way to deal with
this problem is to add an offset, a difference between LSL timestamp and OV server
time stamp.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

EVEDIR = 'D:/data/Records/'
OFFSET = -3251.4

import pycnbi.utils.q_common as qc
#import pycnbi.utils.pycnbi_utils as pu
from pycnbi.utils.convert2fif import pcl2fif
from builtins import input

to_process = []
print('Files to be processed')
for f in qc.get_file_list(EVEDIR, True):
    if f[-8:] == '-eve.txt':
        to_process.append(f)
        print(f)

input('\nPress Enter to start')
try:
    for f in to_process:
        pclfile = f.replace('-eve.txt', '-raw.pcl')
        pcl2fif(pclfile, external_event=f, offset=OFFSET)
except:
    print('\n*** Error occurred. Fix yourself.')
    import pdb

    pdb.set_trace()
