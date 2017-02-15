from __future__ import print_function, division

"""
add_lsl_events.py

Add events recorded with LSL timestamps to raw data files.

Kyuhwa Lee
Swiss Federal Institute of Technology (EPFL)

"""

EVEDIR = 'D:/data/Records/'

import q_common as qc
import pycnbi_config

to_process = []
print('Files to be processed')
for f in qc.get_file_list(EVEDIR, True):
    if f[-8:] == '-eve.txt':
        to_process.append(f)
        print(f)

raw_input('\nPress Enter to start')
try:
    for f in to_process:
        sigfile = f.replace('-eve.txt', '-raw.pcl')
        raw, events = pu.load_raw(sigfile, spfilter='car', events_ext=f)
except:
    print('\n*** Error occurred. Fix yourself.')
    import pdb

    pdb.set_trace()
