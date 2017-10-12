from __future__ import print_function, division

"""
Compute confusion matrix and accuracy from online result logs.

Kyuhwa Lee (kyu.lee@epfl.ch)
Swiss Federal Institute of Technology of Lausanne (EPFL)
"""

LOG_DIR = r'D:\data\MI\rx1\classifier\gait-ULR-250ms'

import pycnbi
import pycnbi.utils.q_common as qc

dtlist = []
gtlist = []
for f in qc.get_file_list(LOG_DIR):
    [basedir, fname, fext] = qc.parse_path_list(f)
    if 'online' not in fname or fext != 'txt':
        continue
    print(f)

    for l in open(f):
        if len(l.strip()) == 0: break
        gt, dt = l.strip().split(',')
        gtlist.append(gt)
        dtlist.append(dt)

print('Ground-truth: %s' % ''.join(gtlist))
print('Detected as : %s' % ''.join(dtlist))
cfmat, acc = qc.confusion_matrix(gtlist, dtlist)
print('\nAverage accuracy: %.3f' % acc)
print(cfmat)
