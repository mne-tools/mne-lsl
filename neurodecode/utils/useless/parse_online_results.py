from __future__ import print_function, division

from neurodecode.utils.io import get_file_list

"""
Compute confusion matrix and accuracy from online result logs.

Kyuhwa Lee (kyu.lee@epfl.ch)
Swiss Federal Institute of Technology of Lausanne (EPFL)
"""

LOG_DIR = r'D:\data\MI\rx1\classifier\gait-ULR-250ms'

from neurodecode.utils.io import parse_path
from neurodecode.utils.math import confusion_matrix

dtlist = []
gtlist = []
for f in get_file_list(LOG_DIR):
    p = parse_path(f)
    if 'online' not in p.name or p.ext != 'txt':
        continue
    print(f)

    for l in open(f):
        if len(l.strip()) == 0: break
        gt, dt = l.strip().split(',')
        gtlist.append(gt)
        dtlist.append(dt)

print('Ground-truth: %s' % ''.join(gtlist))
print('Detected as : %s' % ''.join(dtlist))
cfmat, acc = confusion_matrix(gtlist, dtlist)
print('\nAverage accuracy: %.3f' % acc)
print(cfmat)
