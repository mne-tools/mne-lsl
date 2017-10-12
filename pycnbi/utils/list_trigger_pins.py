from __future__ import print_function, division

"""
List possible trigger values assuming some pins are dead.

For example, current cascading cable for AntNeuro has dead pins of 2,4,7.

Kyuhwa Lee, 2015.

"""

# Set dead bits (bit 0 is the LSB)
DEAD_BITS = [2, 4, 7]
BIT_LENGTH = 8

import numpy as np

assert max(DEAD_BITS) < BIT_LENGTH
x = ['1'] * BIT_LENGTH
for i in DEAD_BITS:
    x[-1 - i] = '0'

dead = int(''.join(x), 2)  # e.g. 01101011 = 107
triggers = set()
for t in range(2 ** BIT_LENGTH):
    triggers.add(t & dead)

print('Total %d trigger values are possible.' % len(triggers))
for t in sorted(triggers):
    print('%s (%d)' % (np.binary_repr(t, width=BIT_LENGTH), t))
