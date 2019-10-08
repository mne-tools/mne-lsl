#!/usr/bin/env python

# 20141028: rchava

from pylab import *
import simplejson

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fname = sys.argv[1]
    else:
        print('[tRaSER] Usage: python convertIMU [filename]')
        sys.exit()

    data = np.load(fname)
    print(data.shape)

    data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    outfname = fname + ".json"
    outfile = open(outfname, "w")
    # simplejson.dump(buff, outfile, separators=(',',';'))
    outfile.close()
