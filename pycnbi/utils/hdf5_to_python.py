from __future__ import print_function, division

"""
HDF5 to Python

Convert HDF5 to usable python numpy arrays

Created on Mon Oct 05 10:02:22 2015

@author: Hoang Pham, hoang.pham@epfl.ch
"""

import h5py
import scipy.io
import numpy as np
import pycnbi.utils.q_common as qc
import pycnbi.utils.pycnbi_utils as pu
import xml.etree.ElementTree as XET  # xml parser
from pycnbi import logger

def hdf5_to_python(data_dir):
    for rawfile in qc.get_file_list(data_dir, fullpath=True):
        # rawfile = 'D:/Hoang/My Documents/Python/artifact_data/arm_move2015.09.30_18.48.16.hdf5'
        # rawfile= 'D:/data/Artifact/eyeroll2015.09.30_18.38.23.hdf5'
        if rawfile.split('.')[-1] != 'hdf5': continue

        f = h5py.File(rawfile)

        # Reading xml properties
        """
        The HDF5 file is structured as folloging
        - AsynchronData
        |- AsynchronSignalType: XML
        |- Time: time index of the trigger data
        |- TypeID: corresponding trigger data, in a shifted 8-bit pattern (see below for explanation)
        |- Value: No idea !

        -RawData
        |- AcquisitionTaskDescription: XML, one node per channel
        |- DAQDeviceCapabilities: XML, one node per channel
        |- DAQDeviceDescription: XML, mostly useless
        |- Sample: Array of channel x time indexes
        |- SessionDescription: XML, mostly useless
        |- SubjectDescription: XML, mostly useless

        -SavedFeatures
        |- NumberOfFeatures: empty ?

        -Version
        |- Float, useless.
        """
        # Read properties from XML
        tree = XET.fromstring(f['RawData']['AcquisitionTaskDescription'][0])
        samplingFreq = int(tree.findall('SamplingFrequency')[0].text)  # todo: typecheck ?

        # Decode the trigger channel
        triggerDataRaw = f['AsynchronData']['TypeID'].value.ravel()  # Get the bit trigger data, flatten it
        bitoffset = min(triggerDataRaw)  # we're looking for the smallest value
        triggerDataExp = np.ravel(triggerDataRaw - bitoffset)

        timestamps = f['AsynchronData']['Time'].value.ravel()  # timestamps

        # We're looking at the indexes where there's a change
        timestampsOffset = np.insert(timestamps, 0, 0)  # offset by one
        timestamps = np.append(timestamps, 0)  # set to the same length
        diff_idx = np.ravel(np.nonzero(
            timestamps - timestampsOffset))  # non-zero element of this are the one we seek,[0] because it's stupid to have a nparray inside a tuple

        # Iterate each bit indexes and convert the 8-bits to decimal values
        triggerData = np.array([])
        for index, current in enumerate(diff_idx):
            tmp = 0
            if index < len(diff_idx) - 1:  # most elements
                for i in range(diff_idx[index], diff_idx[index + 1]):
                    tmp = tmp + 2 ** triggerDataExp[i]
                triggerData = np.append(triggerData, tmp)

        # Get the index, remove the last element to match the size of triggerData
        triggerIndexes = timestamps[diff_idx]
        triggerIndexes = np.delete(triggerIndexes, len(triggerIndexes) - 1)

        # triggerData and triggerIndexes are ready to be inputted to mne.create_event

        logger.info('%s\n%d events found. Event types: %s' % (rawfile, len(triggerIndexes), set(triggerData)))
        merged = np.vstack((triggerIndexes, triggerData)).T
        matfile = rawfile.replace('.hdf5', '.mat')
        matdata = dict(events=merged)
        scipy.io.savemat(matfile, matdata)
        logger.info('Data exported to %s' % matfile)

if __name__ == '__main__':
    data_dir = r'D:\data\TriggerTest'
    hdf5_to_python(data_dir)
